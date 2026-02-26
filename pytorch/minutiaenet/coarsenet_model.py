import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
import os
from .mnet_utils import get_minutiaenet_logger, MnetTimer, logging, DEFAULT_DEVICE, DEFAULT_COARSENET_WEIGHTS_PATH

logger = get_minutiaenet_logger(__name__, level=logging.INFO)


class ImgNormalization(nn.Module):
    def __init__(self, m0=0.0, var0=1.0):
        super().__init__()
        self.m0 = m0
        self.var0 = var0

    def forward(self, x):
        m = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        var = torch.var(x, dim=(1, 2, 3), keepdim=True)
        after = torch.sqrt(self.var0 * torch.square(x - m) / (var + 1e-8))
        return torch.where(x > m, self.m0 + after, self.m0 - after)


class ConvBNPReLU(nn.Module):
    """Conv2D -> BatchNorm -> PReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.99)
        # shared_axes=[1,2] in Keras PReLU means one alpha per channel
        self.prelu = nn.PReLU(num_parameters=out_channels, init=0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class FeatureExtractor(nn.Module):
    """Backbone encoder with residual blocks.

    Matches the original CoarseNet architecture:
    - Initial conv block: 5x5 + two 3x3 convs (64 filters) + maxpool
    - Block 1: Three residual units (128 filters, 3x3) + maxpool
    - Block 2: Two residual units (256 filters, 3x3) + maxpool
    - Block 3: One residual unit (512 filters, 3x3) + 1x1 projection to 256
    """
    def __init__(self):
        super().__init__()
        # Initial conv block
        self.conv1_0 = ConvBNPReLU(1, 64, 5)
        self.conv1_1 = ConvBNPReLU(64, 64, 3)
        self.conv1_2 = ConvBNPReLU(64, 64, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 1: 3 residual units with 128 filters
        self.conv2_1 = ConvBNPReLU(64, 128, 3)
        self.conv2_2 = ConvBNPReLU(128, 128, 3)
        self.conv2_3 = ConvBNPReLU(128, 128, 3)
        self.conv2_1b = ConvBNPReLU(128, 128, 3)
        self.conv2_2b = ConvBNPReLU(128, 128, 3)
        self.conv2_3b = ConvBNPReLU(128, 128, 3)
        self.conv2_1c = ConvBNPReLU(128, 128, 3)
        self.conv2_2c = ConvBNPReLU(128, 128, 3)
        self.conv2_3c = ConvBNPReLU(128, 128, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 2 residual units with 256 filters
        self.conv3_1 = ConvBNPReLU(128, 256, 3)
        self.conv3_2 = ConvBNPReLU(256, 256, 3)
        self.conv3_3 = ConvBNPReLU(256, 256, 3)
        self.conv3_1b = ConvBNPReLU(256, 256, 3)
        self.conv3_2b = ConvBNPReLU(256, 256, 3)
        self.conv3_3b = ConvBNPReLU(256, 256, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 1 residual unit with 512 filters + projection
        self.conv3_1c = ConvBNPReLU(256, 512, 3)
        self.conv3_2c = ConvBNPReLU(512, 512, 3)
        self.conv3_3c = ConvBNPReLU(512, 512, 3)
        self.conv3_4c = ConvBNPReLU(512, 256, 3)

    def forward(self, x):
        # Initial block
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        # Block 1
        res = self.conv2_1(x)
        x = self.conv2_2(res)
        x = self.conv2_3(x)
        x = x + res

        res = self.conv2_1b(x)
        x = self.conv2_2b(res)
        x = self.conv2_3b(x)
        x = x + res

        res = self.conv2_1c(x)
        x = self.conv2_2c(res)
        x = self.conv2_3c(x)
        x = x + res

        conv_block1 = self.pool2(x)

        # Block 2
        res = self.conv3_1(conv_block1)
        x = self.conv3_2(res)
        x = self.conv3_3(x)
        x = x + res

        res = self.conv3_1b(x)
        x = self.conv3_2b(res)
        x = self.conv3_3b(x)
        x = x + res

        conv_block2 = self.pool3(x)

        # Block 3 (no additional pooling)
        res = self.conv3_1c(conv_block2)
        x = self.conv3_2c(res)
        x = self.conv3_3c(x)
        x = x + res
        conv_block3 = self.conv3_4c(x)

        return conv_block2, conv_block3


class OrientationSegmentationHead(nn.Module):
    """Multi-scale ASPP for orientation and segmentation prediction.

    The original uses 3 atrous levels:
    - Level 1: from conv_block3 (dilation=1)
    - Level 2: from conv_block2 (dilation=4)
    - Level 3: from conv_block2 (dilation=8)
    Each produces orientation (90 channels) and segmentation (1 channel) outputs.
    """
    def __init__(self):
        super().__init__()
        # Level 1: dilation=1 from conv_block3 (256 channels)
        self.atrous_1 = ConvBNPReLU(256, 256, 3, dilation=1)
        self.ori_branch_1 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 90, 1))
        self.seg_branch_1 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 1, 1))

        # Level 2: dilation=4 from conv_block2 (256 channels)
        self.atrous_2 = ConvBNPReLU(256, 256, 3, dilation=4)
        self.ori_branch_2 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 90, 1))
        self.seg_branch_2 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 1, 1))

        # Level 3: dilation=8 from conv_block2 (256 channels)
        self.atrous_3 = ConvBNPReLU(256, 256, 3, dilation=8)
        self.ori_branch_3 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 90, 1))
        self.seg_branch_3 = nn.Sequential(ConvBNPReLU(256, 128, 1), nn.Conv2d(128, 1, 1))

    def forward(self, conv_block2, conv_block3):
        # Level 1 from conv_block3
        feat1 = self.atrous_1(conv_block3)
        o1 = self.ori_branch_1(feat1)
        s1 = self.seg_branch_1(feat1)

        # Level 2 from conv_block2
        feat2 = self.atrous_2(conv_block2)
        o2 = self.ori_branch_2(feat2)
        s2 = self.seg_branch_2(feat2)

        # Level 3 from conv_block2
        feat3 = self.atrous_3(conv_block2)
        o3 = self.ori_branch_3(feat3)
        s3 = self.seg_branch_3(feat3)

        # Sum fusion
        ori_out = torch.sigmoid(o1 + o2 + o3)
        seg_out = torch.sigmoid(s1 + s2 + s3)

        return ori_out, seg_out





class EnhancementModule(nn.Module):
    """Gabor filtering and orientation-guided enhancement."""
    def __init__(self):
        super().__init__()
        self.gabor_real = nn.Conv2d(1, 90, 25, padding='same', bias=True)
        self.gabor_imag = nn.Conv2d(1, 90, 25, padding='same', bias=True)

        # Pre-compute circular gaussian kernel for orientation peak detection
        length = 180
        stride = 2
        std = 3
        gaussian_pdf = signal.windows.gaussian(length + 1, std=std)
        y = np.reshape(np.arange(stride / 2, length, stride), [1, 1, -1, 1])
        label = np.reshape(np.arange(stride / 2, length, stride), [1, 1, 1, -1])
        delta = np.array(np.abs(label - y), dtype=int)
        delta = np.minimum(delta, length - delta) + length // 2
        glabel = gaussian_pdf[delta].astype(np.float32)
        self.register_buffer('glabel_tensor', torch.from_numpy(glabel).permute(2, 3, 0, 1))



    def _ori_highest_peak(self, y_pred):
        return F.conv2d(y_pred, self.glabel_tensor, padding='same')

    def _select_max_orientation(self, ori_map):
        max_vals, _ = torch.max(ori_map, dim=1, keepdim=True)
        x = ori_map / (max_vals + 1e-8)
        x = torch.where(x > 0.999, x, torch.zeros_like(x))
        return x / (torch.sum(x, dim=1, keepdim=True) + 1e-8)

    def _atan2(self, y, x):
        angle = torch.atan(y / (x + 1e-8))
        angle = torch.where(x > 0, angle, torch.zeros_like(x))
        angle = torch.where((x < 0) & (y >= 0), angle + np.pi, angle)
        angle = torch.where((x < 0) & (y < 0), angle - np.pi, angle)
        return angle

    def forward(self, original_image, ori_map):
        filtered_real = self.gabor_real(original_image)
        filtered_imag = self.gabor_imag(original_image)

        ori_map = self._ori_highest_peak(ori_map)
        ori_peak = self._select_max_orientation(ori_map)
        upsampled_ori = F.interpolate(ori_peak, scale_factor=8, mode='nearest')

        enh_real = torch.sum(filtered_real * upsampled_ori, dim=1, keepdim=True)
        enh_imag = torch.sum(filtered_imag * upsampled_ori, dim=1, keepdim=True)
        enhanced_phase = self._atan2(enh_imag, enh_real)

        return enh_real, enhanced_phase, upsampled_ori


class MinutiaeHead(nn.Module):
    """Minutiae detection head with residual blocks.

    Matches the original architecture:
    - Block 1: Two residual units (64 filters, 9x9) + maxpool
    - Block 2: Two residual units (128 filters, 5x5) + maxpool
    - Block 3: Dense residual unit (256 filters, 3x3) + maxpool
    - Four output branches: orientation (180), x_offset (8), y_offset (8), score (1)
    """
    def __init__(self):
        super().__init__()
        # Block 1: 64 filters, 9x9 kernels, 2 residual units
        self.mnt_1_1 = ConvBNPReLU(2, 64, 9)
        self.mnt_1_2 = ConvBNPReLU(64, 64, 9)
        self.mnt_1_3 = ConvBNPReLU(64, 64, 9)
        self.mnt_1_1b = ConvBNPReLU(64, 64, 9)
        self.mnt_1_2b = ConvBNPReLU(64, 64, 9)
        self.mnt_1_3b = ConvBNPReLU(64, 64, 9)
        self.mnt_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 128 filters, 5x5 kernels, 2 residual units
        self.mnt_2_1 = ConvBNPReLU(64, 128, 5)
        self.mnt_2_2 = ConvBNPReLU(128, 128, 5)
        self.mnt_2_3 = ConvBNPReLU(128, 128, 5)
        self.mnt_2_1b = ConvBNPReLU(128, 128, 5)
        self.mnt_2_2b = ConvBNPReLU(128, 128, 5)
        self.mnt_2_3b = ConvBNPReLU(128, 128, 5)
        self.mnt_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 256 filters, 3x3 kernels, dense residual
        self.mnt_3_1 = ConvBNPReLU(128, 256, 3)
        self.mnt_3_2 = ConvBNPReLU(256, 256, 3)
        self.mnt_3_3 = ConvBNPReLU(256, 256, 3)
        self.mnt_3_4 = ConvBNPReLU(256, 256, 3)
        self.mnt_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output branches
        # Orientation: concat with ori_out (90 channels) -> 256+90=346 input
        self.mnt_o_1_1 = ConvBNPReLU(346, 256, 1)
        self.mnt_o_1_2 = nn.Conv2d(256, 180, 1)

        # Width (x offset)
        self.mnt_w_1_1 = ConvBNPReLU(256, 256, 1)
        self.mnt_w_1_2 = nn.Conv2d(256, 8, 1)

        # Height (y offset)
        self.mnt_h_1_1 = ConvBNPReLU(256, 256, 1)
        self.mnt_h_1_2 = nn.Conv2d(256, 8, 1)

        # Score
        self.mnt_s_1_1 = ConvBNPReLU(256, 256, 1)
        self.mnt_s_1_2 = nn.Conv2d(256, 1, 1)

    def forward(self, enh_seg_input, ori_features):
        # Block 1
        res = self.mnt_1_1(enh_seg_input)
        x = self.mnt_1_2(res)
        x = self.mnt_1_3(x)
        x = x + res

        res = self.mnt_1_1b(x)
        x = self.mnt_1_2b(res)
        x = self.mnt_1_3b(x)
        x = x + res

        x = self.mnt_pool1(x)

        # Block 2
        res = self.mnt_2_1(x)
        x = self.mnt_2_2(res)
        x = self.mnt_2_3(x)
        x = x + res

        res = self.mnt_2_1b(x)
        x = self.mnt_2_2b(res)
        x = self.mnt_2_3b(x)
        x = x + res

        x = self.mnt_pool2(x)

        # Block 3 (dense residual connections)
        conv1 = self.mnt_3_1(x)
        conv2 = self.mnt_3_2(conv1)
        conv3 = self.mnt_3_3(conv2)
        conv3 = conv3 + conv1
        conv4 = self.mnt_3_4(conv3)
        conv4 = conv4 + conv2

        mnt_features = self.mnt_pool3(conv4)

        # Orientation branch (concatenates with orientation features)
        mnt_o_input = torch.cat([mnt_features, ori_features], dim=1)
        mnt_o = torch.sigmoid(self.mnt_o_1_2(self.mnt_o_1_1(mnt_o_input)))

        # Width branch
        mnt_w = torch.sigmoid(self.mnt_w_1_2(self.mnt_w_1_1(mnt_features)))

        # Height branch
        mnt_h = torch.sigmoid(self.mnt_h_1_2(self.mnt_h_1_1(mnt_features)))

        # Score branch
        mnt_s = torch.sigmoid(self.mnt_s_1_2(self.mnt_s_1_1(mnt_features)))

        return mnt_o, mnt_w, mnt_h, mnt_s


class CoarseNet(nn.Module):
    """Complete CoarseNet model orchestrating the data flow between blocks."""
    def __init__(self):
        super().__init__()
        self.img_norm = ImgNormalization()
        self.feature_extractor = FeatureExtractor()
        self.ori_seg_head = OrientationSegmentationHead()
        self.enhancement_module = EnhancementModule()
        self.minutiae_head = MinutiaeHead()

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the segmentation map."""
        x_norm = self.img_norm(x)
        conv_block2, conv_block3 = self.feature_extractor(x_norm)
        _, seg_map = self.ori_seg_head(conv_block2, conv_block3)
        upsampled_seg = F.interpolate(
            nn.functional.softsign(seg_map), scale_factor=8, mode='nearest'
        )
        return upsampled_seg

    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the enhanced image."""
        x_norm = self.img_norm(x)
        conv_block2, conv_block3 = self.feature_extractor(x_norm)
        ori_map, _ = self.ori_seg_head(conv_block2, conv_block3)
        enh_real, _, _ = self.enhancement_module(x, ori_map)
        return enh_real

    def forward(self, x: torch.Tensor):
        """Full forward pass returning dict of all outputs."""
        x_norm = self.img_norm(x)
        conv_block2, conv_block3 = self.feature_extractor(x_norm)

        ori_map, seg_map = self.ori_seg_head(conv_block2, conv_block3)

        enh_real, enh_phase, upsampled_ori_map = self.enhancement_module(x, ori_map)

        upsampled_seg = F.interpolate(
            nn.functional.softsign(seg_map), scale_factor=8, mode='nearest'
        )
        upsampled_seg_out = F.interpolate(seg_map, scale_factor=8, mode='nearest')

        # Minutiae input: phase image + segmentation
        minutiae_input = torch.cat([enh_phase, upsampled_seg], dim=1)

        mnt_o, mnt_w, mnt_h, mnt_s = self.minutiae_head(minutiae_input, ori_map)

        return {
            'orientation upsample': upsampled_ori_map,
            'segmentation upsample': upsampled_seg_out,
            'segmentation': seg_map,
            'orientation': ori_map,
            'enhanced_real': enh_real,
            'enhanced_phase': enh_phase,
            'minutiae_orientation': mnt_o,
            'minutiae_x_offset': mnt_w,
            'minutiae_y_offset': mnt_h,
            'minutiae_score': mnt_s,
        }

    def time(self, x: torch.Tensor):
        """Full forward with timing instrumentation."""
        x_norm = self.img_norm(x)

        with MnetTimer("Feature Extraction", logger):
            conv_block2, conv_block3 = self.feature_extractor(x_norm)

        with MnetTimer("Orientation and Segmentation Head", logger):
            ori_map, seg_map = self.ori_seg_head(conv_block2, conv_block3)

        with MnetTimer("Enhancement Module", logger):
            enh_real, enh_phase, upsampled_ori_map = self.enhancement_module(x, ori_map)

        upsampled_seg = F.interpolate(
            nn.functional.softsign(seg_map), scale_factor=8, mode='nearest'
        )
        upsampled_seg_out = F.interpolate(seg_map, scale_factor=8, mode='nearest')

        minutiae_input = torch.cat([enh_phase, upsampled_seg], dim=1)

        with MnetTimer("Minutiae Head", logger):
            mnt_o, mnt_w, mnt_h, mnt_s = self.minutiae_head(minutiae_input, ori_map)

        return {
            'orientation upsample': upsampled_ori_map,
            'segmentation upsample': upsampled_seg_out,
            'segmentation': seg_map,
            'orientation': ori_map,
            'enhanced_real': enh_real,
            'enhanced_phase': enh_phase,
            'minutiae_orientation': mnt_o,
            'minutiae_x_offset': mnt_w,
            'minutiae_y_offset': mnt_h,
            'minutiae_score': mnt_s,
        }


def get_coarsenet_core(
    weights_path: str = DEFAULT_COARSENET_WEIGHTS_PATH,
    device: str = DEFAULT_DEVICE
) -> CoarseNet:
    """Load CoarseNet with pre-trained weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    logger.info(f"Selected device: {device}")
    logger.info("Loading CoarseNet architecture...")
    model = CoarseNet()

    logger.info(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.eval()
    model.to(device)

    if device == "cpu":
        logger.info("Moving model to channels_last memory format...")
        model.to(memory_format=torch.channels_last)

    logger.info("CoarseNet ready for inference.")
    return model
