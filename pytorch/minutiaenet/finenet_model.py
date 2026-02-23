import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .mnet_utils import get_minutiaenet_logger, logging, DEFAULT_DEVICE, DEFAULT_FINENET_WEIGHTS_PATH

logger = get_minutiaenet_logger(__name__, level=logging.INFO)


class ConvBNReLU(nn.Module):
    """Conv2D -> BatchNorm -> ReLU (standard block for FineNet/InceptionResNet)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', use_bias=False):
        super().__init__()
        if padding == 'same':
            if isinstance(kernel_size, int):
                pad = kernel_size // 2
            else:
                pad = tuple(k // 2 for k in kernel_size)
        elif padding == 'valid':
            pad = 0
        else:
            pad = padding

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=pad, bias=use_bias
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.001) if not use_bias else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionResNetBlock(nn.Module):
    """Inception-ResNet block (block35, block17, or block8)."""
    def __init__(self, block_type, in_channels, scale=1.0, activation=True):
        super().__init__()
        self.scale = scale
        self.activation = nn.ReLU(inplace=True) if activation else None

        if block_type == 'block35':
            # Branch 0
            self.branch0 = ConvBNReLU(in_channels, 32, 1)
            # Branch 1
            self.branch1 = nn.Sequential(
                ConvBNReLU(in_channels, 32, 1),
                ConvBNReLU(32, 32, 3),
            )
            # Branch 2
            self.branch2 = nn.Sequential(
                ConvBNReLU(in_channels, 32, 1),
                ConvBNReLU(32, 48, 3),
                ConvBNReLU(48, 64, 3),
            )
            mixed_channels = 32 + 32 + 64  # 128
            self.up = nn.Conv2d(mixed_channels, in_channels, 1, bias=True)

        elif block_type == 'block17':
            self.branch0 = ConvBNReLU(in_channels, 192, 1)
            self.branch1 = nn.Sequential(
                ConvBNReLU(in_channels, 128, 1),
                ConvBNReLU(128, 160, (1, 7)),
                ConvBNReLU(160, 192, (7, 1)),
            )
            mixed_channels = 192 + 192  # 384
            self.up = nn.Conv2d(mixed_channels, in_channels, 1, bias=True)

        elif block_type == 'block8':
            self.branch0 = ConvBNReLU(in_channels, 192, 1)
            self.branch1 = nn.Sequential(
                ConvBNReLU(in_channels, 192, 1),
                ConvBNReLU(192, 224, (1, 3)),
                ConvBNReLU(224, 256, (3, 1)),
            )
            mixed_channels = 192 + 256  # 448
            self.up = nn.Conv2d(mixed_channels, in_channels, 1, bias=True)

        self.block_type = block_type

    def forward(self, x):
        if self.block_type == 'block35':
            branches = [self.branch0(x), self.branch1(x), self.branch2(x)]
        else:
            branches = [self.branch0(x), self.branch1(x)]

        mixed = torch.cat(branches, dim=1)
        up = self.up(mixed)
        out = x + up * self.scale

        if self.activation is not None:
            out = self.activation(out)
        return out


class FineNet(nn.Module):
    """FineNet: Lightweight Inception-ResNet for minutiae verification.

    Input: (B, 3, 224, 224) RGB patches
    Output: (B, 2) softmax probabilities [minutiae, non-minutiae]
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # Stem block: 224 -> 35x35x192
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, 3, stride=2, padding='valid'),
            ConvBNReLU(32, 32, 3, padding='valid'),
            ConvBNReLU(32, 64, 3),
            nn.MaxPool2d(3, stride=2),
            ConvBNReLU(64, 80, 1, padding='valid'),
            ConvBNReLU(80, 192, 3, padding='valid'),
            nn.MaxPool2d(3, stride=2),
        )

        # Mixed 5b (Inception-A): 35x35x320
        self.mixed_5b_branch0 = ConvBNReLU(192, 96, 1)
        self.mixed_5b_branch1 = nn.Sequential(
            ConvBNReLU(192, 48, 1),
            ConvBNReLU(48, 64, 5),
        )
        self.mixed_5b_branch2 = nn.Sequential(
            ConvBNReLU(192, 64, 1),
            ConvBNReLU(64, 96, 3),
            ConvBNReLU(96, 96, 3),
        )
        self.mixed_5b_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBNReLU(192, 64, 1),
        )
        # 96 + 64 + 96 + 64 = 320

        # 10x block35: 35x35x320
        self.block35 = nn.Sequential(*[
            InceptionResNetBlock('block35', 320, scale=0.17)
            for _ in range(10)
        ])

        # Mixed 6a (Reduction-A): 35x35x320 -> 17x17x1088
        self.mixed_6a_branch0 = ConvBNReLU(320, 384, 3, stride=2, padding='valid')
        self.mixed_6a_branch1 = nn.Sequential(
            ConvBNReLU(320, 256, 1),
            ConvBNReLU(256, 256, 3),
            ConvBNReLU(256, 384, 3, stride=2, padding='valid'),
        )
        self.mixed_6a_pool = nn.MaxPool2d(3, stride=2)
        # 384 + 384 + 320 = 1088

        # 20x block17: 17x17x1088
        self.block17 = nn.Sequential(*[
            InceptionResNetBlock('block17', 1088, scale=0.1)
            for _ in range(20)
        ])

        # Mixed 7a (Reduction-B): 17x17x1088 -> 8x8x2080
        self.mixed_7a_branch0 = nn.Sequential(
            ConvBNReLU(1088, 256, 1),
            ConvBNReLU(256, 384, 3, stride=2, padding='valid'),
        )
        self.mixed_7a_branch1 = nn.Sequential(
            ConvBNReLU(1088, 256, 1),
            ConvBNReLU(256, 288, 3, stride=2, padding='valid'),
        )
        self.mixed_7a_branch2 = nn.Sequential(
            ConvBNReLU(1088, 256, 1),
            ConvBNReLU(256, 288, 3),
            ConvBNReLU(288, 320, 3, stride=2, padding='valid'),
        )
        self.mixed_7a_pool = nn.MaxPool2d(3, stride=2)
        # 384 + 288 + 320 + 1088 = 2080

        # 9x block8 (scale=0.2) + 1x block8 (scale=1.0, no activation)
        block8_list = [
            InceptionResNetBlock('block8', 2080, scale=0.2)
            for _ in range(9)
        ]
        block8_list.append(InceptionResNetBlock('block8', 2080, scale=1.0, activation=False))
        self.block8 = nn.Sequential(*block8_list)

        # Final conv: 1536
        self.conv_7b = ConvBNReLU(2080, 1536, 1)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Mixed 5b
        branches = [
            self.mixed_5b_branch0(x),
            self.mixed_5b_branch1(x),
            self.mixed_5b_branch2(x),
            self.mixed_5b_pool(x),
        ]
        x = torch.cat(branches, dim=1)

        # 10x block35
        x = self.block35(x)

        # Mixed 6a
        branches = [
            self.mixed_6a_branch0(x),
            self.mixed_6a_branch1(x),
            self.mixed_6a_pool(x),
        ]
        x = torch.cat(branches, dim=1)

        # 20x block17
        x = self.block17(x)

        # Mixed 7a
        branches = [
            self.mixed_7a_branch0(x),
            self.mixed_7a_branch1(x),
            self.mixed_7a_branch2(x),
            self.mixed_7a_pool(x),
        ]
        x = torch.cat(branches, dim=1)

        # 10x block8
        x = self.block8(x)

        # Final conv + pooling + classifier
        x = self.conv_7b(x)
        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)

        return x


def get_finenet_core(
    weights_path: str = DEFAULT_FINENET_WEIGHTS_PATH,
    device: str = DEFAULT_DEVICE
) -> FineNet:
    """Load FineNet with pre-trained weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    logger.info(f"Selected device: {device}")
    logger.info("Loading FineNet architecture...")
    model = FineNet()

    logger.info(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)

    logger.info("FineNet ready for inference.")
    return model
