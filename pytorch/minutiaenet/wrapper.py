import torch
from torch import nn
import torch.nn.functional as F
from .coarsenet_model import CoarseNet
from .finenet_model import FineNet
from .mnet_utils import (
    get_minutiaenet_logger, MnetTimer, logging,
    DEFAULT_COARSENET_WEIGHTS_PATH, DEFAULT_FINENET_WEIGHTS_PATH, DEFAULT_DEVICE,
)
import os
import numpy as np
import time

logger = get_minutiaenet_logger(__name__, level=logging.INFO)


class MinutiaeNetWrapper(nn.Module):
    """Wrapper around CoarseNet (+ optional FineNet) with pre/post-processing."""

    def __init__(self, coarsenet: CoarseNet, finenet: FineNet | None = None):
        super().__init__()
        self.coarsenet = coarsenet
        self.finenet = finenet

    def forward(
        self, x: torch.Tensor,
        minutiae_threshold: float = 0.45,
        use_finenet: bool = False,
        quality_mask: bool = False,
        unmodulated: bool = False,
    ) -> dict[str, torch.Tensor]:

        padded_x = self.preprocess(x)

        with torch.no_grad():
            raw_outputs = self.coarsenet(padded_x)

        post_x = self.postprocess(
            raw_outputs, minutiae_threshold,
            quality_mask=quality_mask, unmodulated=unmodulated,
        )

        if use_finenet and self.finenet is not None:
            post_x = self._apply_finenet(padded_x, post_x, minutiae_threshold)

        return post_x

    def time(
        self, x: torch.Tensor,
        minutiae_threshold: float = 0.5,
        quality_mask: bool = False,
        unmodulated: bool = False,
    ) -> dict[str, torch.Tensor]:
        padded_x = self.preprocess(x)

        with torch.no_grad():
            with MnetTimer("Full Inference", logger):
                raw_outputs = self.coarsenet.time(padded_x)

        with MnetTimer("Post-processing", logger):
            post_x = self.postprocess(
                raw_outputs, minutiae_threshold,
                quality_mask=quality_mask, unmodulated=unmodulated,
            )

        return post_x

    def prepare_input(self, x: np.ndarray) -> torch.Tensor:
        """Converts a numpy image to a torch tensor suitable for the model."""
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
            x = np.expand_dims(x, axis=0)
        if x.ndim == 3:
            if x.shape[0] > 1:
                x = np.expand_dims(x, axis=1)
            else:
                x = np.expand_dims(x, axis=0)
        if x.ndim == 4:
            tensor_x = torch.tensor(x, dtype=torch.float32)
        else:
            raise ValueError("Input numpy array must be 2D, 3D, or 4D.")

        device = next(self.coarsenet.parameters()).device
        tensor_x = tensor_x.to(device)
        return tensor_x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        return F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

    def postprocess(self, outputs: dict, threshold: float,
                    quality_mask: bool = False, unmodulated: bool = False) -> dict[str, torch.Tensor]:
        return postprocess(outputs, threshold, quality_mask=quality_mask, unmodulated=unmodulated)

    def _apply_finenet(self, original_image, post_outputs, threshold):
        """Apply FineNet refinement to filter false positives."""
        if self.finenet is None:
            logger.warning("FineNet requested but not loaded; skipping refinement")
            return post_outputs

        device = next(self.finenet.parameters()).device
        batch_size = original_image.shape[0]
        patch_radius = 22
        finenet_batch_size = 256

        for i in range(batch_size):
            minutiae = post_outputs['minutiae'][i]
            if minutiae.shape[0] == 0:
                continue

            img = original_image[i, 0]  # (H, W)
            coarse_scores = minutiae[:, 3]
            refined_scores = coarse_scores.clone()
            patch_tensors = []
            patch_indices = []

            for j in range(minutiae.shape[0]):
                x_coord = int(minutiae[j, 0].item())
                y_coord = int(minutiae[j, 1].item())

                # Extract patch
                y_begin = max(0, y_coord - patch_radius)
                x_begin = max(0, x_coord - patch_radius)
                y_end = min(img.shape[0], y_coord + patch_radius)
                x_end = min(img.shape[1], x_coord + patch_radius)

                patch = img[y_begin:y_end, x_begin:x_end]
                if patch.numel() == 0:
                    continue

                # Resize to 224x224 and convert to 3-channel
                patch = patch.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
                patch = F.interpolate(patch, size=(224, 224), mode='nearest')
                patch = patch.expand(-1, 3, -1, -1)  # (1, 3, 224, 224)
                patch = patch * 255.0
                patch_tensors.append(patch)
                patch_indices.append(j)

            if patch_tensors:
                patch_batch = torch.cat(patch_tensors, dim=0).to(device)
                patch_indices_tensor = torch.tensor(patch_indices, device=device, dtype=torch.long)

                finenet_scores = []
                with torch.no_grad():
                    for start in range(0, patch_batch.shape[0], finenet_batch_size):
                        end = start + finenet_batch_size
                        probs = self.finenet(patch_batch[start:end])
                        finenet_scores.append(probs[:, 0])  # class 0 = minutiae

                finenet_scores = torch.cat(finenet_scores, dim=0)
                refined_scores[patch_indices_tensor] = (
                    4.0 * coarse_scores[patch_indices_tensor] + finenet_scores
                ) / 5.0

            minutiae[:, 3] = refined_scores

            # Filter by threshold after FineNet refinement
            keep = minutiae[:, 3] > threshold
            post_outputs['minutiae'][i] = minutiae[keep]

        return post_outputs


def get_minutiaenet(
    coarsenet_weights: str = DEFAULT_COARSENET_WEIGHTS_PATH,
    finenet_weights: str | None = None,
    device: str = DEFAULT_DEVICE,
) -> MinutiaeNetWrapper:
    """Load MinutiaeNet (CoarseNet + optional FineNet) with pre-trained weights."""
    if not os.path.exists(coarsenet_weights):
        raise FileNotFoundError(f"CoarseNet weights not found at: {coarsenet_weights}")

    logger.info(f"Selected device: {device}")
    logger.info("Loading CoarseNet architecture...")
    coarsenet = CoarseNet()
    logger.info(f"Loading CoarseNet weights from: {coarsenet_weights}")
    coarsenet.load_state_dict(torch.load(coarsenet_weights, map_location=device))
    coarsenet.eval()

    finenet = None
    if finenet_weights is not None and os.path.exists(finenet_weights):
        logger.info("Loading FineNet architecture...")
        finenet = FineNet()
        logger.info(f"Loading FineNet weights from: {finenet_weights}")
        finenet.load_state_dict(torch.load(finenet_weights, map_location=device))
        finenet.eval()

    logger.info("Creating wrapper and moving to device...")
    wrapper = MinutiaeNetWrapper(coarsenet=coarsenet, finenet=finenet).to(device)

    if device == "cpu":
        logger.info("Moving wrapper to channels_last memory format...")
        wrapper.to(memory_format=torch.channels_last)

    logger.info("MinutiaeNet ready for inference.")
    return wrapper


# ============================================================
# Post-processing functions
# ============================================================

def gaussian_blur_torch(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    def _get_gaussian_kernel1d(kernel_size, sigma, device, dtype):
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords -= kernel_size // 2
        coords_sq = coords * coords
        sigma_sq = sigma * sigma
        g = torch.exp(-(coords_sq) / (2 * sigma_sq))
        g /= g.sum()
        return g

    kernel_1d = _get_gaussian_kernel1d(kernel_size, sigma, device=image.device, dtype=image.dtype)
    B, C, H, W = image.shape

    kernel_h = kernel_1d.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)
    padding = kernel_size // 2

    blurred_h = F.conv2d(image, kernel_h, padding=(0, padding), groups=C)
    blurred_hv = F.conv2d(blurred_h, kernel_v, padding=(padding, 0), groups=C)
    return blurred_hv


def _post_binarize_mask_cv2(seg_map: torch.Tensor) -> torch.Tensor:
    """Binarize and clean the segmentation mask using original Keras logic."""
    import cv2
    device = seg_map.device
    seg_map_np = seg_map.squeeze(1).cpu().numpy()
    batch_size = seg_map_np.shape[0]
    cleaned_masks = []
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for i in range(batch_size):
        seg_out = seg_map_np[i]
        round_seg = np.round(seg_out)
        seg_out = 1.0 - round_seg
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel_close)
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel_open)
        seg_out = cv2.dilate(seg_out, kernel_dilate)
        cleaned_masks.append(seg_out)

    cleaned_tensor = torch.tensor(np.stack(cleaned_masks), device=device, dtype=torch.float32)
    return cleaned_tensor


def angle_delta(A, B, max_D=np.pi*2):
    delta = np.abs(A - B)
    delta = np.minimum(delta, max_D-delta)
    return delta

def distance_keras(y_true, y_pred, max_D=16, max_O=np.pi/6):
    if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
        return np.zeros((y_true.shape[0], y_pred.shape[0]), dtype=bool)

    xy_true = y_true[:, :2].astype(np.float32, copy=False)
    xy_pred = y_pred[:, :2].astype(np.float32, copy=False)
    diff = xy_true[:, None, :] - xy_pred[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32), dtype=np.float32)

    a_true = y_true[:, 2].astype(np.float32, copy=False)[:, None]
    a_pred = y_pred[:, 2].astype(np.float32, copy=False)[None, :]
    O = np.abs(a_true - a_pred)
    O = np.minimum(O, (2 * np.pi) - O)

    return (D<=max_D)*(O<=max_O)

def nms_keras(mnt):
    if mnt.shape[0]==0:
        return mnt
    mnt_sort = mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    inrange = distance_keras(mnt_sort, mnt_sort, max_D=16, max_O=np.pi/6).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in range(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(bool), :]

def py_cpu_nms_keras(det, thresh):
    if det.shape[0]==0:
        return det
    dets = det.tolist()
    dets.sort(key=lambda x:x[3], reverse=True)
    dets = np.array(dets)

    box_sz = 25
    x1 = np.reshape(dets[:,0],[-1,1]) -box_sz
    y1 = np.reshape(dets[:,1],[-1,1]) -box_sz
    x2 = np.reshape(dets[:,0],[-1,1]) +box_sz
    y2 = np.reshape(dets[:,1],[-1,1]) +box_sz
    scores = dets[:, 2]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep, :]

def fuse_nms_keras(mnt, mnt_set_2):
    if mnt.shape[0]==0:
        return mnt_set_2
    if mnt_set_2.shape[0]==0:
        return mnt
    all_mnt = np.concatenate((mnt, mnt_set_2))
    mnt_sort = all_mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    inrange = distance_keras(mnt_sort, mnt_sort, max_D=16, max_O=2*np.pi).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in range(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(bool), :]

def label2mnt_keras(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5, precomputed_argmax=False):
    mnt_s_out = np.squeeze(mnt_s_out)
    mnt_w_out = np.squeeze(mnt_w_out)
    mnt_h_out = np.squeeze(mnt_h_out)
    mnt_o_out = np.squeeze(mnt_o_out)
    if mnt_s_out.ndim == 0:
        return np.zeros((0, 4))

    rows, cols = np.where(mnt_s_out > thresh)
    mnt_list = np.column_stack((rows, cols)).astype(np.int32, copy=False)
    if mnt_list.shape[0] == 0:
        return np.zeros((0, 4))

    if not precomputed_argmax:
        mnt_w_out = np.argmax(mnt_w_out, axis=-1)
        mnt_h_out = np.argmax(mnt_h_out, axis=-1)
        mnt_o_out = np.argmax(mnt_o_out, axis=-1)

    mnt_final = np.zeros((len(mnt_list), 4))
    mnt_final[:, 0] = mnt_list[:, 1]*8 + mnt_w_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 1] = mnt_list[:, 0]*8 + mnt_h_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 2] = (mnt_o_out[mnt_list[:,0], mnt_list[:,1]]*2-89.)/180*np.pi
    mnt_final[mnt_final[:, 2]<0.0, 2] = mnt_final[mnt_final[:, 2]<0.0, 2]+2*np.pi
    mnt_final[:, 2] = (-mnt_final[:, 2]) % (2*np.pi)
    mnt_final[:, 3] = mnt_s_out[mnt_list[:,0], mnt_list[:, 1]]

    return mnt_final

def _post_detect_minutiae_single_keras(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, seg_out, device):
    mnt_s_out_np = mnt_s_out.cpu().numpy()
    mnt_w_out_np = mnt_w_out.permute(1, 2, 0).cpu().numpy()
    mnt_h_out_np = mnt_h_out.permute(1, 2, 0).cpu().numpy()
    mnt_o_out_np = mnt_o_out.permute(1, 2, 0).cpu().numpy()
    seg_out_np = seg_out.cpu().numpy()

    mnt_w_idx = np.argmax(mnt_w_out_np, axis=-1)
    mnt_h_idx = np.argmax(mnt_h_out_np, axis=-1)
    mnt_o_idx = np.argmax(mnt_o_out_np, axis=-1)
    score_map = mnt_s_out_np * seg_out_np

    final_minutiae_score_threashold = 0.45
    early_minutiae_thres = final_minutiae_score_threashold + 0.05

    while final_minutiae_score_threashold >= 0:
        mnt = label2mnt_keras(
            score_map,
            mnt_w_idx,
            mnt_h_idx,
            mnt_o_idx,
            thresh=early_minutiae_thres,
            precomputed_argmax=True,
        )

        mnt_nms_1 = py_cpu_nms_keras(mnt, 0.5)
        mnt_nms_2 = nms_keras(mnt)
        
        if mnt_nms_1.shape[0] > 4 and mnt_nms_2.shape[0] > 4:
            break
        else:
            final_minutiae_score_threashold = final_minutiae_score_threashold - 0.05
            early_minutiae_thres = early_minutiae_thres - 0.05

    mnt_nms = fuse_nms_keras(mnt_nms_1, mnt_nms_2)
    if mnt_nms.shape[0] > 0:
        mnt_nms = mnt_nms[mnt_nms[:, 3] > early_minutiae_thres, :]

    if mnt_nms.shape[0] == 0:
        return torch.empty((0, 4), device=device)

    return torch.tensor(mnt_nms, device=device, dtype=torch.float32)

def _post_detect_minutiae(outputs: dict, cleaned_mask: torch.Tensor, threshold: float,
                          unmodulated: bool = False):
    """Detect, filter and apply NMS to minutiae for an entire batch using Keras logic."""
    mnt_score_batch = outputs['minutiae_score'].squeeze(1)
    mnt_orient_batch = outputs['minutiae_orientation']
    mnt_x_offset_batch = outputs['minutiae_x_offset']
    mnt_y_offset_batch = outputs['minutiae_y_offset']

    batch_size = mnt_score_batch.shape[0]
    final_minutiae_list = []
    final_minutiae_unmod_list = [] if unmodulated else None

    for i in range(batch_size):
        final_minutiae_list.append(
            _post_detect_minutiae_single_keras(
                mnt_score_batch[i], mnt_x_offset_batch[i],
                mnt_y_offset_batch[i], mnt_orient_batch[i],
                cleaned_mask[i], mnt_score_batch.device))

        if unmodulated:
            final_minutiae_unmod_list.append(
                _post_detect_minutiae_single_keras(
                    mnt_score_batch[i], mnt_x_offset_batch[i],
                    mnt_y_offset_batch[i], mnt_orient_batch[i],
                    torch.ones_like(cleaned_mask[i]), mnt_score_batch.device))

    if unmodulated:
        return final_minutiae_list, final_minutiae_unmod_list
    return final_minutiae_list


def postprocess(outputs: dict, threshold: float,
                quality_mask: bool = False, unmodulated: bool = False,
                return_timings: bool = False):
    timings = {
        'mask_ms': 0.0,
        'minutiae_ms': 0.0,
        'orientation_ms': 0.0,
        'enhanced_ms': 0.0,
        'optional_ms': 0.0,
    }

    # 1. Binarize and clean segmentation mask
    t0 = time.perf_counter()
    cleaned_mask = _post_binarize_mask_cv2(outputs['segmentation'])
    cleaned_mask_up = F.interpolate(
        cleaned_mask.unsqueeze(1).float(),
        scale_factor=8, mode='nearest'
    ).squeeze(1)
    timings['mask_ms'] = (time.perf_counter() - t0) * 1000.0

    # 2. Minutiae detection (including NMS)
    t1 = time.perf_counter()
    if unmodulated:
        final_minutiae_list, final_minutiae_unmod_list = _post_detect_minutiae(
            outputs, cleaned_mask, threshold, unmodulated=True)
    else:
        final_minutiae_list = _post_detect_minutiae(outputs, cleaned_mask, threshold)
    timings['minutiae_ms'] = (time.perf_counter() - t1) * 1000.0

    # 3. Orientation field processing
    t2 = time.perf_counter()
    ori = outputs['orientation']
    ori_idx = torch.argmax(ori, dim=1)
    ori_idx_up = F.interpolate(
        ori_idx.unsqueeze(1).float(),
        scale_factor=8, mode='nearest'
    ).squeeze(1)
    orientation_field_raw = (ori_idx_up * 2.0 - 89.) * torch.pi / 180.0
    orientation_field = orientation_field_raw * cleaned_mask_up
    timings['orientation_ms'] = (time.perf_counter() - t2) * 1000.0

    # 4. Enhanced image processing
    t3 = time.perf_counter()
    enh_real = outputs['enhanced_real'].squeeze(1)
    enh_real_raw = enh_real
    enh_real = enh_real * cleaned_mask_up

    # Min-Max normalization for visualization
    b, h, w = enh_real.shape
    enh_flat = enh_real.view(b, -1)
    enh_min = enh_flat.min(dim=1, keepdim=True)[0]
    enh_max = enh_flat.max(dim=1, keepdim=True)[0]
    enh_norm = (enh_flat - enh_min) / (enh_max - enh_min + 1e-8)
    enh_visual = (enh_norm.view(b, h, w) * 255).byte()
    timings['enhanced_ms'] = (time.perf_counter() - t3) * 1000.0

    result = {
        'minutiae': final_minutiae_list,
        'enhanced_image': enh_visual,
        'segmentation_mask': (cleaned_mask_up * 255).byte(),
        'orientation_field': orientation_field,
    }

    if quality_mask:
        seg_continuous_up = F.interpolate(
            outputs['segmentation'], scale_factor=8, mode='bilinear', align_corners=False
        ).squeeze(1)
        result['quality_mask'] = (seg_continuous_up * 255).byte()

    t4 = time.perf_counter()
    if unmodulated:
        result['orientation_field_unmod'] = orientation_field_raw
        result['minutiae_unmod'] = final_minutiae_unmod_list
        enh_flat_raw = enh_real_raw.view(b, -1)
        enh_min_raw = enh_flat_raw.min(dim=1, keepdim=True)[0]
        enh_max_raw = enh_flat_raw.max(dim=1, keepdim=True)[0]
        enh_norm_raw = (enh_flat_raw - enh_min_raw) / (enh_max_raw - enh_min_raw + 1e-8)
        result['enhanced_image_unmod'] = (enh_norm_raw.view(b, h, w) * 255).byte()
    timings['optional_ms'] = (time.perf_counter() - t4) * 1000.0

    if return_timings:
        return result, timings

    return result
