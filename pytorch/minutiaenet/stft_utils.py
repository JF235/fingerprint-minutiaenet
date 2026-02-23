"""
Python3-compatible STFT orientation utilities.
Ported verbatim from CoarseNet/MinutiaeNet_utils.py (Python2),
with only integer division fixes applied.
"""
import math
import numpy as np
import cv2
from skimage.filters import gaussian


def nextpow2(x):
    return int(math.ceil(math.log(x, 2)))


def compute_gradient_norm(input_img):
    input_img = input_img.astype(np.float32)
    Gx, Gy = np.gradient(input_img)
    out = np.sqrt(Gx * Gx + Gy * Gy) + 0.000001
    return out


def LowpassFiltering(img, L):
    h, w = img.shape
    h2, w2 = L.shape
    img = cv2.copyMakeBorder(img, 0, h2 - h, 0, w2 - w, cv2.BORDER_CONSTANT, value=0)
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_fft = img_fft * L
    rec_img = np.fft.ifft2(np.fft.fftshift(img_fft))
    rec_img = np.real(rec_img)
    rec_img = rec_img[:h, :w]
    return rec_img


def FastEnhanceTexture(img, sigma=2.5, show=False):
    img = img.astype(np.float32)
    h, w = img.shape
    h2 = 2 ** nextpow2(h)
    w2 = 2 ** nextpow2(w)

    FFTsize = int(np.max([h2, w2]))
    x, y = np.meshgrid(range(-FFTsize // 2, FFTsize // 2), range(-FFTsize // 2, FFTsize // 2))
    r = np.sqrt(x * x + y * y) + 0.0001
    r = r / FFTsize

    L = 1. / (1 + (2 * math.pi * r * sigma) ** 4)
    img_low = LowpassFiltering(img, L)

    gradim1 = compute_gradient_norm(img)
    gradim1 = LowpassFiltering(gradim1, L)

    gradim2 = compute_gradient_norm(img_low)
    gradim2 = LowpassFiltering(gradim2, L)

    diff = gradim1 - gradim2
    ar1 = np.abs(gradim1)
    diff[ar1 > 1] = diff[ar1 > 1] / ar1[ar1 > 1]
    diff[ar1 <= 1] = 0

    cmin = 0.3
    cmax = 0.7

    weight = (diff - cmin) / (cmax - cmin)
    weight[diff < cmin] = 0
    weight[diff > cmax] = 1

    u = weight * img_low + (1 - weight) * img
    temp = img - u
    lim = 20
    temp1 = (temp + lim) * 255 / (2 * lim)
    temp1[temp1 < 0] = 0
    temp1[temp1 > 255] = 255
    return temp1


class local_STFT:
    def __init__(self, patch, weight=None, dBPass=None):
        if weight is not None:
            patch = patch * weight
        patch = patch - np.mean(patch)
        norm = np.linalg.norm(patch)
        patch = patch / (norm + 0.000001)

        f = np.fft.fft2(patch)
        fshift = np.fft.fftshift(f)
        if dBPass is not None:
            fshift = dBPass * fshift

        self.patch_FFT = fshift
        self.patch = patch
        self.ori = None
        self.fre = None
        self.confidence = None
        self.patch_size = patch.shape[0]

    def analysis(self, r, dir_ind_list=None, N=2):
        assert dir_ind_list is not None
        energy = np.abs(self.patch_FFT)
        energy = energy / (np.sum(energy) + 0.00001)
        nrof_dirs = len(dir_ind_list)

        ori_interval = math.pi / nrof_dirs
        ori_interval2 = ori_interval / 2

        pad_size = 1
        dir_norm = np.zeros((nrof_dirs + 2,))
        for i in range(nrof_dirs):
            tmp = energy[dir_ind_list[i][:, 0], dir_ind_list[i][:, 1]]
            dir_norm[i + 1] = np.sum(tmp)

        dir_norm[0] = dir_norm[nrof_dirs]
        dir_norm[nrof_dirs + 1] = dir_norm[1]

        smoothed_dir_norm = dir_norm.copy()
        for i in range(1, nrof_dirs + 1):
            smoothed_dir_norm[i] = (dir_norm[i - 1] + dir_norm[i] * 4 + dir_norm[i + 1]) / 6

        smoothed_dir_norm[0] = smoothed_dir_norm[nrof_dirs]
        smoothed_dir_norm[nrof_dirs + 1] = smoothed_dir_norm[1]

        den = np.sum(smoothed_dir_norm[1:nrof_dirs + 1]) + 0.00001
        smoothed_dir_norm = smoothed_dir_norm / den

        ori = []
        fre = []
        confidence = []

        wenergy = energy * r
        for i in range(1, nrof_dirs + 1):
            if smoothed_dir_norm[i] > smoothed_dir_norm[i - 1] and smoothed_dir_norm[i] > smoothed_dir_norm[i + 1]:
                tmp_ori = (i - pad_size) * ori_interval + ori_interval2 + math.pi / 2
                ori.append(tmp_ori)
                confidence.append(smoothed_dir_norm[i])
                tmp_fre = np.sum(wenergy[dir_ind_list[i - pad_size][:, 0], dir_ind_list[i - pad_size][:, 1]]) / dir_norm[i]
                tmp_fre = 1 / (tmp_fre + 0.00001)
                fre.append(tmp_fre)

        if len(confidence) > 0:
            confidence = np.asarray(confidence)
            fre = np.asarray(fre)
            ori = np.asarray(ori)
            ind = confidence.argsort()[::-1]
            confidence = confidence[ind]
            fre = fre[ind]
            ori = ori[ind]
            if len(confidence) >= 2 and confidence[0] / confidence[1] > 2.0:
                self.ori = [ori[0]]
                self.fre = [fre[0]]
                self.confidence = [confidence[0]]
            elif len(confidence) > N:
                self.ori = ori[:N]
                self.fre = fre[:N]
                self.confidence = confidence[:N]
            else:
                self.ori = ori
                self.fre = fre
                self.confidence = confidence


def get_ridge_flow_top(local_info):
    blkH, blkW = local_info.shape
    dir_map = np.zeros((blkH, blkW)) - 10
    fre_map = np.zeros((blkH, blkW)) - 10
    for i in range(blkH):
        for j in range(blkW):
            if local_info[i, j].ori is None:
                continue
            dir_map[i, j] = local_info[i, j].ori[0]
            fre_map[i, j] = local_info[i, j].fre[0]
    return dir_map, fre_map


def smooth_dir_map(dir_map, sigma=2.0, mask=None):
    cos2Theta = np.cos(dir_map * 2)
    sin2Theta = np.sin(dir_map * 2)
    if mask is not None:
        assert dir_map.shape[0] == mask.shape[0]
        assert dir_map.shape[1] == mask.shape[1]
        cos2Theta[mask == 0] = 0
        sin2Theta[mask == 0] = 0

    cos2Theta = gaussian(cos2Theta, sigma, channel_axis=None, mode='reflect')
    sin2Theta = gaussian(sin2Theta, sigma, channel_axis=None, mode='reflect')
    dir_map = np.arctan2(sin2Theta, cos2Theta) * 0.5
    return dir_map


def get_maps_STFT(img, patch_size=64, block_size=16, preprocess=False):
    assert len(img.shape) == 2

    nrof_dirs = 16
    ovp_size = (patch_size - block_size) // 2
    if preprocess:
        img = FastEnhanceTexture(img, sigma=2.5, show=False)

    img = np.pad(img, (ovp_size, ovp_size), 'symmetric')
    h, w = img.shape
    blkH = (h - patch_size) // block_size + 1
    blkW = (w - patch_size) // block_size + 1
    local_info = np.empty((blkH, blkW), dtype=object)

    x, y = np.meshgrid(range(-patch_size // 2, patch_size // 2), range(-patch_size // 2, patch_size // 2))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    r = np.sqrt(x * x + y * y) + 0.0001

    RMIN = 3
    RMAX = 18
    FLOW = patch_size / RMAX
    FHIGH = patch_size / RMIN
    dRLow = 1. / (1 + (r / FHIGH) ** 4)
    dRHigh = 1. / (1 + (FLOW / r) ** 4)
    dBPass = dRLow * dRHigh

    dir_arr = np.arctan2(y, x)
    dir_arr[dir_arr < 0] = dir_arr[dir_arr < 0] + math.pi
    dir_ind = np.floor(dir_arr / (math.pi / nrof_dirs))
    dir_ind = dir_ind.astype(np.intp)
    dir_ind[dir_ind == nrof_dirs] = 0

    dir_ind_list = []
    for i in range(nrof_dirs):
        tmp = np.argwhere(dir_ind == i)
        dir_ind_list.append(tmp)

    sigma = patch_size / 3
    weight = np.exp(-(x * x + y * y) / (sigma * sigma))

    for i in range(0, blkH):
        for j in range(0, blkW):
            patch = img[i * block_size:i * block_size + patch_size, j * block_size:j * block_size + patch_size].copy()
            local_info[i, j] = local_STFT(patch, weight, dBPass)
            local_info[i, j].analysis(r, dir_ind_list)

    dir_map, fre_map = get_ridge_flow_top(local_info)
    dir_map = smooth_dir_map(dir_map)

    return dir_map, fre_map


def fuse_minu_orientation(dir_map, mnt, mode=1, block_size=16):
    """Fuse minutiae orientation with STFT-based dir_map.
    mode=3: average-based fusion (what the Keras demo uses).
    """
    blkH, blkW = dir_map.shape
    dir_map_mod = dir_map % (2 * np.pi)

    if mode == 2:
        return

    for k in range(mnt.shape[0]):
        row = int(mnt[k, 1] // block_size)
        col = int(mnt[k, 0] // block_size)
        row = min(row, blkH - 1)
        col = min(col, blkW - 1)
        ori_value = dir_map_mod[row, col]
        mnt_ori = mnt[k, 2]

        # Determine closest orientation considering pi ambiguity
        if 0 < mnt_ori <= np.pi / 2:
            if 0 < ori_value <= np.pi / 2:
                fixed_ori = ori_value
            elif np.pi / 2 < ori_value <= np.pi:
                if (ori_value - mnt_ori) < (np.pi - ori_value + mnt_ori):
                    fixed_ori = ori_value
                else:
                    fixed_ori = ori_value + np.pi
            elif np.pi < ori_value <= 3 * np.pi / 2:
                fixed_ori = ori_value - np.pi
            elif 3 * np.pi / 2 < ori_value <= 2 * np.pi:
                if (np.pi * 2 - ori_value + mnt_ori) < (ori_value - np.pi - mnt_ori):
                    fixed_ori = ori_value
                else:
                    fixed_ori = ori_value - np.pi
            else:
                fixed_ori = ori_value
        elif np.pi / 2 < mnt_ori <= np.pi:
            if 0 < ori_value <= np.pi / 2:
                if (mnt_ori - ori_value) < (np.pi - ori_value + mnt_ori):
                    fixed_ori = ori_value
                else:
                    fixed_ori = ori_value + np.pi
            elif np.pi / 2 < ori_value <= np.pi:
                fixed_ori = ori_value
            elif np.pi < ori_value <= 3 * np.pi / 2:
                if (ori_value - mnt_ori) < (mnt_ori - ori_value + np.pi):
                    fixed_ori = ori_value
                else:
                    fixed_ori = ori_value - np.pi
            elif 3 * np.pi / 2 < ori_value <= 2 * np.pi:
                fixed_ori = ori_value - np.pi
            else:
                fixed_ori = ori_value
        elif np.pi < mnt_ori <= 3 * np.pi / 2:
            if 0 < ori_value <= np.pi / 2:
                fixed_ori = ori_value + np.pi
            elif np.pi / 2 < ori_value <= np.pi:
                if (mnt_ori - ori_value) < (ori_value + np.pi - mnt_ori):
                    fixed_ori = ori_value
                else:
                    fixed_ori = ori_value + np.pi
            elif np.pi < ori_value <= 3 * np.pi / 2:
                fixed_ori = ori_value
            elif 3 * np.pi / 2 < ori_value <= 2 * np.pi:
                if (ori_value - mnt_ori) < (mnt_ori - ori_value + np.pi):
                    fixed_ori = ori_value
                else:
                    fixed_ori = ori_value - np.pi
            else:
                fixed_ori = ori_value
        elif 3 * np.pi / 2 < mnt_ori <= 2 * np.pi:
            if 0 < ori_value <= np.pi / 2:
                if (np.pi - mnt_ori + ori_value) < (mnt_ori - np.pi - ori_value):
                    fixed_ori = ori_value
                else:
                    fixed_ori = ori_value + np.pi
            elif np.pi / 2 < ori_value <= np.pi:
                fixed_ori = ori_value + np.pi
            elif np.pi < ori_value <= 3 * np.pi / 2:
                if (mnt_ori - ori_value) < (np.pi * 2 - mnt_ori + ori_value - np.pi):
                    fixed_ori = ori_value
                else:
                    fixed_ori = ori_value - np.pi
            elif 3 * np.pi / 2 < ori_value <= 2 * np.pi:
                fixed_ori = ori_value
            else:
                fixed_ori = ori_value
        else:
            fixed_ori = ori_value

        if mode == 1:
            mnt[k, 2] = fixed_ori
        elif mode == 3:
            mnt[k, 2] = (mnt_ori + fixed_ori) / 2
