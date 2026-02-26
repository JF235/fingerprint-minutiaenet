"""
PyTorch MinutiaeNet demo — faithful reproduction of the Keras pipeline.

Changes vs original PyTorch version:
  1. Uses CPU to avoid GPU kernel differences.
  2. Computes STFT orientation (FastEnhanceTexture + get_maps_STFT) exactly like Keras.
  3. Applies fuse_minu_orientation(dir_map, mnt, mode=3) after NMS to override
     network orientation with STFT orientation — this was the main missing step.
  4. OF visualization uses the STFT dir_map (not network orientation).
"""
import sys, os
import math
import numpy as np
from scipy import ndimage
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# PyTorch model + STFT utilities (Python3-compatible)
sys.path.append(os.path.realpath('pytorch'))
from minutiaenet.wrapper import get_minutiaenet
from minutiaenet.stft_utils import FastEnhanceTexture, get_maps_STFT, fuse_minu_orientation

# Keras utilities for IO only
sys.path.append(os.path.realpath('CoarseNet'))
from MinutiaeNet_utils import get_files_in_folder, init_log


def mnt_writer(mnt, image_name, image_size, file_name):
    """Python3-compatible mnt_writer."""
    with open(file_name, 'w') as f:
        f.write('%s\n' % image_name)
        f.write('%d %d %d\n' % (mnt.shape[0], image_size[0], image_size[1]))
        for i in range(mnt.shape[0]):
            f.write('%d %d %.6f %.4f\n' % (mnt[i, 0], mnt[i, 1], mnt[i, 2], mnt[i, 3]))


def draw_minutiae(image, minutiae, fname, saveimage=False, r=15):
    """Python3-compatible draw_minutiae."""
    image = np.squeeze(image)
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    if minutiae.shape[0] > 0:
        plt.plot(minutiae[:, 0], minutiae[:, 1], 'rs', fillstyle='none', linewidth=1)
        for x, y, o, s in minutiae:
            plt.plot([x, x + r * np.cos(o)], [y, y + r * np.sin(o)], 'r-')
    plt.axis([0, image.shape[1], image.shape[0], 0])
    plt.axis('off')
    if saveimage:
        plt.savefig(fname, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


def show_orientation_field(img, dir_map, mask=None, fname=None):
    h, w = img.shape[:2]
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)
    blkH, blkW = dir_map.shape
    blk_size = h / blkH
    R = blk_size / 2 * 0.8
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    for i in range(blkH):
        y0 = int(i * blk_size + blk_size / 2)
        for j in range(blkW):
            x0 = int(j * blk_size + blk_size / 2)
            ori = dir_map[i, j]
            if mask[y0, x0] == 0 or ori < -9:
                continue
            x1 = x0 - R * math.cos(ori)
            x2 = x0 + R * math.cos(ori)
            y1 = y0 - R * math.sin(ori)
            y2 = y0 + R * math.sin(ori)
            ax.plot([x1, x2], [y1, y2], 'r-', lw=2)
    ax.axis('off')
    if fname is not None:
        fig.savefig(fname, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show(block=True)


def main():
    output_dir = 'output_fine_new'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging = init_log(output_dir)
    inference_set = ['Dataset/CoarseNet_test/']

    # Force CPU for reproducibility
    device = "cpu"
    logging.info(f"Using device: {device}")

    model = get_minutiaenet(
        coarsenet_weights='Models/CoarseNet.pth',
        finenet_weights='Models/FineNet.pth',
        device=device
    )

    for deploy_set in inference_set:
        set_name = deploy_set.split('/')[-2]
        img_paths, img_names = get_files_in_folder(deploy_set + 'img_files/', '.bmp')

        logging.info("Predicting \"%s\":" % (set_name))

        for i, img_name in enumerate(img_names):
            logging.info("\"%s\" %d / %d: %s" % (set_name, i + 1, len(img_names), img_name))

            image = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
            img_size = image.shape
            img_size = np.array(img_size, dtype=np.int32) // 8 * 8
            image = image[:img_size[0], :img_size[1]]
            original_image = image.copy()

            # ---- STFT orientation (exactly like Keras) ----
            texture_img = FastEnhanceTexture(image, sigma=2.5, show=False)
            dir_map, fre_map = get_maps_STFT(texture_img, patch_size=64, block_size=16, preprocess=True)

            # ---- PyTorch inference ----
            img_norm = image.astype(np.float32) / 255.0
            img_tensor = model.prepare_input(img_norm)
            padded_x = model.preprocess(img_tensor)

            with torch.no_grad():
                raw_outputs = model.coarsenet(padded_x)

            # ---- Post-processing (replicates Keras exactly) ----
            outputs = model.postprocess(raw_outputs, threshold=0.45)
            # ---- FineNet refinement ----
            outputs = model._apply_finenet(padded_x, outputs, threshold=0.45)

            minutiae_list = outputs['minutiae'][0].cpu().numpy()
            seg_out = outputs['segmentation_mask'][0].cpu().numpy()

            # ---- Fuse minutiae orientation with STFT dir_map ----
            fuse_minu_orientation(dir_map, minutiae_list, mode=3)

            # ---- Segmentation mask for OF visualization ----
            final_mask = (seg_out > 127).astype(np.uint8)

            # ---- OF visualization using STFT dir_map (like Keras) ----
            show_orientation_field(original_image, dir_map + np.pi, mask=final_mask,
                                   fname="%s/%s_OF.jpg" % (output_dir, img_name))

            # ---- Write minutiae and images ----
            mnt_writer(minutiae_list, img_name, img_size, "%s/%s.mnt" % (output_dir, img_name))
            draw_minutiae(original_image, minutiae_list, "%s/%s_minu.jpg" % (output_dir, img_name), saveimage=True)

            cv2.imwrite("%s/%s_seg.jpg" % (output_dir, img_name), final_mask * 255)


if __name__ == '__main__':
    main()
