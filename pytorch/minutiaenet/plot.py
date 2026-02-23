import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from pathlib import Path
import glob


def plot_img(ax: plt.Axes, image: np.ndarray):
    """Plot a grayscale image on an axes."""
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_ori_field(ax: plt.Axes, orientation_field: np.ndarray, stride: int = 16):
    """Overlay orientation field segments on an axes."""
    height, width = orientation_field.shape
    segment_length = stride * 0.45

    for r in range(stride // 2, height, stride):
        for c in range(stride // 2, width, stride):
            angle = orientation_field[r, c]
            if angle != 0:
                dx = segment_length * np.cos(angle)
                dy = segment_length * np.sin(angle)
                ax.plot([c - dx, c + dx], [r - dy, r + dy], 'r-', linewidth=1)


def plot_mnt(ax: plt.Axes, minutiae: np.ndarray, r: int = 10):
    """Overlay minutiae (squares + direction) on an axes."""
    if minutiae.shape[0] == 0:
        return

    ax.plot(
        minutiae[:, 0], minutiae[:, 1],
        'rs', fillstyle='none', markersize=6, markeredgewidth=1
    )

    for x, y, angle, score in minutiae:
        ax.plot([x, x + r * np.cos(angle)], [y, y + r * np.sin(angle)], 'r-', linewidth=1.5)


def plot_raw_output(
    output: dict,
    orig_img: np.ndarray | None = None,
    figsize: tuple = (20, 6),
    stride: int = 16,
):
    """Plot raw model output as a 1x4 grid."""
    orientation_field = output['orientation_field'].squeeze()
    enhanced_image = output['enhanced_image'].squeeze()
    minutiae = output['minutiae']

    input_image = enhanced_image if orig_img is None else orig_img

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    ax1 = axes[0]
    plot_img(ax1, orientation_field)
    ax1.set_title("Orientation Field")

    ax2 = axes[1]
    plot_img(ax2, enhanced_image)
    ax2.set_title("Enhanced Image")

    ax3 = axes[2]
    plot_img(ax3, input_image)
    plot_ori_field(ax3, orientation_field, stride=stride)
    ax3.set_title(f"Orientation Field (Stride: {stride})")

    ax4 = axes[3]
    plot_img(ax4, input_image)
    plot_mnt(ax4, minutiae)
    ax4.set_title(f"Detected Minutiae ({len(minutiae)})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_output(
    result: dict,
    save_path: str | None = None,
    stride: int = 16,
    figsize: tuple = (20, 6),
):
    """Plot inference results as a 1x4 grid."""
    try:
        input_image = np.array(Image.open(result['input_path']).convert('L'))
    except FileNotFoundError:
        print(f"Error: Input image not found at {result['input_path']}")
        return

    orientation_field = result['orientation_field'].squeeze()
    enhanced_image = result['enhanced_image'].squeeze()
    minutiae = result['minutiae'][0]

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    plot_img(axes[0], orientation_field)
    axes[0].set_title("Orientation Field")

    plot_img(axes[1], enhanced_image)
    axes[1].set_title("Enhanced Image")

    plot_img(axes[2], input_image)
    plot_ori_field(axes[2], orientation_field, stride=stride)
    axes[2].set_title(f"Orientation Field (Stride: {stride})")

    plot_img(axes[3], input_image)
    plot_mnt(axes[3], minutiae)
    axes[3].set_title(f"Detected Minutiae ({len(minutiae)})")

    base_name = os.path.basename(result['input_path'])
    fig.suptitle(f"MinutiaeNet Results for: {base_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_from_output_folder(
    output_path: str,
    image_filename: str,
    save_path: str | None = None,
    stride: int = 16,
    degrees: bool = False,
):
    """Plot results from output folder structure."""
    print(f"INFO: Generating visualization for '{image_filename}' from '{output_path}'...")

    image_basename = os.path.basename(image_filename)
    base_name = Path(image_basename).stem

    enhanced_dirs = ['enh*']
    orientation_dirs = ['ori*', 'orientation*']
    minutiae_dirs = ['mnt*', 'minutiae*']

    def find_file_by_dir_patterns(base_dir, dir_patterns, filename):
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            return candidate
        for pat in dir_patterns:
            glob_path = os.path.join(base_dir, pat)
            for match in glob.glob(glob_path):
                if os.path.isdir(match):
                    full = os.path.join(match, filename)
                    if os.path.exists(full):
                        return full
        return None

    enhanced_path = find_file_by_dir_patterns(output_path, enhanced_dirs, image_basename)
    orientation_path = find_file_by_dir_patterns(output_path, orientation_dirs, image_basename)
    minutiae_path = find_file_by_dir_patterns(output_path, minutiae_dirs, f"{base_name}.txt")

    for name, path in (('enhanced', enhanced_path), ('orientation', orientation_path), ('minutiae', minutiae_path)):
        if path is None:
            print(f"ERROR: Could not locate {name} file for '{image_basename}' in '{output_path}'.")
            return
        if not os.path.exists(path):
            print(f"ERROR: Required file not found: {path}")
            return

    enhanced_image = np.array(Image.open(enhanced_path).convert('L'))
    orientation_img = np.array(Image.open(orientation_path))
    orientation_field = np.deg2rad(orientation_img.astype(np.float32) - 90.0)

    minutiae = np.loadtxt(minutiae_path, delimiter=',', skiprows=1)
    if minutiae.ndim == 1 and minutiae.size > 0:
        minutiae = np.expand_dims(minutiae, 0)
    elif minutiae.size == 0:
        minutiae = np.empty((0, 4))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_img(axes[0], enhanced_image)
    axes[0].set_title("Enhanced Image")

    plot_img(axes[1], enhanced_image)
    plot_ori_field(axes[1], orientation_field, stride=stride)
    axes[1].set_title(f"Orientation Field (Stride: {stride})")

    plot_img(axes[2], enhanced_image)
    if degrees:
        minutiae[:, 2] = np.deg2rad(minutiae[:, 2])
    plot_mnt(axes[2], minutiae)
    axes[2].set_title(f"Detected Minutiae ({len(minutiae)})")

    fig.suptitle(f"MinutiaeNet Results for: {image_filename}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    else:
        print("WARNING: save_path not provided. Plot will not be displayed in headless environments.")

    plt.close(fig)
