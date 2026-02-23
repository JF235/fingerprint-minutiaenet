import sys
import os
sys.path.append(os.path.abspath('..'))  # Ensure minutiaenet module can be imported

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from minutiaenet.api import postprocess
from minutiaenet.wrapper import get_minutiaenet
from minutiaenet.mnet_utils import DEFAULT_COARSENET_WEIGHTS_PATH, DEFAULT_FINENET_WEIGHTS_PATH
from minutiaenet.plot import plot_img, plot_ori_field, plot_mnt

plt.rcParams['figure.figsize'] = (10, 8)

image_path = "/home/joaocontreras/work/fingernet/datasets/NISTSD27/images/B101L9U.bmp"

# Load image using PIL
img_pil = Image.open(image_path).convert("L")
img_np = np.array(img_pil, dtype=np.float32) / 255.0

plt.imshow(img_np, cmap='gray')
plt.title("Original Fingerprint Input")
plt.axis('off')
plt.show()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = get_minutiaenet(
    coarsenet_weights=DEFAULT_COARSENET_WEIGHTS_PATH,
    finenet_weights=DEFAULT_FINENET_WEIGHTS_PATH,
    device=device
)
model.eval()
print("Model loaded successfully.")

# Prepare batch tensor: [Batch, Channels, Height, Width]
img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    raw_outputs = model.coarsenet(img_tensor)

print("Raw CoarseNet keys:", list(raw_outputs.keys()))

with torch.no_grad():
    # Direct full forward pass via the model:
    final_outputs = model(img_tensor)
    
print("Post-processed keys:", list(final_outputs.keys()))

# Extract outputs for the first (and only) image in our batch
minutiae = final_outputs["minutiae"][0].cpu().numpy()
enhanced_img = final_outputs["enhanced_image"][0].cpu().numpy()
seg_mask = final_outputs["segmentation_mask"][0].cpu().numpy()
ori_field = final_outputs["orientation_field"][0].cpu().numpy()

print(f"Detected {len(minutiae)} minutiae.")

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# 1. Orientation Field (Domain Knowledge)
plot_img(axes[0], ori_field)
axes[0].set_title("Orientation Field (Radians)")

# 2. Enhanced Image (Domain Knowledge)
plot_img(axes[1], enhanced_img)
axes[1].set_title("Enhanced Image")

# 3. Orientation Map Plotted over Image
stride = 16
plot_img(axes[2], img_np)
plot_ori_field(axes[2], ori_field, stride=stride)
axes[2].set_title(f"Orientation Flow (Stride: {stride})")

# 4. Final Minutiae Output over Original Image
plot_img(axes[3], img_np)
plot_mnt(axes[3], minutiae)
axes[3].set_title(f"Detected Minutiae ({len(minutiae)})")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

