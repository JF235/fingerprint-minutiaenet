"""Convert FineNet Keras HDF5 weights to PyTorch state_dict."""
import torch
import h5py
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from minutiaenet.finenet_model import FineNet

# =====================
# Build layer ordering
# =====================
# Keras auto-numbers Conv2D as conv2d_1..conv2d_203 and
# BatchNormalization as batch_normalization_1..batch_normalization_203
# in the order they are created during model construction.

# Each entry: PyTorch module path (relative to FineNet)
# conv2d_N -> CONV_PATHS[N-1].conv
# batch_normalization_N -> CONV_PATHS[N-1].bn
CONV_PATHS = []

# Stem (5 ConvBNReLU): stem is Sequential with MaxPool2d at indices 3 and 6
CONV_PATHS.extend([
    'stem.0',    # conv2d_1: Conv(3, 32, 3, stride=2, valid)
    'stem.1',    # conv2d_2: Conv(32, 32, 3, valid)
    'stem.2',    # conv2d_3: Conv(32, 64, 3)
    # stem.3 = MaxPool2d
    'stem.4',    # conv2d_4: Conv(64, 80, 1, valid)
    'stem.5',    # conv2d_5: Conv(80, 192, 3, valid)
    # stem.6 = MaxPool2d
])

# mixed_5b (7 convs)
CONV_PATHS.extend([
    'mixed_5b_branch0',      # conv2d_6:  Conv(192, 96, 1)
    'mixed_5b_branch1.0',    # conv2d_7:  Conv(192, 48, 1)
    'mixed_5b_branch1.1',    # conv2d_8:  Conv(48, 64, 5)
    'mixed_5b_branch2.0',    # conv2d_9:  Conv(192, 64, 1)
    'mixed_5b_branch2.1',    # conv2d_10: Conv(64, 96, 3)
    'mixed_5b_branch2.2',    # conv2d_11: Conv(96, 96, 3)
    'mixed_5b_pool.1',       # conv2d_12: Conv(192, 64, 1) [pool.0=AvgPool]
])

# 10x block35 (6 convs per block: branch0=1, branch1=2, branch2=3)
for i in range(10):
    CONV_PATHS.extend([
        f'block35.{i}.branch0',       # 1x1
        f'block35.{i}.branch1.0',     # 1x1
        f'block35.{i}.branch1.1',     # 3x3
        f'block35.{i}.branch2.0',     # 1x1
        f'block35.{i}.branch2.1',     # 3x3
        f'block35.{i}.branch2.2',     # 3x3
    ])

# mixed_6a (4 convs)
CONV_PATHS.extend([
    'mixed_6a_branch0',      # conv: 3x3 stride=2 valid
    'mixed_6a_branch1.0',    # conv: 1x1
    'mixed_6a_branch1.1',    # conv: 3x3
    'mixed_6a_branch1.2',    # conv: 3x3 stride=2 valid
])

# 20x block17 (4 convs per block: branch0=1, branch1=3)
for i in range(20):
    CONV_PATHS.extend([
        f'block17.{i}.branch0',       # 1x1
        f'block17.{i}.branch1.0',     # 1x1
        f'block17.{i}.branch1.1',     # 1x7
        f'block17.{i}.branch1.2',     # 7x1
    ])

# mixed_7a (7 convs)
CONV_PATHS.extend([
    'mixed_7a_branch0.0',    # 1x1
    'mixed_7a_branch0.1',    # 3x3 stride=2 valid
    'mixed_7a_branch1.0',    # 1x1
    'mixed_7a_branch1.1',    # 3x3 stride=2 valid
    'mixed_7a_branch2.0',    # 1x1
    'mixed_7a_branch2.1',    # 3x3
    'mixed_7a_branch2.2',    # 3x3 stride=2 valid
])

# 10x block8 (4 convs per block: branch0=1, branch1=3)
for i in range(10):
    CONV_PATHS.extend([
        f'block8.{i}.branch0',       # 1x1
        f'block8.{i}.branch1.0',     # 1x1
        f'block8.{i}.branch1.1',     # 1x3
        f'block8.{i}.branch1.2',     # 3x1
    ])

assert len(CONV_PATHS) == 203, f"Expected 203, got {len(CONV_PATHS)}"

# Block projection convs (named, with bias, no BN)
BLOCK_PROJ = {}
for i in range(10):
    BLOCK_PROJ[f'block35_{i+1}_conv'] = f'block35.{i}.up'
for i in range(20):
    BLOCK_PROJ[f'block17_{i+1}_conv'] = f'block17.{i}.up'
for i in range(10):
    BLOCK_PROJ[f'block8_{i+1}_conv'] = f'block8.{i}.up'


def convert_weights(keras_path: str, output_path: str):
    """Convert Keras HDF5 weights to a PyTorch state_dict for FineNet."""
    print(f"Starting conversion from '{keras_path}'...")

    model = FineNet()
    state_dict = model.state_dict()

    with h5py.File(keras_path, 'r') as f:
        # Navigate to model_weights group if present
        root = f['model_weights'] if 'model_weights' in f else f

        # --- Conv2D + BatchNorm (conv2d_1..203 + batch_normalization_1..203) ---
        for idx, pt_path in enumerate(CONV_PATHS, start=1):
            keras_conv = f'conv2d_{idx}'
            keras_bn = f'batch_normalization_{idx}'

            # Conv2D kernel: (H,W,Cin,Cout) -> (Cout,Cin,H,W)
            kernel = torch.from_numpy(
                root[keras_conv][keras_conv]['kernel:0'][()]
            ).permute(3, 2, 0, 1)
            state_dict[f'{pt_path}.conv.weight'] = kernel

            # BatchNorm (scale=False: no gamma, only beta + running stats)
            state_dict[f'{pt_path}.bn.weight'] = torch.ones(kernel.shape[0])
            state_dict[f'{pt_path}.bn.bias'] = torch.from_numpy(
                root[keras_bn][keras_bn]['beta:0'][()]
            )
            state_dict[f'{pt_path}.bn.running_mean'] = torch.from_numpy(
                root[keras_bn][keras_bn]['moving_mean:0'][()]
            )
            state_dict[f'{pt_path}.bn.running_var'] = torch.from_numpy(
                root[keras_bn][keras_bn]['moving_variance:0'][()]
            )

        # --- Block projection convs (with bias, no BN) ---
        for keras_name, pt_path in BLOCK_PROJ.items():
            state_dict[f'{pt_path}.weight'] = torch.from_numpy(
                root[keras_name][keras_name]['kernel:0'][()]
            ).permute(3, 2, 0, 1)
            state_dict[f'{pt_path}.bias'] = torch.from_numpy(
                root[keras_name][keras_name]['bias:0'][()]
            )

        # --- conv_7b (named conv + named BN) ---
        state_dict['conv_7b.conv.weight'] = torch.from_numpy(
            root['conv_7b']['conv_7b']['kernel:0'][()]
        ).permute(3, 2, 0, 1)

        conv_7b_bn_name = 'conv_7b_bn'
        state_dict['conv_7b.bn.weight'] = torch.ones(1536)
        state_dict['conv_7b.bn.bias'] = torch.from_numpy(
            root[conv_7b_bn_name][conv_7b_bn_name]['beta:0'][()]
        )
        state_dict['conv_7b.bn.running_mean'] = torch.from_numpy(
            root[conv_7b_bn_name][conv_7b_bn_name]['moving_mean:0'][()]
        )
        state_dict['conv_7b.bn.running_var'] = torch.from_numpy(
            root[conv_7b_bn_name][conv_7b_bn_name]['moving_variance:0'][()]
        )

        # --- Prediction / classifier (Dense layer) ---
        # Try both 'predictions' and 'prediction' as Keras naming can vary
        pred_name = 'predictions' if 'predictions' in root else 'prediction'
        # Keras Dense kernel: (in, out) -> PyTorch Linear weight: (out, in)
        state_dict['classifier.weight'] = torch.from_numpy(
            root[pred_name][pred_name]['kernel:0'][()]
        ).T
        state_dict['classifier.bias'] = torch.from_numpy(
            root[pred_name][pred_name]['bias:0'][()]
        )

    # Load and save
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), output_path)
    print(f"Weights converted and saved to '{output_path}'")


def verify_weights(keras_path: str, pytorch_path: str):
    """Verify PyTorch weights match the original Keras HDF5."""
    print("\n--- Verifying weights ---")

    model = FineNet()
    model.load_state_dict(torch.load(pytorch_path, weights_only=True))
    model.eval()
    pt_sd = model.state_dict()

    errors = 0

    with h5py.File(keras_path, 'r') as f:
        root = f['model_weights'] if 'model_weights' in f else f

        # Check numbered conv+bn layers
        for idx, pt_path in enumerate(CONV_PATHS, start=1):
            keras_conv = f'conv2d_{idx}'
            keras_bn = f'batch_normalization_{idx}'
            print(f"Checking conv2d_{idx} -> {pt_path}...", end=' ')

            keras_kernel = torch.from_numpy(
                root[keras_conv][keras_conv]['kernel:0'][()]
            ).permute(3, 2, 0, 1)
            if not torch.allclose(pt_sd[f'{pt_path}.conv.weight'], keras_kernel):
                errors += 1; print("FAIL (kernel)"); continue

            keras_beta = torch.from_numpy(root[keras_bn][keras_bn]['beta:0'][()])
            if not torch.allclose(pt_sd[f'{pt_path}.bn.bias'], keras_beta):
                errors += 1; print("FAIL (bn.bias)"); continue

            if not torch.allclose(pt_sd[f'{pt_path}.bn.weight'], torch.ones_like(pt_sd[f'{pt_path}.bn.weight'])):
                errors += 1; print("FAIL (bn.weight should be ones)"); continue

            print("OK")

        # Check block projection convs
        for keras_name, pt_path in BLOCK_PROJ.items():
            print(f"Checking {keras_name} -> {pt_path}...", end=' ')
            keras_kernel = torch.from_numpy(
                root[keras_name][keras_name]['kernel:0'][()]
            ).permute(3, 2, 0, 1)
            if not torch.allclose(pt_sd[f'{pt_path}.weight'], keras_kernel):
                errors += 1; print("FAIL (kernel)"); continue
            keras_bias = torch.from_numpy(root[keras_name][keras_name]['bias:0'][()])
            if not torch.allclose(pt_sd[f'{pt_path}.bias'], keras_bias):
                errors += 1; print("FAIL (bias)"); continue
            print("OK")

        # Check conv_7b
        print("Checking conv_7b...", end=' ')
        keras_kernel = torch.from_numpy(
            root['conv_7b']['conv_7b']['kernel:0'][()]
        ).permute(3, 2, 0, 1)
        if not torch.allclose(pt_sd['conv_7b.conv.weight'], keras_kernel):
            errors += 1; print("FAIL (kernel)")
        else:
            print("OK")

        # Check classifier
        pred_name = 'predictions' if 'predictions' in root else 'prediction'
        print(f"Checking {pred_name} -> classifier...", end=' ')
        keras_kernel = torch.from_numpy(root[pred_name][pred_name]['kernel:0'][()]).T
        if not torch.allclose(pt_sd['classifier.weight'], keras_kernel):
            errors += 1; print("FAIL (kernel)")
        else:
            print("OK")

    print("--- Verification complete ---")
    if errors == 0:
        print("All weights verified successfully.")
    else:
        print(f"{errors} tensor(s) did not match.")
    return errors == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert FineNet Keras weights to PyTorch")
    parser.add_argument("--keras", type=str, required=True, help="Path to Keras .h5 file")
    parser.add_argument("--pytorch", type=str, required=True, help="Output path for .pth file")
    parser.add_argument("--verify", action='store_true', help="Verify converted weights")
    args = parser.parse_args()

    convert_weights(args.keras, args.pytorch)
    if args.verify:
        verify_weights(args.keras, args.pytorch)


if __name__ == '__main__':
    main()
