"""Convert CoarseNet Keras HDF5 weights to PyTorch state_dict."""
import torch
import h5py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from minutiaenet.coarsenet_model import CoarseNet

# =====================
# Layer Mappings
# =====================

# Maps: keras_conv_name -> (pytorch_module_prefix, keras_bn_name, keras_prelu_name)
CONV_BN_PRELU_MAP = {
    # Feature extractor - Initial block
    'conv1_0': ('feature_extractor.conv1_0', 'bn-1_0', 'prelu-1_0'),
    'conv1_1': ('feature_extractor.conv1_1', 'bn-1_1', 'prelu-1_1'),
    'conv1_2': ('feature_extractor.conv1_2', 'bn-1_2', 'prelu-1_2'),
    # Feature extractor - Block 1 (128 filters, 3 residual units)
    'conv2_1': ('feature_extractor.conv2_1', 'bn-2_1', 'prelu-2_1'),
    'conv2_2': ('feature_extractor.conv2_2', 'bn-2_2', 'prelu-2_2'),
    'conv2_3': ('feature_extractor.conv2_3', 'bn-2_3', 'prelu-2_3'),
    'conv2_1b': ('feature_extractor.conv2_1b', 'bn-2_1b', 'prelu-2_1b'),
    'conv2_2b': ('feature_extractor.conv2_2b', 'bn-2_2b', 'prelu-2_2b'),
    'conv2_3b': ('feature_extractor.conv2_3b', 'bn-2_3b', 'prelu-2_3b'),
    'conv2_1c': ('feature_extractor.conv2_1c', 'bn-2_1c', 'prelu-2_1c'),
    'conv2_2c': ('feature_extractor.conv2_2c', 'bn-2_2c', 'prelu-2_2c'),
    'conv2_3c': ('feature_extractor.conv2_3c', 'bn-2_3c', 'prelu-2_3c'),
    # Feature extractor - Block 2 (256 filters, 2 residual units)
    'conv3_1': ('feature_extractor.conv3_1', 'bn-3_1', 'prelu-3_1'),
    'conv3_2': ('feature_extractor.conv3_2', 'bn-3_2', 'prelu-3_2'),
    'conv3_3': ('feature_extractor.conv3_3', 'bn-3_3', 'prelu-3_3'),
    'conv3_1b': ('feature_extractor.conv3_1b', 'bn-3_1b', 'prelu-3_1b'),
    'conv3_2b': ('feature_extractor.conv3_2b', 'bn-3_2b', 'prelu-3_2b'),
    'conv3_3b': ('feature_extractor.conv3_3b', 'bn-3_3b', 'prelu-3_3b'),
    # Feature extractor - Block 3 (512 filters + projection)
    'conv3_1c': ('feature_extractor.conv3_1c', 'bn-3_1c', 'prelu-3_1c'),
    'conv3_2c': ('feature_extractor.conv3_2c', 'bn-3_2c', 'prelu-3_2c'),
    'conv3_3c': ('feature_extractor.conv3_3c', 'bn-3_3c', 'prelu-3_3c'),
    'conv3_4c': ('feature_extractor.conv3_4c', 'bn-3_4c', 'prelu-3_4c'),
    # Orientation/Segmentation Head - ASPP convolutions
    'conv4_1': ('ori_seg_head.atrous_1', 'bn-4_1', 'prelu-4_1'),
    'atrousconv4_2': ('ori_seg_head.atrous_2', 'bn-4_2', 'prelu-4_2'),
    'atrousconv4_3': ('ori_seg_head.atrous_3', 'bn-4_3', 'prelu-4_3'),
    # Ori branch ConvBNPReLU layers
    'convori_1_1': ('ori_seg_head.ori_branch_1.0', 'bn-ori_1_1', 'prelu-ori_1_1'),
    'convori_2_1': ('ori_seg_head.ori_branch_2.0', 'bn-ori_2_1', 'prelu-ori_2_1'),
    'convori_3_1': ('ori_seg_head.ori_branch_3.0', 'bn-ori_3_1', 'prelu-ori_3_1'),
    # Seg branch ConvBNPReLU layers
    'convseg_1_1': ('ori_seg_head.seg_branch_1.0', 'bn-seg_1_1', 'prelu-seg_1_1'),
    'convseg_2_1': ('ori_seg_head.seg_branch_2.0', 'bn-seg_2_1', 'prelu-seg_2_1'),
    'convseg_3_1': ('ori_seg_head.seg_branch_3.0', 'bn-seg_3_1', 'prelu-seg_3_1'),
    # Minutiae head - Block 1
    'convmnt_1_1': ('minutiae_head.mnt_1_1', 'bn-mnt_1_1', 'prelu-mnt_1_1'),
    'convmnt_1_2': ('minutiae_head.mnt_1_2', 'bn-mnt_1_2', 'prelu-mnt_1_2'),
    'convmnt_1_3': ('minutiae_head.mnt_1_3', 'bn-mnt_1_3', 'prelu-mnt_1_3'),
    'convmnt_1_1b': ('minutiae_head.mnt_1_1b', 'bn-mnt_1_1b', 'prelu-mnt_1_1b'),
    'convmnt_1_2b': ('minutiae_head.mnt_1_2b', 'bn-mnt_1_2b', 'prelu-mnt_1_2b'),
    'convmnt_1_3b': ('minutiae_head.mnt_1_3b', 'bn-mnt_1_3b', 'prelu-mnt_1_3b'),
    # Minutiae head - Block 2
    'convmnt_2_1': ('minutiae_head.mnt_2_1', 'bn-mnt_2_1', 'prelu-mnt_2_1'),
    'convmnt_2_2': ('minutiae_head.mnt_2_2', 'bn-mnt_2_2', 'prelu-mnt_2_2'),
    'convmnt_2_3': ('minutiae_head.mnt_2_3', 'bn-mnt_2_3', 'prelu-mnt_2_3'),
    'convmnt_2_1b': ('minutiae_head.mnt_2_1b', 'bn-mnt_2_1b', 'prelu-mnt_2_1b'),
    'convmnt_2_2b': ('minutiae_head.mnt_2_2b', 'bn-mnt_2_2b', 'prelu-mnt_2_2b'),
    'convmnt_2_3b': ('minutiae_head.mnt_2_3b', 'bn-mnt_2_3b', 'prelu-mnt_2_3b'),
    # Minutiae head - Block 3
    'convmnt_3_1': ('minutiae_head.mnt_3_1', 'bn-mnt_3_1', 'prelu-mnt_3_1'),
    'convmnt_3_2': ('minutiae_head.mnt_3_2', 'bn-mnt_3_2', 'prelu-mnt_3_2'),
    'convmnt_3_3': ('minutiae_head.mnt_3_3', 'bn-mnt_3_3', 'prelu-mnt_3_3'),
    'convmnt_3_4': ('minutiae_head.mnt_3_4', 'bn-mnt_3_4', 'prelu-mnt_3_4'),
    # Minutiae output branches
    'convmnt_o_1_1': ('minutiae_head.mnt_o_1_1', 'bn-mnt_o_1_1', 'prelu-mnt_o_1_1'),
    'convmnt_w_1_1': ('minutiae_head.mnt_w_1_1', 'bn-mnt_w_1_1', 'prelu-mnt_w_1_1'),
    'convmnt_h_1_1': ('minutiae_head.mnt_h_1_1', 'bn-mnt_h_1_1', 'prelu-mnt_h_1_1'),
    'convmnt_s_1_1': ('minutiae_head.mnt_s_1_1', 'bn-mnt_s_1_1', 'prelu-mnt_s_1_1'),
}

# Final Conv2D layers (no BN/PReLU, just kernel+bias)
FINAL_CONV_MAP = {
    'ori_1_2': 'ori_seg_head.ori_branch_1.1',
    'ori_2_2': 'ori_seg_head.ori_branch_2.1',
    'ori_3_2': 'ori_seg_head.ori_branch_3.1',
    'seg_1_2': 'ori_seg_head.seg_branch_1.1',
    'seg_2_2': 'ori_seg_head.seg_branch_2.1',
    'seg_3_2': 'ori_seg_head.seg_branch_3.1',
    'mnt_o_1_2': 'minutiae_head.mnt_o_1_2',
    'mnt_w_1_2': 'minutiae_head.mnt_w_1_2',
    'mnt_h_1_2': 'minutiae_head.mnt_h_1_2',
    'mnt_s_1_2': 'minutiae_head.mnt_s_1_2',
    'enh_img_real_1': 'enhancement_module.gabor_real',
    'enh_img_imag_1': 'enhancement_module.gabor_imag',
}


def convert_weights(keras_path: str, output_path: str):
    """Convert Keras HDF5 weights to a PyTorch state_dict for CoarseNet."""
    print(f"Starting conversion from '{keras_path}'...")

    model = CoarseNet()
    state_dict = model.state_dict()

    with h5py.File(keras_path, 'r') as f:
        # ConvBNPReLU blocks
        for keras_conv_name, (pt_prefix, keras_bn_name, keras_prelu_name) in CONV_BN_PRELU_MAP.items():
            # Conv2D: kernel (H,W,Cin,Cout) -> (Cout,Cin,H,W)
            state_dict[f'{pt_prefix}.conv.weight'] = torch.from_numpy(
                f[keras_conv_name][keras_conv_name]['kernel:0'][()]
            ).permute(3, 2, 0, 1)
            state_dict[f'{pt_prefix}.conv.bias'] = torch.from_numpy(
                f[keras_conv_name][keras_conv_name]['bias:0'][()]
            )

            # BatchNorm
            state_dict[f'{pt_prefix}.bn.weight'] = torch.from_numpy(
                f[keras_bn_name][keras_bn_name]['gamma:0'][()]
            )
            state_dict[f'{pt_prefix}.bn.bias'] = torch.from_numpy(
                f[keras_bn_name][keras_bn_name]['beta:0'][()]
            )
            state_dict[f'{pt_prefix}.bn.running_mean'] = torch.from_numpy(
                f[keras_bn_name][keras_bn_name]['moving_mean:0'][()]
            )
            state_dict[f'{pt_prefix}.bn.running_var'] = torch.from_numpy(
                f[keras_bn_name][keras_bn_name]['moving_variance:0'][()]
            )

            # PReLU: Keras shape (1,1,C) -> squeeze to (C,)
            state_dict[f'{pt_prefix}.prelu.weight'] = torch.from_numpy(
                f[keras_prelu_name][keras_prelu_name]['alpha:0'][()]
            ).squeeze()

        # Final Conv2D layers
        for keras_name, pt_name in FINAL_CONV_MAP.items():
            state_dict[f'{pt_name}.weight'] = torch.from_numpy(
                f[keras_name][keras_name]['kernel:0'][()]
            ).permute(3, 2, 0, 1)
            if f'{pt_name}.bias' in state_dict:
                state_dict[f'{pt_name}.bias'] = torch.from_numpy(
                    f[keras_name][keras_name]['bias:0'][()]
                )

    # Load and save
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), output_path)
    print(f"Weights converted and saved to '{output_path}'")


def verify_weights(keras_path: str, pytorch_path: str):
    """Verify PyTorch weights match the original Keras HDF5."""
    print("\n--- Verifying weights ---")

    model = CoarseNet()
    model.load_state_dict(torch.load(pytorch_path, weights_only=True))
    model.eval()
    pt_sd = model.state_dict()

    errors = 0

    with h5py.File(keras_path, 'r') as f:
        for keras_conv_name, (pt_prefix, keras_bn_name, keras_prelu_name) in CONV_BN_PRELU_MAP.items():
            print(f"Checking {keras_conv_name} -> {pt_prefix}...", end=' ')

            keras_kernel = torch.from_numpy(
                f[keras_conv_name][keras_conv_name]['kernel:0'][()]
            ).permute(3, 2, 0, 1)
            if not torch.allclose(pt_sd[f'{pt_prefix}.conv.weight'], keras_kernel):
                errors += 1; print("FAIL (kernel)"); continue

            keras_gamma = torch.from_numpy(f[keras_bn_name][keras_bn_name]['gamma:0'][()])
            if not torch.allclose(pt_sd[f'{pt_prefix}.bn.weight'], keras_gamma):
                errors += 1; print("FAIL (bn.weight)"); continue

            keras_alpha = torch.from_numpy(
                f[keras_prelu_name][keras_prelu_name]['alpha:0'][()]
            ).squeeze()
            if not torch.allclose(pt_sd[f'{pt_prefix}.prelu.weight'], keras_alpha):
                errors += 1; print("FAIL (prelu.weight)"); continue

            print("OK")

        for keras_name, pt_name in FINAL_CONV_MAP.items():
            print(f"Checking {keras_name} -> {pt_name}...", end=' ')

            keras_kernel = torch.from_numpy(
                f[keras_name][keras_name]['kernel:0'][()]
            ).permute(3, 2, 0, 1)
            if not torch.allclose(pt_sd[f'{pt_name}.weight'], keras_kernel):
                errors += 1; print("FAIL (kernel)"); continue

            print("OK")

    print("--- Verification complete ---")
    if errors == 0:
        print("All weights verified successfully.")
    else:
        print(f"{errors} tensor(s) did not match.")
    return errors == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert CoarseNet Keras weights to PyTorch")
    parser.add_argument("--keras", type=str, required=True, help="Path to Keras .h5 file")
    parser.add_argument("--pytorch", type=str, required=True, help="Output path for .pth file")
    parser.add_argument("--verify", action='store_true', help="Verify converted weights")
    args = parser.parse_args()

    convert_weights(args.keras, args.pytorch)
    if args.verify:
        verify_weights(args.keras, args.pytorch)


if __name__ == '__main__':
    main()
