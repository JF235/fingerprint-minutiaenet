"""Utility to inspect HDF5 weight files and print layer structure."""
import h5py
import sys
import numpy as np


def inspect_h5(path):
    """Print all layers and weight shapes in an HDF5 file."""
    with h5py.File(path, 'r') as f:
        print(f"=== Inspecting: {path} ===")
        print(f"Top-level keys: {list(f.keys())}\n")

        for layer_name in sorted(f.keys()):
            layer = f[layer_name]
            if isinstance(layer, h5py.Group):
                print(f"Layer: {layer_name}")
                for sub_name in layer.keys():
                    sub = layer[sub_name]
                    if isinstance(sub, h5py.Group):
                        for w_name in sub.keys():
                            w = sub[w_name]
                            print(f"  {layer_name}/{sub_name}/{w_name}: shape={w.shape}, dtype={w.dtype}")
                    elif isinstance(sub, h5py.Dataset):
                        print(f"  {layer_name}/{sub_name}: shape={sub.shape}, dtype={sub.dtype}")
            elif isinstance(layer, h5py.Dataset):
                print(f"Dataset: {layer_name}: shape={layer.shape}")
            print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <path_to_h5_file>")
        sys.exit(1)
    inspect_h5(sys.argv[1])
