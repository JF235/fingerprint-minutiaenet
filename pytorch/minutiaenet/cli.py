import argparse
import json
import sys


def parse_gpus(gpus_str: str):
    """Parse GPU specification from string.

    Examples:
        "0" -> 0 (CPU)
        "1" -> 1 (single GPU)
        "2" -> 2 (2 GPUs: 0,1)
        "[0,1,2,3]" -> [0,1,2,3] (specific GPUs)
    """
    if gpus_str.lower() == 'none' or gpus_str == '0':
        return 0

    try:
        return int(gpus_str)
    except ValueError:
        pass

    try:
        parsed = json.loads(gpus_str)
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            return parsed
        raise ValueError("GPU list must contain only integers")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid GPU specification: {gpus_str}")


def infer_command(args):
    """Execute full inference."""
    from .api import run_inference

    gpus = parse_gpus(args.gpus)

    print(f"\n{'=' * 70}")
    print("MinutiaeNet - Full Inference")
    print(f"{'=' * 70}")
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"GPUs:        {gpus}")
    print(f"Batch Size:  {args.batch_size} per GPU")
    print(f"Workers:     {args.cores} per GPU")
    print(f"Recursive:   {args.recursive}")
    print(f"Compile:     {args.compile}")
    print(f"Max Dim:     {args.max_dim}")
    print(f"Strategy:    {args.strategy}")
    print(f"CPU Workers: {args.cpu_workers}")
    print(f"Quality Mask:{args.quality_mask}")
    print(f"Unmodulated: {args.unmodulated}")
    print(f"Full Extr.:  {args.full}")
    print(f"FineNet:     {args.finenet}")
    print(f"{'=' * 70}\n")

    run_inference(
        input_path=args.input,
        output_path=args.output,
        coarsenet_weights=args.weights,
        finenet_weights=args.finenet_weights,
        gpus=gpus,
        batch_size=args.batch_size,
        num_workers=args.cores,
        recursive=args.recursive,
        mnt_degrees=args.degrees,
        compile_model=args.compile,
        max_image_dim=args.max_dim,
        strategy=args.strategy,
        num_cpu_workers=args.cpu_workers,
        quality_mask=args.quality_mask,
        unmodulated=args.unmodulated,
        full=args.full,
        use_finenet=args.finenet,
    )


def forward_command(args):
    """Alias for infer_command."""
    infer_command(args)


def plot_command(args):
    """Generate visualization for processed results."""
    from .plot import plot_from_output_folder

    print(f"\n{'=' * 70}")
    print("MinutiaeNet - Plot Results")
    print(f"{'=' * 70}")
    print(f"Output Path: {args.output}")
    print(f"Image:       {args.image}")

    if args.save:
        print(f"Save To:     {args.save}")
    print(f"{'=' * 70}\n")

    plot_from_output_folder(
        output_path=args.output,
        image_filename=args.image,
        save_path=args.save,
        stride=args.stride,
        degrees=args.degrees,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='minutiaenet',
        description='MinutiaeNet - Robust Minutiae Extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full inference on single GPU
  minutiaenet infer images/ output/ --gpus 1 --batch-size 8

  # Multi-GPU inference (4 GPUs: 0,1,2,3)
  minutiaenet forward images/ output/ --gpus 4 --batch-size 4 --recursive

  # Specific GPUs
  minutiaenet infer images/ output/ --gpus [2,3] --batch-size 8

  # With FineNet verification
  minutiaenet infer images/ output/ --gpus 1 --finenet --finenet-weights Models/FineNet.pth

  # Plot results
  minutiaenet plot output/ image.png --save viz.png
        """
    )

    subcommand_names = ('infer', 'forward', 'plot')

    def add_inference_args(sp):
        sp.add_argument('input', type=str, help='Input: image file, directory, or .txt list')
        sp.add_argument('output', type=str, help='Output directory for results')
        sp.add_argument('--gpus', type=str, default='1',
                        help='GPU config: "0" (CPU), "1" (single GPU), "2" (2 GPUs), "[0,1,2]" (specific GPUs)')
        sp.add_argument('--weights', type=str, default=None, help='Path to CoarseNet weights (.pth)')
        sp.add_argument('--finenet-weights', type=str, default=None, help='Path to FineNet weights (.pth)')
        sp.add_argument('--finenet', action='store_true', help='Use FineNet for minutiae verification')
        sp.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size per GPU (default: 4)')
        sp.add_argument('--cores', type=int, default=4, help='CPU cores for data loading per GPU (default: 4)')
        sp.add_argument('--recursive', '-r', action='store_true', help='Search for images recursively')
        sp.add_argument('--degrees', action='store_true', help='Save minutiae angles in degrees')
        sp.add_argument('--compile', action='store_true', help='Compile model with torch.compile (experimental)')
        sp.add_argument('--max-dim', type=int, default=1024, help='Max image dimension before resizing (default: 1024)')
        sp.add_argument('--strategy', type=str, default='full_gpu', choices=['hybrid', 'full_gpu'],
                        help="Execution strategy (default: full_gpu)")
        sp.add_argument('--cpu-workers', type=int, default=4,
                        help='CPU threads for post-processing (default: 4)')
        sp.add_argument('--quality-mask', action='store_true', help='Export continuous quality mask')
        sp.add_argument('--unmodulated', action='store_true',
                        help='Export unmodulated orientation, enhanced, and minutiae')
        sp.add_argument('--full', action='store_true',
                        help='Full extraction: all outputs plus quality mask and unmodulated')

    if any(h in sys.argv for h in ('-h', '--help')) and not any(cmd in sys.argv for cmd in subcommand_names):
        subparsers_temp = parser.add_subparsers(dest='command', required=False, help='Command to execute')

        for name in ('infer', 'forward'):
            sp = subparsers_temp.add_parser(name)
            add_inference_args(sp)

        sp = subparsers_temp.add_parser('plot')
        sp.add_argument('output', type=str, help='Output directory containing results')
        sp.add_argument('image', type=str, help='Image filename to visualize')
        sp.add_argument('--save', type=str, default=None, help='Path to save visualization')
        sp.add_argument('--stride', type=int, default=16, help='Stride for orientation visualization (default: 16)')

        print(parser.format_help())
        print('\nSUBCOMMANDS:\n')
        for name, sp in subparsers_temp.choices.items():
            print(f"== {name} ==")
            print(sp.format_help())
        return

    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to execute')

    # --- 'infer' command ---
    infer_parser = subparsers.add_parser(
        'infer', help='Run full inference (all outputs)',
        description='Execute complete MinutiaeNet inference pipeline'
    )
    add_inference_args(infer_parser)
    infer_parser.set_defaults(func=infer_command)

    # --- 'forward' command (alias) ---
    forward_parser = subparsers.add_parser(
        'forward', help='Run full inference (alias for infer)',
        description='Execute complete MinutiaeNet inference pipeline (alias for infer)'
    )
    add_inference_args(forward_parser)
    forward_parser.set_defaults(func=forward_command)

    # --- 'plot' command ---
    plot_parser = subparsers.add_parser(
        'plot', help='Visualize inference results',
        description='Generate visualization from saved results'
    )
    plot_parser.add_argument('output', type=str, help='Output directory containing results')
    plot_parser.add_argument('image', type=str, help='Image filename to visualize')
    plot_parser.add_argument('--save', type=str, default=None, help='Path to save visualization')
    plot_parser.add_argument('--stride', type=int, default=16,
                             help='Stride for orientation visualization (default: 16)')
    plot_parser.add_argument('--degrees', action='store_true',
                             help='Interpret stored angles as degrees')
    plot_parser.set_defaults(func=plot_command)

    args = parser.parse_args()

    if hasattr(args, 'weights') and args.weights is None:
        from .mnet_utils import DEFAULT_COARSENET_WEIGHTS_PATH
        args.weights = DEFAULT_COARSENET_WEIGHTS_PATH

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
