import logging
import os
import glob
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import timedelta
import warnings
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .wrapper import MinutiaeNetWrapper, get_minutiaenet, postprocess
from .mnet_utils import get_minutiaenet_logger, MnetTimer, DEFAULT_COARSENET_WEIGHTS_PATH, DEFAULT_FINENET_WEIGHTS_PATH

logger = get_minutiaenet_logger('minutiaenet.api', level=logging.DEBUG)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _init_profile_metrics() -> dict[str, deque]:
    window = 30
    return {
        'load_ms': deque(maxlen=window),
        'h2d_ms': deque(maxlen=window),
        'coarse_gpu_ms': deque(maxlen=window),
        'post_cpu_ms': deque(maxlen=window),
        'post_mask_ms': deque(maxlen=window),
        'post_minutiae_ms': deque(maxlen=window),
        'post_orientation_ms': deque(maxlen=window),
        'post_enhanced_ms': deque(maxlen=window),
        'post_optional_ms': deque(maxlen=window),
        'finenet_ms': deque(maxlen=window),
        'd2h_ms': deque(maxlen=window),
        'pack_ms': deque(maxlen=window),
        'submit_ms': deque(maxlen=window),
        'queue_wait_ms': deque(maxlen=window),
    }


def _profile_avg(metrics: dict[str, deque], key: str) -> float:
    values = metrics[key]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _update_profile_tqdm(iterator, metrics: dict[str, deque], queue_depth: int):
    iterator.set_postfix({
        'load': f"{_profile_avg(metrics, 'load_ms'):.1f}ms",
        'h2d': f"{_profile_avg(metrics, 'h2d_ms'):.1f}ms",
        'coarse': f"{_profile_avg(metrics, 'coarse_gpu_ms'):.1f}ms",
        'post': f"{_profile_avg(metrics, 'post_cpu_ms'):.1f}ms",
        'p_msk': f"{_profile_avg(metrics, 'post_mask_ms'):.1f}ms",
        'p_mnt': f"{_profile_avg(metrics, 'post_minutiae_ms'):.1f}ms",
        'p_ori': f"{_profile_avg(metrics, 'post_orientation_ms'):.1f}ms",
        'p_enh': f"{_profile_avg(metrics, 'post_enhanced_ms'):.1f}ms",
        'p_opt': f"{_profile_avg(metrics, 'post_optional_ms'):.1f}ms",
        'fine': f"{_profile_avg(metrics, 'finenet_ms'):.1f}ms",
        'd2h': f"{_profile_avg(metrics, 'd2h_ms'):.1f}ms",
        'pack': f"{_profile_avg(metrics, 'pack_ms'):.1f}ms",
        'submit': f"{_profile_avg(metrics, 'submit_ms'):.1f}ms",
        'wait': f"{_profile_avg(metrics, 'queue_wait_ms'):.1f}ms",
        'q': queue_depth,
    }, refresh=False)


def _log_profile_summary(profile_name: str, metrics: dict[str, deque]):
    parts = [
        f"load={_profile_avg(metrics, 'load_ms'):.2f}ms",
        f"h2d={_profile_avg(metrics, 'h2d_ms'):.2f}ms",
        f"coarse_gpu={_profile_avg(metrics, 'coarse_gpu_ms'):.2f}ms",
        f"post_cpu={_profile_avg(metrics, 'post_cpu_ms'):.2f}ms",
        f"post_mask={_profile_avg(metrics, 'post_mask_ms'):.2f}ms",
        f"post_minutiae={_profile_avg(metrics, 'post_minutiae_ms'):.2f}ms",
        f"post_orientation={_profile_avg(metrics, 'post_orientation_ms'):.2f}ms",
        f"post_enhanced={_profile_avg(metrics, 'post_enhanced_ms'):.2f}ms",
        f"post_optional={_profile_avg(metrics, 'post_optional_ms'):.2f}ms",
        f"finenet={_profile_avg(metrics, 'finenet_ms'):.2f}ms",
        f"d2h={_profile_avg(metrics, 'd2h_ms'):.2f}ms",
        f"pack={_profile_avg(metrics, 'pack_ms'):.2f}ms",
        f"submit={_profile_avg(metrics, 'submit_ms'):.2f}ms",
        f"wait={_profile_avg(metrics, 'queue_wait_ms'):.2f}ms",
    ]
    logger.info(f"[{profile_name}] Profiling summary (moving average window=30): " + " | ".join(parts))


class FingerprintDataset(Dataset):
    """Dataset for loading fingerprint images."""

    def __init__(self, image_paths: list[str], max_dim: int):
        self.image_paths = image_paths
        self.max_dim = max_dim

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img_pil = Image.open(img_path).convert("L")

            if img_pil.height > self.max_dim or img_pil.width > self.max_dim:
                logger.warning(
                    f"Image {os.path.basename(img_path)} with size {img_pil.size} "
                    f"exceeds max_dim of {self.max_dim}. Resizing."
                )
                img_pil.thumbnail(
                    (self.max_dim, self.max_dim), Image.Resampling.LANCZOS
                )

            img_np = np.array(img_pil, dtype=np.float32) / 255.0

            return {
                "image": torch.from_numpy(img_np).unsqueeze(0),
                "path": img_path,
                "original_shape": img_np.shape,
            }
        except Exception as e:
            logger.warning(f"Could not load image {img_path}. Skipping. Error: {e}")
            return None


def find_image_paths(input_path: str, recursive: bool = True) -> list[str]:
    """Find all image paths from input."""
    image_paths = []

    if os.path.isfile(input_path):
        _, ext = os.path.splitext(input_path)
        if ext.lower() in [".txt", ".list"]:
            with open(input_path, "r") as f:
                for line in f:
                    path = line.strip()
                    if path:
                        image_paths.append(path)
        else:
            single_supported = [".png", ".wsq", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]
            if ext.lower() in single_supported:
                image_paths.append(input_path)
            else:
                raise ValueError(
                    f"Only {single_supported} files are supported (received: {input_path})"
                )
    elif os.path.isdir(input_path):
        extensions = ["png", "bmp", "jpg", "jpeg", "tif", "tiff"]
        for ext in extensions:
            pattern = (
                f"{input_path}/**/*.{ext}" if recursive else f"{input_path}/*.{ext}"
            )
            image_paths.extend(glob.glob(pattern, recursive=recursive))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    if not image_paths:
        raise ValueError(f"No images found in: {input_path}")

    return sorted(image_paths)


def dynamic_padding_collate(batch):
    """Custom collate that pads images to the max size within a batch."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None

    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    images, paths, orig_shapes = [], [], []
    for item in batch:
        img = item["image"]
        _, h, w = img.shape
        padding = (0, max_w - w, 0, max_h - h)
        padded_img = torch.nn.functional.pad(img, padding, mode="constant", value=1.0)

        images.append(padded_img)
        paths.append(item["path"])
        orig_shapes.append(item["original_shape"])

    batch_tensors = torch.stack(images)
    batch_paths = paths
    batch_orig_shapes = (
        torch.tensor([s[0] for s in orig_shapes]),
        torch.tensor([s[1] for s in orig_shapes]),
    )

    return batch_tensors, batch_paths, batch_orig_shapes


def save_results(result_item: dict, output_path: str, mnt_degrees: bool = True,
                 input_base_path: str = None):
    """Save inference results to disk."""
    input_path = result_item["input_path"]

    if input_base_path and os.path.isdir(input_base_path):
        rel_path = os.path.relpath(input_path, input_base_path)
        rel_dir = os.path.dirname(rel_path)
        original_filename = os.path.basename(rel_path)
    else:
        rel_dir = ""
        original_filename = os.path.basename(input_path)

    base_name = os.path.splitext(original_filename)[0]

    # Save minutiae (.min) - padrão: ângulo CCW em graus (int), qualidade 0-100 (int)
    minutiae = result_item["minutiae"].copy()
    angle_ccw_deg = np.round((-np.rad2deg(minutiae[:, 2])) % 360).astype(int)
    quality_int = np.round(minutiae[:, 3] * 100).astype(int)
    minutiae_out = np.column_stack([
        minutiae[:, 0].astype(int),
        minutiae[:, 1].astype(int),
        angle_ccw_deg,
        quality_int,
    ])

    minutiae_path = os.path.join(output_path, "minutiae", rel_dir, f"{base_name}.min")
    os.makedirs(os.path.dirname(minutiae_path), exist_ok=True)
    np.savetxt(
        minutiae_path, minutiae_out,
        fmt="%d",
        header="X Y ANGLE QUALITY", comments="#MIN ", delimiter=" "
    )

    # Save enhanced image
    enhanced_path = os.path.join(output_path, "enhanced", rel_dir, original_filename)
    os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
    Image.fromarray(result_item["enhanced_image"]).save(enhanced_path)

    # Save mask
    mask_path = os.path.join(output_path, "mask", rel_dir, original_filename)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    Image.fromarray(result_item["segmentation_mask"]).save(mask_path)

    # Save orientation field (encoded as PNG)
    ori_cpu = result_item["orientation_field"]
    orientation_path = os.path.join(output_path, "ori", rel_dir, original_filename)
    os.makedirs(os.path.dirname(orientation_path), exist_ok=True)
    angles_deg_shifted = np.round(np.rad2deg(ori_cpu) + 90).astype(np.uint8)
    Image.fromarray(angles_deg_shifted).save(orientation_path)

    if 'quality_mask' in result_item:
        qmask_path = os.path.join(output_path, "quality_mask", rel_dir, original_filename)
        os.makedirs(os.path.dirname(qmask_path), exist_ok=True)
        Image.fromarray(result_item["quality_mask"]).save(qmask_path)

    if 'enhanced_image_unmod' in result_item:
        path = os.path.join(output_path, "enhanced_unmod", rel_dir, original_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(result_item["enhanced_image_unmod"]).save(path)

    if 'orientation_field_unmod' in result_item:
        path = os.path.join(output_path, "ori_unmod", rel_dir, original_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        angles = np.round(np.rad2deg(result_item["orientation_field_unmod"]) + 90).astype(np.uint8)
        Image.fromarray(angles).save(path)

    if 'minutiae_unmod' in result_item:
        mnt_unmod = result_item["minutiae_unmod"].copy()
        angle_ccw_deg_unmod = np.round((-np.rad2deg(mnt_unmod[:, 2])) % 360).astype(int)
        quality_int_unmod = np.round(mnt_unmod[:, 3] * 100).astype(int)
        mnt_unmod_out = np.column_stack([
            mnt_unmod[:, 0].astype(int),
            mnt_unmod[:, 1].astype(int),
            angle_ccw_deg_unmod,
            quality_int_unmod,
        ])
        mnt_unmod_path = os.path.join(output_path, "minutiae_unmod", rel_dir, f"{base_name}.min")
        os.makedirs(os.path.dirname(mnt_unmod_path), exist_ok=True)
        np.savetxt(
            mnt_unmod_path, mnt_unmod_out,
            fmt="%d",
            header="X Y ANGLE QUALITY", comments="#MIN ", delimiter=" "
        )


def postprocess_and_save_batch(
    raw_outputs_cpu: dict,
    batch_paths: list[str],
    batch_orig_shapes: tuple,
    padded_shape: tuple,
    output_path: str,
    mnt_degrees: bool,
    input_base_path: str = None,
    quality_mask: bool = False,
    unmodulated: bool = False,
):
    """Post-process and save results for a batch."""
    worker_id = threading.get_ident()
    logger.info("CPU worker started processing batch",
                extra={'cpu_worker_id': worker_id, 'first_image': os.path.basename(batch_paths[0])})
    try:
        with MnetTimer("Post-processing", logger):
            final_outputs = postprocess(raw_outputs_cpu, threshold=0.45,
                                        quality_mask=quality_mask, unmodulated=unmodulated)

        for i in range(len(batch_paths)):
            orig_h, orig_w = batch_orig_shapes[0][i].item(), batch_orig_shapes[1][i].item()

            minutiae = final_outputs["minutiae"][i].numpy()
            enhanced_img = final_outputs["enhanced_image"][i][:orig_h, :orig_w].numpy()
            seg_mask = final_outputs["segmentation_mask"][i][:orig_h, :orig_w].numpy()
            ori_field = final_outputs["orientation_field"][i][:orig_h, :orig_w].numpy()

            result_item = {
                "input_path": batch_paths[i],
                "minutiae": minutiae,
                "enhanced_image": enhanced_img,
                "segmentation_mask": seg_mask,
                "orientation_field": ori_field,
            }

            if quality_mask:
                result_item["quality_mask"] = final_outputs["quality_mask"][i][:orig_h, :orig_w].numpy()
            if unmodulated:
                result_item["enhanced_image_unmod"] = final_outputs["enhanced_image_unmod"][i][:orig_h, :orig_w].numpy()
                result_item["orientation_field_unmod"] = final_outputs["orientation_field_unmod"][i][:orig_h, :orig_w].numpy()
                result_item["minutiae_unmod"] = final_outputs["minutiae_unmod"][i].numpy()

            save_results(result_item, output_path, mnt_degrees, input_base_path)

        logger.info("CPU worker finished batch",
                    extra={'cpu_worker_id': worker_id, 'batch_size': len(batch_paths)})
    except Exception as e:
        warnings.warn(f"Failed post-processing batch starting with {os.path.basename(batch_paths[0])}. Error: {e}")


def create_output_directories(output_path: str, quality_mask: bool = False, unmodulated: bool = False):
    os.makedirs(os.path.join(output_path, "minutiae"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "mask"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "ori"), exist_ok=True)
    if quality_mask:
        os.makedirs(os.path.join(output_path, "quality_mask"), exist_ok=True)
    if unmodulated:
        os.makedirs(os.path.join(output_path, "enhanced_unmod"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "ori_unmod"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "minutiae_unmod"), exist_ok=True)


def setup_ddp(rank: int, world_size: int, local_device_idx: int, timeout_minutes: int = 30):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # CUDA_VISIBLE_DEVICES remaps selected physical GPUs to logical indices.
    torch.cuda.set_device(local_device_idx)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
        device_id=torch.device(f"cuda:{local_device_idx}"),
    )


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _ddp_launch_target(rank: int, world_size: int, config: dict):
    # With CUDA_VISIBLE_DEVICES already set, rank == logical CUDA index.
    runner = InferenceRunner(config)
    runner.setup(rank, world_size, rank)
    runner.run()


def run_inference(
    input_path: str,
    output_path: str,
    coarsenet_weights: str = DEFAULT_COARSENET_WEIGHTS_PATH,
    finenet_weights: str | None = None,
    gpus: int | list[int] | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    recursive: bool = True,
    mnt_degrees: bool = True,
    compile_model: bool = False,
    max_image_dim: int = 1024,
    strategy: str = 'hybrid',
    num_cpu_workers: int = 4,
    quality_mask: bool = False,
    unmodulated: bool = False,
    full: bool = False,
    use_finenet: bool = False,
    profile: bool = False,
):
    """
    Run MinutiaeNet inference on images.

    Args:
        input_path: Path to image, directory, or text file with image paths
        output_path: Directory to save results
        coarsenet_weights: Path to CoarseNet weights (.pth file)
        finenet_weights: Path to FineNet weights (.pth file), optional
        gpus: GPU configuration:
            - None or 0: Use CPU
            - int (e.g., 1): Use first N GPUs (logical IDs 0..N-1 after remap)
            - int (e.g., 2): Use 2 GPUs with DDP (logical IDs 0,1)
            - list[int] (e.g., [2,3]): Select specific physical GPUs in CLI, then use logical 0,1
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers per GPU
        recursive: Search for images recursively
        mnt_degrees: Save minutiae angles in degrees
        compile_model: Use torch.compile
        max_image_dim: Max dimension before resizing
        strategy: 'hybrid' or 'full_gpu'
        num_cpu_workers: CPU threads for post-processing
        quality_mask: Export continuous quality mask
        unmodulated: Export unmodulated outputs
        full: Export all outputs
        use_finenet: Use FineNet for minutiae verification
        profile: Enable detailed stage profiling in tqdm/logs
    """
    if full:
        quality_mask = True
        unmodulated = True

    image_paths = find_image_paths(input_path, recursive)

    if os.path.isdir(input_path):
        input_base_path = input_path
    elif os.path.isfile(input_path):
        input_base_path = os.path.dirname(input_path)
    else:
        input_base_path = None

    config = locals()

    use_cpu = (gpus is None or gpus == 0 or not torch.cuda.is_available())
    requested_world_size = 0
    if not use_cpu:
        if isinstance(gpus, int):
            requested_world_size = gpus
        elif isinstance(gpus, list):
            requested_world_size = len(gpus)
        else:
            raise ValueError(f"Unsupported GPU configuration type: {type(gpus)}")

    if not use_cpu:
        visible_cuda_count = torch.cuda.device_count()
        if requested_world_size > visible_cuda_count:
            visible_names = []
            for i in range(visible_cuda_count):
                try:
                    visible_names.append(f"cuda:{i}={torch.cuda.get_device_name(i)}")
                except Exception:
                    visible_names.append(f"cuda:{i}=<unavailable>")
            visible_desc = ", ".join(visible_names) if visible_names else "<none>"
            raise ValueError(
                f"Requested {requested_world_size} GPU(s), but only {visible_cuda_count} "
                f"visible to CUDA. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} | "
                f"visible devices: {visible_desc}. "
                "This usually means one or more requested GPU IDs are invalid for this host, "
                "or a parent environment/scheduler already restricted visible GPUs."
            )
    is_ddp = (not use_cpu and requested_world_size > 1)

    if use_cpu:
        logger.info("Starting Inference on CPU")
        runner = InferenceRunner(config)
        runner.setup()
        runner.run()

    elif is_ddp:
        world_size = requested_world_size
        logger.info(
            f"Starting Distributed Inference on {world_size} logical GPU(s). "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

        mp.spawn(
            _ddp_launch_target,
            nprocs=world_size,
            args=(world_size, config),
            join=True,
        )
    else:
        logger.info(
            "Starting Inference on single logical GPU (cuda:0). "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
        config['gpus'] = True
        runner = InferenceRunner(config)
        runner.setup(rank=-1, world_size=1, gpu_id=0)
        runner.run()


def _save_results_chunk(
    results_chunk: list[dict],
    output_path: str,
    mnt_degrees: bool,
    worker_rank: int = -1,
    input_base_path: str = None,
):
    desc = f"Saving (Worker {worker_rank})" if worker_rank >= 0 else "Saving Results"
    for result_item in results_chunk:
        try:
            save_results(result_item, output_path, mnt_degrees, input_base_path)
        except Exception as e:
            base_name = os.path.basename(result_item.get('input_path', 'unknown_file'))
            logger.warning(f"Failed to save result for {base_name}. Error: {e}")


class InferenceRunner:
    def __init__(self, config: dict):
        self.config = config
        self.strategy = None
        self.rank = -1
        self.world_size = 1
        self.device = "cpu"
        self.is_main_process = True
        self.model = None
        self.dataloader = None

    def setup(self, rank: int = -1, world_size: int = 1, gpu_id: int = 0):
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank <= 0)

        if self.world_size > 1:
            setup_ddp(self.rank, self.world_size, gpu_id)
            self.device = f"cuda:{gpu_id}"
        elif self.config['gpus'] and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        if self.is_main_process:
            logger.info(f"Setting up runner on device: {self.device}")
            create_output_directories(
                self.config['output_path'],
                quality_mask=self.config.get('quality_mask', False),
                unmodulated=self.config.get('unmodulated', False),
            )

        if self.world_size > 1:
            dist.barrier()

        self.model = get_minutiaenet(
            coarsenet_weights=self.config['coarsenet_weights'],
            finenet_weights=self.config.get('finenet_weights'),
            device=self.device,
        )
        if self.config['compile_model']:
            if self.is_main_process:
                logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        dataset = FingerprintDataset(self.config['image_paths'], max_dim=self.config['max_image_dim'])
        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False
            )

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=(self.config['num_workers'] > 0),
            prefetch_factor=(4 if self.config['num_workers'] > 0 else None),
            collate_fn=dynamic_padding_collate,
        )

        self.strategy = self.config['strategy']

    def run(self):
        if self.is_main_process:
            logger.info(f"Executing with strategy: '{self.strategy}'")

        if self.strategy == 'hybrid':
            self._run_hybrid()
        elif self.strategy == 'full_gpu':
            self._run_full_gpu()
        else:
            raise ValueError(f"Unknown execution strategy: {self.strategy}")

        if self.world_size > 1:
            dist.barrier()
            cleanup_ddp()

        if self.is_main_process:
            logger.info("Inference Complete!")

    def _run_hybrid(self):
        if self.is_main_process:
            logger.info("Starting inference loop...")

        num_cpu_workers = self.config['num_cpu_workers']
        profile_enabled = self.config.get('profile', False)
        profile_metrics = _init_profile_metrics() if profile_enabled else None
        prev_batch_end = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_cpu_workers) as executor:
            futures = []
            max_queue_size = 2 * num_cpu_workers

            with torch.no_grad():
                desc = f"GPU {self.rank}" if self.world_size > 1 else "Processing"
                iterator = tqdm(self.dataloader, desc=desc, disable=not self.is_main_process)

                for batch_tensors, batch_paths, batch_orig_shapes in iterator:
                    batch_loop_start = time.perf_counter()
                    load_ms = (batch_loop_start - prev_batch_end) * 1000.0

                    if batch_tensors is None:
                        prev_batch_end = time.perf_counter()
                        continue

                    _, _, padded_h, padded_w = batch_tensors.shape

                    if self.device.startswith("cuda"):
                        ev_h2d_start = torch.cuda.Event(enable_timing=True)
                        ev_h2d_end = torch.cuda.Event(enable_timing=True)
                        ev_gpu_end = torch.cuda.Event(enable_timing=True)
                        ev_h2d_start.record()
                        batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                        ev_h2d_end.record()
                        raw_outputs = self.model.coarsenet(batch_tensors)
                        ev_gpu_end.record()
                        torch.cuda.synchronize(self.device)
                        h2d_ms = ev_h2d_start.elapsed_time(ev_h2d_end)
                        gpu_ms = ev_h2d_end.elapsed_time(ev_gpu_end)
                    else:
                        t_h2d_start = time.perf_counter()
                        batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                        t_h2d_end = time.perf_counter()
                        raw_outputs = self.model.coarsenet(batch_tensors)
                        t_gpu_end = time.perf_counter()
                        h2d_ms = (t_h2d_end - t_h2d_start) * 1000.0
                        gpu_ms = (t_gpu_end - t_h2d_end) * 1000.0

                    t_d2h_start = time.perf_counter()
                    raw_outputs_cpu = {k: v.detach().cpu() for k, v in raw_outputs.items()}
                    d2h_ms = (time.perf_counter() - t_d2h_start) * 1000.0

                    t_submit_start = time.perf_counter()
                    future = executor.submit(
                        postprocess_and_save_batch,
                        raw_outputs_cpu, batch_paths, batch_orig_shapes,
                        (padded_h, padded_w), self.config['output_path'], self.config['mnt_degrees'],
                        self.config.get('input_base_path'),
                        self.config.get('quality_mask', False),
                        self.config.get('unmodulated', False),
                    )
                    futures.append(future)
                    submit_ms = (time.perf_counter() - t_submit_start) * 1000.0

                    queue_wait_ms = 0.0
                    if len(futures) >= max_queue_size:
                        t_wait_start = time.perf_counter()
                        futures.pop(0).result()
                        queue_wait_ms = (time.perf_counter() - t_wait_start) * 1000.0

                    if profile_enabled and self.is_main_process:
                        profile_metrics['load_ms'].append(load_ms)
                        profile_metrics['h2d_ms'].append(h2d_ms)
                        profile_metrics['coarse_gpu_ms'].append(gpu_ms)
                        profile_metrics['post_cpu_ms'].append(0.0)
                        profile_metrics['post_mask_ms'].append(0.0)
                        profile_metrics['post_minutiae_ms'].append(0.0)
                        profile_metrics['post_orientation_ms'].append(0.0)
                        profile_metrics['post_enhanced_ms'].append(0.0)
                        profile_metrics['post_optional_ms'].append(0.0)
                        profile_metrics['finenet_ms'].append(0.0)
                        profile_metrics['d2h_ms'].append(d2h_ms)
                        profile_metrics['pack_ms'].append(0.0)
                        profile_metrics['submit_ms'].append(submit_ms)
                        profile_metrics['queue_wait_ms'].append(queue_wait_ms)
                        _update_profile_tqdm(iterator, profile_metrics, len(futures))

                    prev_batch_end = time.perf_counter()

            if self.is_main_process:
                logger.info("Inference complete. Finalizing post-processing...")
            for future in tqdm(futures, desc=f"Finalizing (Worker {self.rank})", disable=not self.is_main_process):
                future.result()

        if profile_enabled and self.is_main_process:
            _log_profile_summary("hybrid", profile_metrics)

    def _run_full_gpu(self):
        num_save_workers = self.config['num_cpu_workers']
        chunk_size = self.config['batch_size'] * 10
        profile_enabled = self.config.get('profile', False)
        profile_metrics = _init_profile_metrics() if profile_enabled else None
        prev_batch_end = time.perf_counter()
        max_save_queue_size = 2 * num_save_workers

        with ThreadPoolExecutor(max_workers=num_save_workers) as save_executor:
            futures = []
            chunk_to_save = []

            with torch.no_grad():
                desc = f"GPU {self.rank}" if self.world_size > 1 else "Processing"
                iterator = tqdm(self.dataloader, desc=desc, disable=not self.is_main_process)

                for batch_tensors, batch_paths, batch_orig_shapes in iterator:
                    batch_loop_start = time.perf_counter()
                    load_ms = (batch_loop_start - prev_batch_end) * 1000.0

                    if batch_tensors is None:
                        prev_batch_end = time.perf_counter()
                        continue

                    if self.device.startswith("cuda"):
                        ev_h2d_start = torch.cuda.Event(enable_timing=True)
                        ev_h2d_end = torch.cuda.Event(enable_timing=True)
                        ev_gpu_end = torch.cuda.Event(enable_timing=True)
                        ev_h2d_start.record()
                        batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                        ev_h2d_end.record()
                    else:
                        t_h2d_start = time.perf_counter()
                        batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                        t_h2d_end = time.perf_counter()

                    _qm = self.config.get('quality_mask', False)
                    _um = self.config.get('unmodulated', False)
                    _uf = self.config.get('use_finenet', False)

                    if profile_enabled:
                        padded_tensors = self.model.preprocess(batch_tensors)

                        if self.device.startswith("cuda"):
                            ev_coarse_end = torch.cuda.Event(enable_timing=True)
                            raw_outputs = self.model.coarsenet(padded_tensors)
                            ev_coarse_end.record()
                            torch.cuda.synchronize(self.device)
                            h2d_ms = ev_h2d_start.elapsed_time(ev_h2d_end)
                            coarse_gpu_ms = ev_h2d_end.elapsed_time(ev_coarse_end)
                        else:
                            t_coarse_end = time.perf_counter()
                            raw_outputs = self.model.coarsenet(padded_tensors)
                            t_coarse_done = time.perf_counter()
                            h2d_ms = (t_h2d_end - t_h2d_start) * 1000.0
                            coarse_gpu_ms = (t_coarse_done - t_coarse_end) * 1000.0

                        t_post_start = time.perf_counter()
                        final_outputs, post_breakdown = postprocess(
                            raw_outputs,
                            threshold=0.45,
                            quality_mask=_qm,
                            unmodulated=_um,
                            return_timings=True,
                        )
                        post_cpu_ms = (time.perf_counter() - t_post_start) * 1000.0

                        finenet_ms = 0.0
                        if _uf and self.model.finenet is not None:
                            t_fine_start = time.perf_counter()
                            final_outputs = self.model._apply_finenet(padded_tensors, final_outputs, threshold=0.45)
                            finenet_ms = (time.perf_counter() - t_fine_start) * 1000.0
                    else:
                        final_outputs = self.model(batch_tensors, quality_mask=_qm, unmodulated=_um, use_finenet=_uf)

                        if self.device.startswith("cuda"):
                            ev_gpu_end.record()
                            torch.cuda.synchronize(self.device)
                            h2d_ms = ev_h2d_start.elapsed_time(ev_h2d_end)
                            coarse_gpu_ms = ev_h2d_end.elapsed_time(ev_gpu_end)
                        else:
                            t_gpu_end = time.perf_counter()
                            h2d_ms = (t_h2d_end - t_h2d_start) * 1000.0
                            coarse_gpu_ms = (t_gpu_end - t_h2d_end) * 1000.0
                        post_cpu_ms = 0.0
                        finenet_ms = 0.0

                    t_pack_start = time.perf_counter()
                    for i in range(len(batch_paths)):
                        orig_h, orig_w = batch_orig_shapes[0][i].item(), batch_orig_shapes[1][i].item()
                        result_item = {
                            'input_path': batch_paths[i],
                            'minutiae': final_outputs['minutiae'][i].cpu().numpy(),
                            'enhanced_image': final_outputs['enhanced_image'][i][:orig_h, :orig_w].cpu().numpy(),
                            'segmentation_mask': final_outputs['segmentation_mask'][i][:orig_h, :orig_w].cpu().numpy(),
                            'orientation_field': final_outputs['orientation_field'][i][:orig_h, :orig_w].cpu().numpy(),
                        }
                        if _qm and 'quality_mask' in final_outputs:
                            result_item['quality_mask'] = final_outputs['quality_mask'][i][:orig_h, :orig_w].cpu().numpy()
                        if _um:
                            result_item['enhanced_image_unmod'] = final_outputs['enhanced_image_unmod'][i][:orig_h, :orig_w].cpu().numpy()
                            result_item['orientation_field_unmod'] = final_outputs['orientation_field_unmod'][i][:orig_h, :orig_w].cpu().numpy()
                            result_item['minutiae_unmod'] = final_outputs['minutiae_unmod'][i].cpu().numpy()
                        chunk_to_save.append(result_item)
                    pack_ms = (time.perf_counter() - t_pack_start) * 1000.0

                    submit_ms = 0.0
                    if len(chunk_to_save) >= chunk_size:
                        t_submit_start = time.perf_counter()
                        future = save_executor.submit(
                            _save_results_chunk,
                            chunk_to_save,
                            self.config['output_path'],
                            self.config['mnt_degrees'],
                            self.rank,
                            self.config.get('input_base_path'),
                        )
                        futures.append(future)
                        submit_ms = (time.perf_counter() - t_submit_start) * 1000.0
                        chunk_to_save = []

                    queue_wait_ms = 0.0
                    if len(futures) >= max_save_queue_size:
                        t_wait_start = time.perf_counter()
                        futures.pop(0).result()
                        queue_wait_ms = (time.perf_counter() - t_wait_start) * 1000.0

                    if profile_enabled and self.is_main_process:
                        profile_metrics['load_ms'].append(load_ms)
                        profile_metrics['h2d_ms'].append(h2d_ms)
                        profile_metrics['coarse_gpu_ms'].append(coarse_gpu_ms)
                        profile_metrics['post_cpu_ms'].append(post_cpu_ms)
                        profile_metrics['post_mask_ms'].append(post_breakdown['mask_ms'])
                        profile_metrics['post_minutiae_ms'].append(post_breakdown['minutiae_ms'])
                        profile_metrics['post_orientation_ms'].append(post_breakdown['orientation_ms'])
                        profile_metrics['post_enhanced_ms'].append(post_breakdown['enhanced_ms'])
                        profile_metrics['post_optional_ms'].append(post_breakdown['optional_ms'])
                        profile_metrics['finenet_ms'].append(finenet_ms)
                        profile_metrics['d2h_ms'].append(0.0)
                        profile_metrics['pack_ms'].append(pack_ms)
                        profile_metrics['submit_ms'].append(submit_ms)
                        profile_metrics['queue_wait_ms'].append(queue_wait_ms)
                        _update_profile_tqdm(iterator, profile_metrics, len(futures))

                    prev_batch_end = time.perf_counter()

            if chunk_to_save:
                future = save_executor.submit(
                    _save_results_chunk,
                    chunk_to_save,
                    self.config['output_path'],
                    self.config['mnt_degrees'],
                    self.rank,
                    self.config.get('input_base_path'),
                )
                futures.append(future)

            if self.is_main_process:
                logger.info("Inference complete. Waiting for save workers to finish...")

            for future in tqdm(futures, desc=f"Finalizing Save (Worker {self.rank})", disable=not self.is_main_process):
                future.result()

        if profile_enabled and self.is_main_process:
            _log_profile_summary("full_gpu", profile_metrics)
