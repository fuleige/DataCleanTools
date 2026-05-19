from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SigLIP/SigLIP2 image embedding throughput.")
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL manifest with local_path fields.")
    parser.add_argument("--model-path", required=True, type=str, help="Local Hugging Face model directory or model id.")
    parser.add_argument("--sample-size", type=int, default=8192, help="Number of images to benchmark.")
    parser.add_argument("--batch-size", type=int, default=256, help="Per-GPU batch size.")
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated CUDA device ids.")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--model-mode", default="vision", choices=["vision", "full"], help="Load only SigLIP vision tower or full SigLIP model.")
    parser.add_argument("--sample-mode", default="first", choices=["first", "even"])
    parser.add_argument("--total-count", type=int, default=None, help="Total dataset count for ETA estimation.")
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def load_paths(manifest: Path, sample_size: int, sample_mode: str) -> tuple[list[str], int]:
    if sample_mode == "first":
        paths = []
        total_seen = 0
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total_seen += 1
                if len(paths) < sample_size:
                    row = json.loads(line)
                    local_path = row.get("local_path")
                    if local_path:
                        paths.append(local_path)
        return paths, total_seen

    total_lines = count_jsonl(manifest)
    if total_lines == 0:
        return [], 0
    sample_size = min(sample_size, total_lines)
    targets = {math.floor(i * total_lines / sample_size) + 1 for i in range(sample_size)}
    paths = []
    with manifest.open("r", encoding="utf-8") as handle:
        row_idx = 0
        for line in handle:
            if not line.strip():
                continue
            row_idx += 1
            if row_idx not in targets:
                continue
            row = json.loads(line)
            local_path = row.get("local_path")
            if local_path:
                paths.append(local_path)
    return paths, total_lines


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def split_paths(paths: list[str], workers: int) -> list[list[str]]:
    chunks = [[] for _ in range(workers)]
    for idx, path in enumerate(paths):
        chunks[idx % workers].append(path)
    return chunks


def dtype_for(torch: Any, dtype: str, device: str) -> Any:
    if dtype == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def feature_tensor(output: Any) -> Any:
    if hasattr(output, "float"):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state[:, 0]
    raise RuntimeError("model output does not contain image features")


def worker_main(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device_id = payload["device_id"]
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(int(device_id)) if device.startswith("cuda") else None

    load_started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(payload["model_path"], local_files_only=True)
    dtype = dtype_for(torch, payload["dtype"], device)
    model_cls = SiglipVisionModel if payload["model_mode"] == "vision" else AutoModel
    try:
        model = model_cls.from_pretrained(payload["model_path"], dtype=dtype, local_files_only=True).to(device)
    except TypeError:
        model = model_cls.from_pretrained(payload["model_path"], torch_dtype=dtype, local_files_only=True).to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    paths: list[str] = payload["paths"]
    batch_size = int(payload["batch_size"])
    warmup_batches = int(payload["warmup_batches"])
    dimension = 0
    failed = 0

    def encode_batch(batch_paths: list[str]) -> int:
        nonlocal dimension, failed
        images = []
        for image_path in batch_paths:
            try:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
            except Exception:
                failed += 1
        if not images:
            return 0
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        with torch.inference_mode():
            if payload["model_mode"] == "vision":
                features = feature_tensor(model(**inputs))
            elif hasattr(model, "get_image_features"):
                features = feature_tensor(model.get_image_features(**inputs))
            else:
                features = feature_tensor(model.vision_model(**inputs))
                if hasattr(model, "visual_projection"):
                    features = model.visual_projection(features)
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
        dimension = int(features.shape[1])
        return int(features.shape[0])

    for warmup_idx in range(warmup_batches):
        start = warmup_idx * batch_size
        if start >= len(paths):
            break
        encode_batch(paths[start : start + batch_size])
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    embed_started = time.perf_counter()
    encoded = 0
    for start in range(0, len(paths), batch_size):
        encoded += encode_batch(paths[start : start + batch_size])
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    embed_seconds = time.perf_counter() - embed_started

    return {
        "device": device,
        "input_count": len(paths),
        "encoded_count": encoded,
        "failed_count": failed,
        "dimension": dimension,
        "load_seconds": load_seconds,
        "embed_seconds": embed_seconds,
        "images_per_second": encoded / embed_seconds if embed_seconds else 0.0,
    }


def main() -> None:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("--devices must contain at least one device id")
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.warmup_batches < 0:
        raise SystemExit("--warmup-batches must be >= 0")
    paths, observed_total = load_paths(args.manifest, args.sample_size, args.sample_mode)
    if not paths:
        raise SystemExit("manifest does not contain any rows with local_path")
    chunks = split_paths(paths, len(devices))
    total_count = args.total_count or observed_total

    benchmark_started = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(devices)) as pool:
        results = pool.map(
            worker_main,
            [
                {
                    "device_id": device,
                    "paths": chunk,
                    "model_path": args.model_path,
                    "batch_size": args.batch_size,
                    "dtype": args.dtype,
                    "warmup_batches": args.warmup_batches,
                    "model_mode": args.model_mode,
                }
                for device, chunk in zip(devices, chunks, strict=True)
            ],
        )
    wall_seconds = time.perf_counter() - benchmark_started

    encoded = sum(item["encoded_count"] for item in results)
    max_embed_seconds = max((item["embed_seconds"] for item in results), default=0.0)
    throughput_excluding_load = encoded / max_embed_seconds if max_embed_seconds else 0.0
    throughput_including_load = encoded / wall_seconds if wall_seconds else 0.0
    eta_seconds = total_count / throughput_excluding_load if throughput_excluding_load else None
    payload = {
        "manifest": str(args.manifest),
        "model_path": args.model_path,
        "sample_mode": args.sample_mode,
        "sample_size": len(paths),
        "batch_size_per_gpu": args.batch_size,
        "devices": devices,
        "dtype": args.dtype,
        "model_mode": args.model_mode,
        "total_count_for_eta": total_count,
        "wall_seconds_including_load": wall_seconds,
        "encoded_count": encoded,
        "throughput_images_per_second_excluding_model_load": throughput_excluding_load,
        "throughput_images_per_second_including_model_load": throughput_including_load,
        "estimated_full_dataset_seconds_excluding_model_load": eta_seconds,
        "estimated_full_dataset_hours_excluding_model_load": eta_seconds / 3600 if eta_seconds else None,
        "workers": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
