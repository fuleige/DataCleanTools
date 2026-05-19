from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SigLIP/SigLIP2 image embeddings with multi-process multi-GPU workers.")
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL manifest with local_path fields.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for embeddings.npy, ids.json, and refs JSONL.")
    parser.add_argument("--model-path", required=True, type=str, help="Local Hugging Face model directory.")
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated CUDA device ids. Repeat ids to run multiple workers per GPU.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per worker.")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--dimension", type=int, default=768, help="Output embedding dimension.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N image limit for smoke tests.")
    parser.add_argument("--progress-interval", type=float, default=10.0, help="Seconds between progress lines.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embedding outputs.")
    return parser.parse_args()


def read_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            local_path = row.get("local_path")
            if not local_path:
                continue
            rows.append(
                {
                    "image_id": row["image_id"],
                    "uri": row.get("uri"),
                    "local_path": local_path,
                    "quality_status": row.get("quality_status", "keep"),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def split_contiguous(rows: list[dict[str, Any]], workers: int) -> list[tuple[int, list[dict[str, Any]]]]:
    chunks: list[tuple[int, list[dict[str, Any]]]] = []
    total = len(rows)
    for worker_idx in range(workers):
        start = total * worker_idx // workers
        end = total * (worker_idx + 1) // workers
        chunks.append((start, rows[start:end]))
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


def encode_worker(payload: dict[str, Any], progress_queue: mp.Queue) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    worker_id = int(payload["worker_id"])
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, SiglipVisionModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = True

        device_id = str(payload["device_id"])
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            torch.cuda.set_device(int(device_id))

        load_started = time.perf_counter()
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                processor = AutoImageProcessor.from_pretrained(payload["model_path"], local_files_only=True)
                dtype = dtype_for(torch, str(payload["dtype"]), device)
                try:
                    model = SiglipVisionModel.from_pretrained(payload["model_path"], dtype=dtype, local_files_only=True)
                except TypeError:
                    model = SiglipVisionModel.from_pretrained(payload["model_path"], torch_dtype=dtype, local_files_only=True)
        model = model.to(device)
        model.eval()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        progress_queue.put({"type": "loaded", "worker_id": worker_id, "load_seconds": time.perf_counter() - load_started})

        rows: list[dict[str, Any]] = payload["rows"]
        start_index = int(payload["start_index"])
        batch_size = int(payload["batch_size"])
        expected_dimension = int(payload["dimension"])
        vector_path = Path(payload["vector_path"])
        vector_memmap = np.lib.format.open_memmap(
            vector_path,
            mode="r+",
            dtype=np.float32,
            shape=(int(payload["total_rows"]), expected_dimension),
        )

        encoded = 0
        embed_started = time.perf_counter()
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            images = []
            for row in batch_rows:
                with Image.open(row["local_path"]) as image:
                    images.append(image.convert("RGB").copy())
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            with torch.inference_mode():
                features = feature_tensor(model(**inputs))
                features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
            if int(features.shape[1]) != expected_dimension:
                raise RuntimeError(f"worker {worker_id} got dimension {features.shape[1]}, expected {expected_dimension}")
            vectors = features.detach().cpu().numpy().astype(np.float32, copy=False)
            write_start = start_index + batch_start
            vector_memmap[write_start : write_start + vectors.shape[0]] = vectors
            encoded += int(vectors.shape[0])
            progress_queue.put({"type": "progress", "worker_id": worker_id, "count": int(vectors.shape[0])})

        vector_memmap.flush()
        del vector_memmap
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        progress_queue.put(
            {
                "type": "done",
                "worker_id": worker_id,
                "device": device,
                "encoded_count": encoded,
                "embed_seconds": time.perf_counter() - embed_started,
            }
        )
    except BaseException:
        progress_queue.put({"type": "error", "worker_id": worker_id, "traceback": traceback.format_exc()})


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "vectors": output_dir / "embeddings.npy",
        "vectors_tmp": output_dir / "embeddings.tmp.npy",
        "ids": output_dir / "ids.json",
        "ids_tmp": output_dir / "ids.tmp.json",
        "refs": output_dir / "embedding_refs.jsonl",
        "refs_tmp": output_dir / "embedding_refs.tmp.jsonl",
        "summary": output_dir / "embedding_summary.json",
        "summary_tmp": output_dir / "embedding_summary.tmp.json",
    }


def ensure_output_available(paths: dict[str, Path], overwrite: bool) -> None:
    final_paths = [paths["vectors"], paths["ids"], paths["refs"], paths["summary"]]
    existing = [path for path in final_paths if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise SystemExit(f"output already exists; pass --overwrite to replace: {existing_text}")
    for path in paths.values():
        if path.exists() and (overwrite or path.name.endswith(".tmp") or ".tmp." in path.name):
            path.unlink()


def write_metadata(rows: list[dict[str, Any]], paths: dict[str, Path], embedding_uri: str) -> None:
    with paths["ids_tmp"].open("w", encoding="utf-8") as ids_f, paths["refs_tmp"].open("w", encoding="utf-8") as refs_f:
        ids_f.write("[")
        for row_index, row in enumerate(rows):
            if row_index:
                ids_f.write(",")
            ids_f.write(json.dumps(row["image_id"], ensure_ascii=False))
            refs_f.write(
                json.dumps(
                    {
                        "image_id": row["image_id"],
                        "uri": row.get("uri"),
                        "local_path": row["local_path"],
                        "quality_status": row.get("quality_status", "keep"),
                        "row_index": row_index,
                        "embedding_uri": embedding_uri,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            refs_f.write("\n")
        ids_f.write("]\n")


def print_progress(total: int, processed: int, started: float, loaded_workers: int, done_workers: int, worker_count: int) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = processed / elapsed
    remaining = total - processed
    eta = remaining / rate if rate > 0 else None
    eta_text = f"{eta / 60:.1f} min" if eta is not None else "unknown"
    print(
        f"[embedding] processed={processed}/{total} ({processed / total * 100:.2f}%) "
        f"rate={rate:.1f} img/s eta={eta_text} loaded={loaded_workers}/{worker_count} done={done_workers}/{worker_count}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("--devices must contain at least one device id")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.dimension <= 0:
        raise SystemExit("--dimension must be > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir)
    ensure_output_available(paths, args.overwrite)

    print(f"[embedding] reading manifest: {args.manifest}", flush=True)
    rows = read_manifest(args.manifest, args.limit)
    if not rows:
        raise SystemExit("manifest does not contain any rows with local_path")
    total = len(rows)
    print(f"[embedding] rows={total} workers={len(devices)} batch_size={args.batch_size}", flush=True)

    vector_memmap = np.lib.format.open_memmap(
        paths["vectors_tmp"],
        mode="w+",
        dtype=np.float32,
        shape=(total, args.dimension),
    )
    vector_memmap.flush()
    del vector_memmap

    chunks = split_contiguous(rows, len(devices))
    ctx = mp.get_context("spawn")
    progress_queue: mp.Queue = ctx.Queue()
    processes: list[mp.Process] = []
    started = time.perf_counter()

    for worker_id, (device_id, (start_index, chunk_rows)) in enumerate(zip(devices, chunks, strict=True)):
        process = ctx.Process(
            target=encode_worker,
            args=(
                {
                    "worker_id": worker_id,
                    "device_id": device_id,
                    "rows": chunk_rows,
                    "start_index": start_index,
                    "batch_size": args.batch_size,
                    "dimension": args.dimension,
                    "total_rows": total,
                    "vector_path": str(paths["vectors_tmp"]),
                    "model_path": args.model_path,
                    "dtype": args.dtype,
                },
                progress_queue,
            ),
        )
        process.start()
        processes.append(process)

    processed = 0
    loaded_workers = 0
    done_workers = 0
    worker_results: list[dict[str, Any]] = []
    last_print = 0.0
    failed = False

    try:
        while done_workers < len(processes):
            now = time.perf_counter()
            try:
                message = progress_queue.get(timeout=1.0)
            except queue.Empty:
                message = None
            if message is not None:
                message_type = message.get("type")
                if message_type == "loaded":
                    loaded_workers += 1
                elif message_type == "progress":
                    processed += int(message["count"])
                elif message_type == "done":
                    done_workers += 1
                    worker_results.append(message)
                elif message_type == "error":
                    failed = True
                    print(message["traceback"], file=sys.stderr, flush=True)
                    break
            if now - last_print >= args.progress_interval:
                print_progress(total, processed, started, loaded_workers, done_workers, len(processes))
                last_print = now
            for process in processes:
                if process.exitcode not in (None, 0):
                    failed = True
                    print(f"worker process {process.pid} exited with code {process.exitcode}", file=sys.stderr, flush=True)
                    break
            if failed:
                break
    finally:
        if failed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join()

    if failed or processed != total:
        raise SystemExit(f"embedding extraction failed or incomplete: processed={processed}, total={total}")

    print_progress(total, processed, started, loaded_workers, done_workers, len(processes))
    print("[embedding] writing ids and refs", flush=True)
    embedding_uri = paths["vectors"].resolve().as_uri()
    write_metadata(rows, paths, embedding_uri)

    summary = {
        "stage": "embedding",
        "provider": "siglip2",
        "model_name": Path(args.model_path).name or str(args.model_path),
        "model_version": str(args.model_path),
        "model_mode": "vision",
        "dimension": args.dimension,
        "candidate_count": total,
        "success_count": processed,
        "failed_count": 0,
        "batch_size": args.batch_size,
        "worker_count": len(devices),
        "devices": devices,
        "dtype": args.dtype,
        "elapsed_seconds": time.perf_counter() - started,
        "throughput_images_per_second": processed / max(time.perf_counter() - started, 1e-9),
        "embedding_uri": embedding_uri,
        "workers": sorted(worker_results, key=lambda item: int(item["worker_id"])),
    }
    paths["summary_tmp"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    paths["vectors_tmp"].replace(paths["vectors"])
    paths["ids_tmp"].replace(paths["ids"])
    paths["refs_tmp"].replace(paths["refs"])
    paths["summary_tmp"].replace(paths["summary"])
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
