from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS IVF-Flat vector index on GPU.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing embeddings/.")
    parser.add_argument("--vectors", type=Path, default=None, help="Path to embeddings.npy.")
    parser.add_argument("--ids", type=Path, default=None, help="Path to ids.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for FAISS index and metadata.")
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated CUDA device ids.")
    parser.add_argument("--nlist", type=int, default=16384, help="IVF centroid/list count.")
    parser.add_argument("--nprobe", type=int, default=64, help="IVF probes for optional candidate search.")
    parser.add_argument("--train-size", type=int, default=524288, help="Number of vectors sampled for IVF training.")
    parser.add_argument("--add-batch-size", type=int, default=100000, help="Vectors added per batch.")
    parser.add_argument("--search-batch-size", type=int, default=8192, help="Queries searched per batch.")
    parser.add_argument("--top-k", type=int, default=100, help="Candidate count when --write-candidates is enabled.")
    parser.add_argument("--use-float16", action=argparse.BooleanOptionalAction, default=True, help="Store/search GPU index in float16.")
    parser.add_argument(
        "--shard",
        action="store_true",
        help="Shard database across GPUs instead of replicating it. Uses less memory, usually less throughput.",
    )
    parser.add_argument(
        "--indices-options",
        default="32",
        choices=["32", "64", "cpu"],
        help="How FAISS stores vector ids on GPU.",
    )
    parser.add_argument("--temp-memory-mib", type=int, default=None, help="Optional FAISS GPU temp memory per device.")
    parser.add_argument("--write-candidates", action="store_true", help="Also search all vectors and write hnsw_candidates.jsonl.")
    parser.add_argument("--save-index", action=argparse.BooleanOptionalAction, default=True, help="Save CPU FAISS index to disk.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N vector limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--progress-interval", type=float, default=10.0, help="Seconds between progress lines.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.run_dir:
        run_dir = args.run_dir
        vectors = args.vectors or run_dir / "embeddings" / "embeddings.npy"
        ids = args.ids or run_dir / "embeddings" / "ids.json"
        output_dir = args.output_dir or run_dir / "indexes"
    else:
        if not args.vectors or not args.ids or not args.output_dir:
            raise SystemExit("provide either --run-dir or all of --vectors, --ids, and --output-dir")
        run_dir = args.output_dir.parent
        vectors = args.vectors
        ids = args.ids
        output_dir = args.output_dir
    return {"run_dir": run_dir, "vectors": vectors, "ids": ids, "output_dir": output_dir}


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "index": output_dir / "faiss_ivf_flat.index",
        "index_tmp": output_dir / "faiss_ivf_flat.tmp.index",
        "metadata": output_dir / "index_metadata.json",
        "metadata_tmp": output_dir / "index_metadata.tmp.json",
        "candidates": output_dir / "hnsw_candidates.jsonl",
        "candidates_tmp": output_dir / "hnsw_candidates.tmp.jsonl",
    }


def stage_output_paths(run_dir: Path) -> dict[str, Path]:
    summary_dir = run_dir / "summaries" / "vector_index"
    return {
        "summary_dir": summary_dir,
        "summary": summary_dir / "stage_summary.json",
        "sample": summary_dir / "sample.jsonl",
        "errors": summary_dir / "errors.jsonl",
    }


def ensure_outputs(paths: dict[str, Path], *, overwrite: bool, write_candidates: bool, save_index: bool) -> None:
    final_paths = [paths["metadata"]]
    if save_index:
        final_paths.append(paths["index"])
    if write_candidates:
        final_paths.append(paths["candidates"])
    existing = [path for path in final_paths if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise SystemExit(f"output already exists; pass --overwrite to replace: {existing_text}")
    for key, path in paths.items():
        if path.exists() and (overwrite or "tmp" in key):
            path.unlink()


def load_ids(path: Path, limit: int | None) -> list[str]:
    ids = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        ids = ids[:limit]
    return ids


def sample_training_vectors(vectors: np.ndarray, train_size: int) -> np.ndarray:
    total = int(vectors.shape[0])
    train_count = min(max(1, train_size), total)
    if train_count == total:
        return np.asarray(vectors, dtype=np.float32)
    indices = np.linspace(0, total - 1, train_count, dtype=np.int64)
    return np.asarray(vectors[indices], dtype=np.float32)


def gpu_index_nprobe(index: Any, nprobe: int) -> None:
    if hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)
    if hasattr(index, "count") and hasattr(index, "at"):
        import faiss  # type: ignore

        for idx in range(index.count()):
            child = faiss.downcast_index(index.at(idx))
            if hasattr(child, "nprobe"):
                child.nprobe = int(nprobe)


def indices_option(faiss: Any, value: str) -> Any:
    if value == "32":
        return faiss.INDICES_32_BIT
    if value == "64":
        return faiss.INDICES_64_BIT
    return faiss.INDICES_CPU


def build_gpu_index(vectors: np.ndarray, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import faiss  # type: ignore

    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("FAISS GPU support is unavailable; install faiss-gpu-cu12 in the runtime environment.")
    device_ids = [int(item.strip()) for item in args.devices.split(",") if item.strip()]
    if not device_ids:
        raise SystemExit("--devices must include at least one CUDA device id")

    total = int(vectors.shape[0])
    if total <= 0:
        raise SystemExit("vectors must contain at least one row")
    dimension = int(vectors.shape[1])
    train_count = min(max(1, int(args.train_size)), total)
    nlist = min(int(args.nlist), train_count)
    faiss.omp_set_num_threads(max(1, min(os.cpu_count() or 1, 32)))

    resources = []
    for _ in device_ids:
        resource = faiss.StandardGpuResources()
        if args.temp_memory_mib is not None:
            resource.setTempMemory(int(args.temp_memory_mib) * 1024 * 1024)
        resources.append(resource)

    cloner_options = faiss.GpuMultipleClonerOptions()
    cloner_options.shard = bool(args.shard)
    cloner_options.useFloat16 = bool(args.use_float16)
    cloner_options.indicesOptions = indices_option(faiss, args.indices_options)

    cpu_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    gpu_index = faiss.index_cpu_to_gpu_multiple_py(resources, cpu_index, co=cloner_options, gpus=device_ids)

    started = time.perf_counter()
    train_vectors = sample_training_vectors(vectors, train_count)
    gpu_index.train(train_vectors)
    train_seconds = time.perf_counter() - started
    del train_vectors

    add_started = time.perf_counter()
    last_print = 0.0
    for start in range(0, total, int(args.add_batch_size)):
        end = min(start + int(args.add_batch_size), total)
        gpu_index.add(np.asarray(vectors[start:end], dtype=np.float32))
        now = time.perf_counter()
        if now - last_print >= float(args.progress_interval):
            print(f"[index] added={end}/{total} ({end / total * 100:.2f}%)", flush=True)
            last_print = now
    add_seconds = time.perf_counter() - add_started
    gpu_index_nprobe(gpu_index, int(args.nprobe))

    metrics = {
        "device_ids": device_ids,
        "dimension": dimension,
        "index_type": "ivf_flat",
        "metric": "cosine",
        "nlist": nlist,
        "nprobe": int(args.nprobe),
        "train_size": train_count,
        "use_float16": bool(args.use_float16),
        "shard": bool(args.shard),
        "indices_options": args.indices_options,
        "train_seconds": train_seconds,
        "add_seconds": add_seconds,
    }
    return gpu_index, metrics


def save_cpu_index(gpu_index: Any, path: Path) -> None:
    import faiss  # type: ignore

    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index, str(path))


def write_candidates(
    *,
    gpu_index: Any,
    vectors: np.ndarray,
    ids: list[str],
    path: Path,
    top_k: int,
    search_batch_size: int,
    progress_interval: float,
) -> dict[str, Any]:
    total = int(vectors.shape[0])
    started = time.perf_counter()
    last_print = 0.0
    rows = 0
    with path.open("w", encoding="utf-8") as handle:
        for start in range(0, total, search_batch_size):
            end = min(start + search_batch_size, total)
            queries = np.asarray(vectors[start:end], dtype=np.float32)
            distances, indices = gpu_index.search(queries, min(top_k, total))
            for offset, image_id in enumerate(ids[start:end]):
                candidates = [
                    {"image_id": ids[int(index)], "score": float(score)}
                    for index, score in zip(indices[offset], distances[offset], strict=False)
                    if int(index) >= 0
                ]
                handle.write(json.dumps({"image_id": image_id, "candidates": candidates}, ensure_ascii=False))
                handle.write("\n")
                rows += 1
            now = time.perf_counter()
            if now - last_print >= progress_interval:
                elapsed = max(now - started, 1e-9)
                rate = rows / elapsed
                remaining = (total - rows) / rate if rate > 0 else math.inf
                print(
                    f"[search] rows={rows}/{total} ({rows / total * 100:.2f}%) "
                    f"rate={rate:.1f} rows/s eta={remaining / 60:.1f} min",
                    flush=True,
                )
                last_print = now
    elapsed = time.perf_counter() - started
    return {
        "candidate_rows": rows,
        "search_seconds": elapsed,
        "search_rows_per_second": rows / elapsed if elapsed else 0.0,
    }


def sha256_file(path: Path) -> str | None:
    import hashlib

    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stage_outputs(run_dir: Path, metadata: dict[str, Any], candidates_path: Path | None) -> list[dict[str, Any]]:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return []
    paths = stage_output_paths(run_dir)
    paths["summary_dir"].mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sample_count = 0
    with paths["sample"].open("w", encoding="utf-8") as sample_f:
        if candidates_path and candidates_path.exists():
            with candidates_path.open("r", encoding="utf-8") as candidates_f:
                for line in candidates_f:
                    if sample_count >= 20:
                        break
                    sample_f.write(line)
                    sample_count += 1
    paths["errors"].write_text("", encoding="utf-8")
    return [
        artifact("vector_index_summary", "vector_index", paths["summary"], 1, metadata["created_at"]),
        artifact("vector_index_sample", "vector_index", paths["sample"], sample_count, metadata["created_at"]),
        artifact("vector_index_errors", "vector_index", paths["errors"], 0, metadata["created_at"]),
    ]


def update_run_state(run_dir: Path, metadata: dict[str, Any], artifacts: list[dict[str, Any]]) -> None:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    now = metadata["created_at"]
    previous_artifacts = state.get("artifacts", [])
    names = {item["name"] for item in artifacts}
    state["artifacts"] = [item for item in previous_artifacts if item.get("name") not in names] + artifacts
    state.setdefault("stages", {})["vector_index"] = {
        "stage_name": "vector_index",
        "status": "succeeded",
        "started_at": metadata["started_at"],
        "finished_at": now,
        "duration_ms": int(metadata["elapsed_seconds"] * 1000),
        "force": False,
        "processed_items": metadata["vector_count"],
        "total_items": metadata["vector_count"],
        "succeeded_items": metadata["vector_count"],
        "failed_items": 0,
        "artifact_uri": metadata.get("summary_uri") or metadata["metadata_uri"],
        "error_json": None,
        "metrics_json": metadata,
    }
    state["status"] = "paused"
    state["current_stage"] = "vector_index"
    state["finished_at"] = now
    state["updated_at"] = now
    (run_dir / "artifact_index.json").write_text(json.dumps(state["artifacts"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def artifact(name: str, stage: str, path: Path, records: int | None, created_at: str) -> dict[str, Any]:
    return {
        "name": name,
        "stage": stage,
        "type": "file",
        "uri": path.resolve().as_uri(),
        "records": records,
        "checksum": sha256_file(path),
        "created_at": created_at,
    }


def main() -> None:
    args = parse_args()
    if args.nlist <= 0:
        raise SystemExit("--nlist must be > 0")
    if args.nprobe <= 0:
        raise SystemExit("--nprobe must be > 0")
    if args.train_size <= 0:
        raise SystemExit("--train-size must be > 0")
    if args.add_batch_size <= 0:
        raise SystemExit("--add-batch-size must be > 0")
    if args.search_batch_size <= 0:
        raise SystemExit("--search-batch-size must be > 0")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    paths = resolve_paths(args)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_paths(output_dir)
    ensure_outputs(out, overwrite=args.overwrite, write_candidates=args.write_candidates, save_index=args.save_index)

    vectors = np.load(paths["vectors"], mmap_mode="r")
    if args.limit is not None:
        vectors = vectors[: args.limit]
    if vectors.dtype != np.float32:
        raise SystemExit(f"expected float32 vectors, got {vectors.dtype}")
    if vectors.ndim != 2:
        raise SystemExit(f"expected 2D vectors, got shape={vectors.shape}")
    ids = load_ids(paths["ids"], args.limit)
    if len(ids) != int(vectors.shape[0]):
        raise SystemExit(f"ids count {len(ids)} does not match vector count {vectors.shape[0]}")

    started_at = utc_now()
    started = time.perf_counter()
    print(
        f"[index] vectors={vectors.shape[0]} dimension={vectors.shape[1]} "
        f"devices={args.devices} nlist={args.nlist} nprobe={args.nprobe} shard={args.shard}",
        flush=True,
    )
    gpu_index, index_metrics = build_gpu_index(vectors, args)

    index_uri = None
    if args.save_index:
        print("[index] converting GPU index to CPU and saving", flush=True)
        save_cpu_index(gpu_index, out["index_tmp"])
        out["index_tmp"].replace(out["index"])
        index_uri = out["index"].resolve().as_uri()

    candidate_metrics: dict[str, Any] = {}
    candidate_uri = None
    if args.write_candidates:
        print("[index] searching all vectors and writing candidates", flush=True)
        candidate_metrics = write_candidates(
            gpu_index=gpu_index,
            vectors=vectors,
            ids=ids,
            path=out["candidates_tmp"],
            top_k=int(args.top_k),
            search_batch_size=int(args.search_batch_size),
            progress_interval=float(args.progress_interval),
        )
        out["candidates_tmp"].replace(out["candidates"])
        candidate_uri = out["candidates"].resolve().as_uri()

    created_at = utc_now()
    metadata = {
        "stage": "vector_index",
        "backend": "faiss_gpu_ivf_flat",
        "metric": "cosine",
        "index_type": "ivf_flat",
        "vector_count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "top_k": int(args.top_k) if args.write_candidates else None,
        "index_file_uri": index_uri,
        "candidate_uri": candidate_uri,
        "metadata_uri": out["metadata"].resolve().as_uri(),
        "started_at": started_at,
        "created_at": created_at,
        "elapsed_seconds": time.perf_counter() - started,
        **index_metrics,
        **candidate_metrics,
    }
    summary_path = stage_output_paths(paths["run_dir"])["summary"]
    if (paths["run_dir"] / "state.json").exists():
        metadata["summary_uri"] = summary_path.resolve().as_uri()
    out["metadata_tmp"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out["metadata_tmp"].replace(out["metadata"])

    artifacts = [artifact("index_metadata", "vector_index", out["metadata"], 1, created_at)]
    if args.save_index:
        artifacts.append(artifact("faiss_index", "vector_index", out["index"], int(vectors.shape[0]), created_at))
    if args.write_candidates:
        artifacts.append(artifact("hnsw_candidates", "vector_index", out["candidates"], int(vectors.shape[0]), created_at))
    artifacts.extend(write_stage_outputs(paths["run_dir"], metadata, out["candidates"] if args.write_candidates else None))
    update_run_state(paths["run_dir"], metadata, artifacts)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
