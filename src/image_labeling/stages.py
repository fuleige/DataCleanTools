from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import random
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError
from sklearn.cluster import KMeans, MiniBatchKMeans

from .config import QualityConfig
from .jsonio import read_json, read_jsonl, write_json, write_jsonl
from .logging_utils import duration_ms, utc_now
from .providers import build_embedding_provider, build_prelabel_provider
from .storage import RunContext, record_data_artifact, sha256_file
from .uri import ensure_local_uri, is_local_uri, uri_to_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ACTION_PRIORITY = {"keep": 0, "needs_review": 1, "quarantine": 2, "drop": 3}
PROGRESS_INTERVAL = 1000
QUALITY_CHECK_IMPL_VERSION = "quality_check_sharded_v1"
_QUALITY_WORKER_CONFIG: QualityConfig | None = None


def run_ingest(ctx: RunContext) -> None:
    rows, input_errors = _load_input_rows(ctx)
    seen_image_ids: set[str] = set()
    seen_uris: set[str] = set()
    duplicate_errors: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    progress_key = "ingest_manifest"
    last_reported = 0
    output_count = 0

    ctx.report_progress("task_started", key=progress_key, description="ingest manifest rows", total=len(rows))
    ctx.update_stage_progress("ingest", processed_items=0, total_items=len(rows), progress_phase="manifest_build")

    manifest_path = ctx.path("data", "manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            uri = row["uri"]
            image_id = row.get("image_id") or stable_image_id(ctx.config.project.dataset_id, uri)
            if image_id in seen_image_ids or uri in seen_uris:
                duplicate_errors.append(
                    {
                        "image_id": image_id,
                        "uri": uri,
                        "error_code": "duplicate_input",
                        "message": "duplicate image_id or uri ignored",
                    }
                )
                continue
            seen_image_ids.add(image_id)
            seen_uris.add(uri)
            local_path = str(uri_to_path(uri)) if is_local_uri(uri) else None
            file_path = Path(local_path) if local_path else None
            manifest_row = {
                "image_id": image_id,
                "uri": uri,
                "local_path": local_path,
                "metadata": row.get("metadata") or {},
                "file_size": file_path.stat().st_size if file_path and file_path.exists() else None,
                "content_hash": (
                    sha256_file(file_path)
                    if ctx.config.input.compute_content_hash and file_path and file_path.exists()
                    else None
                ),
            }
            f.write(json.dumps(manifest_row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            output_count += 1
            if len(sample_rows) < ctx.config.runtime.sample_limit:
                sample_rows.append(manifest_row)
            if idx % PROGRESS_INTERVAL == 0 or idx == len(rows):
                ctx.report_progress("task_advance", key=progress_key, advance=idx - last_reported)
                ctx.update_stage_progress("ingest", processed_items=idx, total_items=len(rows))
                last_reported = idx

    ctx.report_progress(
        "task_completed",
        key=progress_key,
        description=f"ingest manifest rows ({output_count} unique)",
        completed=len(rows),
        total=len(rows),
    )
    ctx.state["input_manifest_uri"] = ctx.uri(manifest_path)
    record_data_artifact(ctx, "manifest", "ingest", manifest_path, output_count)

    errors = [*input_errors, *duplicate_errors]
    summary = {
        "stage": "ingest",
        "input_count": len(rows),
        "output_count": output_count,
        "duplicate_count": len(duplicate_errors),
        "invalid_uri_count": len(input_errors),
        "manifest_uri": ctx.uri(manifest_path),
        "source_type": ctx.config.input.type,
    }
    ctx.write_stage_outputs("ingest", summary, sample=sample_rows, errors=errors)


def run_quality_check(ctx: RunContext) -> None:
    manifest_path = ctx.path("data", "manifest.jsonl")
    ingest_summary = read_json(ctx.summary_path("ingest"), default={}) or {}
    total_rows = int(ingest_summary.get("output_count") or _count_jsonl_rows(manifest_path))
    workers = max(1, ctx.config.runtime.quality_check_workers)
    executor_mode = ctx.config.runtime.quality_check_executor
    effective_executor = "sequential" if workers == 1 else executor_mode
    shard_size = max(1, ctx.config.runtime.quality_check_shard_size)
    resume_shards = ctx.config.runtime.quality_check_resume_shards
    quality_path = ctx.path("data", "quality_results.jsonl")
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = ctx.path("data", "quality_check_shards")
    tmp_dir = ctx.path("data", "quality_check_shards_tmp")
    checkpoint_path = ctx.path("checkpoints", "quality_check_shards.jsonl")
    shard_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch(exist_ok=True)

    specs = _quality_shard_specs(total_rows, shard_size)
    checkpoints = _load_quality_checkpoints(checkpoint_path)
    config_hash = _quality_config_hash(ctx)
    force_recompute = bool(ctx.state.get("stages", {}).get("quality_check", {}).get("force"))
    progress_key = "quality_check_images"
    last_reported = 0
    reused_shards = 0
    completed_shards = 0

    ctx.report_progress("task_started", key=progress_key, description="quality check images", total=total_rows)
    ctx.update_stage_progress(
        "quality_check",
        processed_items=0,
        total_items=total_rows,
        executor=executor_mode,
        effective_executor=effective_executor,
        worker_count=workers,
        shard_size=shard_size,
        total_shards=len(specs),
        resume_shards=resume_shards,
    )

    def report_processed(idx: int, *, current_shard: int | None = None) -> None:
        nonlocal last_reported
        if idx <= last_reported:
            if current_shard is not None:
                ctx.update_stage_progress(
                    "quality_check",
                    processed_items=last_reported,
                    total_items=total_rows,
                    executor=executor_mode,
                    effective_executor=effective_executor,
                    current_shard=current_shard,
                    completed_shards=completed_shards,
                    reused_shards=reused_shards,
                )
            return
        if idx % PROGRESS_INTERVAL == 0 or idx == total_rows or current_shard is not None:
            ctx.report_progress("task_advance", key=progress_key, advance=idx - last_reported)
            ctx.update_stage_progress(
                "quality_check",
                processed_items=idx,
                total_items=total_rows,
                executor=executor_mode,
                effective_executor=effective_executor,
                current_shard=current_shard,
                completed_shards=completed_shards,
                reused_shards=reused_shards,
            )
            last_reported = idx

    executor: Executor | None = None
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_f:
            for spec in specs:
                shard_path = _quality_shard_path(shard_dir, spec["shard_id"])
                checkpoint = checkpoints.get(spec["shard_id"])
                can_reuse = (
                    resume_shards
                    and not force_recompute
                    and _quality_shard_is_reusable(checkpoint, shard_path, spec["input_count"], config_hash)
                )
                rows = _read_manifest_shard(manifest_f, spec["start_index"], spec["input_count"], parse=not can_reuse)
                if can_reuse:
                    reused_shards += 1
                    completed_shards += 1
                    report_processed(spec["end_index"], current_shard=spec["shard_id"])
                    continue

                attempt = int((checkpoint or {}).get("attempt") or 0) + 1
                tmp_path = tmp_dir / f"shard_{spec['shard_id']:06d}.attempt_{attempt}.tmp.jsonl"
                started_at = utc_now()
                if executor is None:
                    executor = _create_quality_executor(ctx.config.quality, workers, executor_mode)
                _append_quality_checkpoint(
                    checkpoint_path,
                    _quality_checkpoint_record(
                        ctx,
                        spec,
                        status="running",
                        attempt=attempt,
                        config_hash=config_hash,
                        output_path=shard_path,
                        started_at=started_at,
                    ),
                )
                try:
                    output_count = _write_quality_shard(
                        ctx,
                        rows,
                        tmp_path,
                        workers,
                        executor_mode,
                        executor,
                        report_processed,
                    )
                    if output_count != spec["input_count"]:
                        raise ValueError(
                            f"quality shard {spec['shard_id']} wrote {output_count} rows, expected {spec['input_count']}"
                        )
                    tmp_path.replace(shard_path)
                    if _count_jsonl_rows(shard_path) != spec["input_count"]:
                        raise ValueError(f"quality shard {spec['shard_id']} failed row-count validation")
                    completed_shards += 1
                    _append_quality_checkpoint(
                        checkpoint_path,
                        _quality_checkpoint_record(
                            ctx,
                            spec,
                            status="succeeded",
                            attempt=attempt,
                            config_hash=config_hash,
                            output_path=shard_path,
                            started_at=started_at,
                            finished_at=utc_now(),
                            output_count=output_count,
                        ),
                    )
                    report_processed(spec["end_index"], current_shard=spec["shard_id"])
                except BaseException as exc:
                    status = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    _append_quality_checkpoint(
                        checkpoint_path,
                        _quality_checkpoint_record(
                            ctx,
                            spec,
                            status=status,
                            attempt=attempt,
                            config_hash=config_hash,
                            output_path=shard_path,
                            started_at=started_at,
                            finished_at=utc_now(),
                            error={"error_code": exc.__class__.__name__, "message": str(exc)},
                        ),
                    )
                    raise
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    merge_summary = _merge_quality_shards(ctx, specs, shard_dir, quality_path)

    ctx.report_progress(
        "task_completed",
        key=progress_key,
        description="quality check images",
        completed=merge_summary["output_count"],
        total=total_rows,
    )
    record_data_artifact(ctx, "quality_results", "quality_check", quality_path, merge_summary["output_count"])
    record_data_artifact(ctx, "quality_checkpoints", "quality_check", checkpoint_path, len(specs))

    summary = {
        "stage": "quality_check",
        "input_count": total_rows,
        "output_count": merge_summary["output_count"],
        "failed_count": merge_summary["error_count"],
        "executor": executor_mode,
        "effective_executor": effective_executor,
        "worker_count": workers,
        "shard_size": shard_size,
        "shard_count": len(specs),
        "completed_shards": completed_shards,
        "reused_shards": reused_shards,
        "implementation_version": QUALITY_CHECK_IMPL_VERSION,
        "checkpoint_uri": ctx.uri(checkpoint_path),
        "shard_dir_uri": ctx.uri(shard_dir),
        "status_counts": dict(merge_summary["status_counts"]),
        "reason_counts": dict(merge_summary["reason_counts"].most_common()),
        "config": ctx.config.quality.model_dump(mode="json"),
    }
    ctx.write_stage_outputs(
        "quality_check",
        summary,
        sample=merge_summary["sample_rows"],
        errors_prewritten=True,
        errors_count=merge_summary["error_count"],
    )


def _quality_shard_specs(total_rows: int, shard_size: int) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    for shard_id, start_index in enumerate(range(1, total_rows + 1, shard_size)):
        end_index = min(start_index + shard_size - 1, total_rows)
        specs.append(
            {
                "shard_id": shard_id,
                "start_index": start_index,
                "end_index": end_index,
                "input_count": end_index - start_index + 1,
            }
        )
    return specs


def _quality_shard_path(shard_dir: Path, shard_id: int) -> Path:
    return shard_dir / f"shard_{shard_id:06d}.jsonl"


def _quality_config_hash(ctx: RunContext) -> str:
    payload = {
        "quality": ctx.config.quality.model_dump(mode="json"),
        "runtime": {
            "quality_check_shard_size": ctx.config.runtime.quality_check_shard_size,
        },
        "implementation_version": QUALITY_CHECK_IMPL_VERSION,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_quality_checkpoints(path: Path) -> dict[int, dict[str, Any]]:
    checkpoints: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return checkpoints
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                checkpoints[int(row["shard_id"])] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return checkpoints


def _append_quality_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _quality_checkpoint_record(
    ctx: RunContext,
    spec: dict[str, int],
    *,
    status: str,
    attempt: int,
    config_hash: str,
    output_path: Path,
    started_at: str,
    finished_at: str | None = None,
    output_count: int | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": "quality_check",
        "shard_id": spec["shard_id"],
        "start_index": spec["start_index"],
        "end_index": spec["end_index"],
        "input_count": spec["input_count"],
        "output_count": output_count,
        "status": status,
        "attempt": attempt,
        "config_hash": config_hash,
        "implementation_version": QUALITY_CHECK_IMPL_VERSION,
        "output_uri": ctx.uri(output_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms(started_at, finished_at) if finished_at else None,
        "error_json": error,
    }


def _quality_shard_is_reusable(
    checkpoint: dict[str, Any] | None,
    shard_path: Path,
    input_count: int,
    config_hash: str,
) -> bool:
    if not checkpoint or checkpoint.get("status") != "succeeded":
        return False
    if checkpoint.get("implementation_version") != QUALITY_CHECK_IMPL_VERSION:
        return False
    if checkpoint.get("config_hash") != config_hash:
        return False
    if int(checkpoint.get("output_count") or -1) != input_count:
        return False
    return shard_path.exists() and _count_jsonl_rows(shard_path) == input_count


def _read_manifest_shard(
    manifest_f: TextIO,
    start_index: int,
    input_count: int,
    *,
    parse: bool,
) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    seen = 0
    while seen < input_count:
        line = manifest_f.readline()
        if line == "":
            raise ValueError(f"manifest ended before shard starting at row {start_index} was complete")
        line = line.strip()
        if not line:
            continue
        idx = start_index + seen
        if parse:
            rows.append((idx, json.loads(line)))
        seen += 1
    return rows


def _write_quality_shard(
    ctx: RunContext,
    rows: list[tuple[int, dict[str, Any]]],
    tmp_path: Path,
    workers: int,
    executor_mode: str,
    executor: Executor | None,
    report_processed,
) -> int:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        tmp_path.unlink()
    output_count = 0
    results = _iter_quality_results(ctx, rows, workers, executor_mode, executor)
    with tmp_path.open("w", encoding="utf-8") as out_f:
        for idx, _row, result in results:
            out_f.write(json.dumps(result, ensure_ascii=False, sort_keys=True))
            out_f.write("\n")
            output_count += 1
            report_processed(idx)
    return output_count


def _merge_quality_shards(
    ctx: RunContext,
    specs: list[dict[str, int]],
    shard_dir: Path,
    quality_path: Path,
) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    output_count = 0
    error_count = 0
    tmp_path = quality_path.with_name(f"{quality_path.name}.tmp")
    errors_path = ctx.errors_path("quality_check")
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_tmp_path = errors_path.with_name(f"{errors_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as out_f, errors_tmp_path.open("w", encoding="utf-8") as err_f:
        for spec in specs:
            shard_path = _quality_shard_path(shard_dir, spec["shard_id"])
            shard_count = 0
            with shard_path.open("r", encoding="utf-8") as shard_f:
                for line in shard_f:
                    line = line.strip()
                    if not line:
                        continue
                    result = json.loads(line)
                    out_f.write(json.dumps(result, ensure_ascii=False, sort_keys=True))
                    out_f.write("\n")
                    output_count += 1
                    shard_count += 1
                    if len(sample_rows) < ctx.config.runtime.sample_limit:
                        sample_rows.append(result)
                    status_counts[result["quality_status"]] += 1
                    reason_counts.update(result["quality_reasons"])
                    if result["quality_status"] == "drop":
                        error = {
                            "image_id": result["image_id"],
                            "uri": result["uri"],
                            "error_code": ",".join(result["quality_reasons"]) or "quality_drop",
                            "message": result.get("message", "image dropped by quality rules"),
                        }
                        err_f.write(json.dumps(error, ensure_ascii=False, sort_keys=True))
                        err_f.write("\n")
                        error_count += 1
            if shard_count != spec["input_count"]:
                raise ValueError(
                    f"quality shard {spec['shard_id']} has {shard_count} rows, expected {spec['input_count']}"
                )
    tmp_path.replace(quality_path)
    errors_tmp_path.replace(errors_path)
    return {
        "output_count": output_count,
        "sample_rows": sample_rows,
        "error_count": error_count,
        "status_counts": status_counts,
        "reason_counts": reason_counts,
    }


def _iter_manifest_rows(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as in_f:
        row_idx = 0
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row_idx += 1
            yield row_idx, json.loads(line)


def _iter_jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _ensure_in_memory_row_limit(ctx: RunContext, stage: str, path: Path) -> int:
    row_count = _count_jsonl_rows(path)
    if row_count > ctx.config.runtime.max_in_memory_rows:
        raise ValueError(
            f"{stage} currently loads {row_count} rows from {path.name} into memory; "
            f"runtime.max_in_memory_rows is {ctx.config.runtime.max_in_memory_rows}. "
            "Run only earlier stages or implement a sharded/streaming version for this stage."
        )
    return row_count


def _iter_quality_results_sequential(ctx: RunContext, rows: list[tuple[int, dict[str, Any]]]):
    for idx, row in rows:
        yield idx, row, _quality_for_row(ctx, row)


def _iter_quality_results(
    ctx: RunContext,
    rows: list[tuple[int, dict[str, Any]]],
    workers: int,
    executor_mode: str,
    executor: Executor | None,
):
    if workers <= 1:
        yield from _iter_quality_results_sequential(ctx, rows)
        return
    if executor is None:
        raise RuntimeError("quality_check executor was not initialized")
    if executor_mode == "process":
        yield from _iter_quality_results_process_pool(rows, workers, executor)
        return
    yield from _iter_quality_results_thread_pool(ctx, rows, workers, executor)


def _create_quality_executor(quality_config: QualityConfig, workers: int, executor_mode: str) -> Executor | None:
    if workers <= 1:
        return None
    if executor_mode == "process":
        return ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_quality_worker,
            initargs=(quality_config.model_dump(mode="json"),),
            mp_context=_quality_process_context(),
        )
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="quality-check")


def _quality_process_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("forkserver")
    except ValueError:
        return multiprocessing.get_context("spawn")


def _init_quality_worker(quality_config: dict[str, Any]) -> None:
    global _QUALITY_WORKER_CONFIG
    _QUALITY_WORKER_CONFIG = QualityConfig.model_validate(quality_config)


def _quality_for_row_worker(row: dict[str, Any]) -> dict[str, Any]:
    if _QUALITY_WORKER_CONFIG is None:
        raise RuntimeError("quality worker config is not initialized")
    return _quality_for_row_with_config(_QUALITY_WORKER_CONFIG, row)


def _quality_process_chunksize(row_count: int, workers: int) -> int:
    return max(1, min(128, math.ceil(row_count / max(workers * 16, 1))))


def _iter_quality_results_process_pool(rows: list[tuple[int, dict[str, Any]]], workers: int, executor: Executor):
    row_values = (row for _, row in rows)
    chunksize = _quality_process_chunksize(len(rows), workers)
    for (idx, row), result in zip(rows, executor.map(_quality_for_row_worker, row_values, chunksize=chunksize)):
        yield idx, row, result


def _iter_quality_results_thread_pool(
    ctx: RunContext,
    rows: list[tuple[int, dict[str, Any]]],
    workers: int,
    executor: Executor,
):
    if not rows:
        return
    max_in_flight = max(workers * 4, 1)
    row_iter = iter(rows)
    pending: dict[Future, tuple[int, dict[str, Any]]] = {}
    buffered: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    next_output_idx = rows[0][0]
    exhausted = False

    def fill_pending() -> None:
        nonlocal exhausted
        while not exhausted and len(pending) + len(buffered) < max_in_flight:
            try:
                idx, row = next(row_iter)
            except StopIteration:
                exhausted = True
                break
            pending[executor.submit(_quality_for_row, ctx, row)] = (idx, row)

    fill_pending()
    while pending or buffered:
        while next_output_idx in buffered:
            row, result = buffered.pop(next_output_idx)
            yield next_output_idx, row, result
            next_output_idx += 1

        fill_pending()
        if not pending:
            continue

        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            idx, row = pending.pop(future)
            buffered[idx] = (row, future.result())


def run_thumbnail(ctx: RunContext) -> None:
    quality_path = ctx.path("data", "quality_results.jsonl")
    sample_rows: list[dict[str, Any]] = []
    thumb_dir = ctx.path("thumbnails")
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_index = ctx.path("data", "thumbnails.jsonl")
    thumb_index.parent.mkdir(parents=True, exist_ok=True)
    thumb_tmp = thumb_index.with_name(f"{thumb_index.name}.tmp")
    errors_path = ctx.errors_path("thumbnail")
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_tmp = errors_path.with_name(f"{errors_path.name}.tmp")
    input_count = 0
    success_count = 0
    failed_count = 0

    with thumb_tmp.open("w", encoding="utf-8") as thumb_f, errors_tmp.open("w", encoding="utf-8") as err_f:
        for row in _iter_jsonl_rows(quality_path):
            input_count += 1
            if not row.get("local_path"):
                error = _stage_error(row, "non_local_image", "thumbnail currently requires local image paths")
                err_f.write(json.dumps(error, ensure_ascii=False, sort_keys=True))
                err_f.write("\n")
                failed_count += 1
                continue
            if row["quality_status"] == "drop":
                continue
            try:
                output_path = thumb_dir / f"{row['image_id']}.jpg"
                with Image.open(row["local_path"]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((ctx.config.thumbnail.size, ctx.config.thumbnail.size))
                    image.save(output_path, format="JPEG", quality=ctx.config.thumbnail.quality)
                thumb_row = {
                    "image_id": row["image_id"],
                    "thumbnail_uri": ctx.uri(output_path),
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "quality_status": row["quality_status"],
                }
                thumb_f.write(json.dumps(thumb_row, ensure_ascii=False, sort_keys=True))
                thumb_f.write("\n")
                success_count += 1
                if len(sample_rows) < ctx.config.runtime.sample_limit:
                    sample_rows.append(thumb_row)
            except Exception as exc:  # noqa: BLE001
                error = _stage_error(row, exc.__class__.__name__, str(exc))
                err_f.write(json.dumps(error, ensure_ascii=False, sort_keys=True))
                err_f.write("\n")
                failed_count += 1

    thumb_tmp.replace(thumb_index)
    errors_tmp.replace(errors_path)
    record_data_artifact(ctx, "thumbnails", "thumbnail", thumb_index, success_count)
    summary = {
        "stage": "thumbnail",
        "input_count": input_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "thumbnail_dir": ctx.uri(thumb_dir),
    }
    ctx.write_stage_outputs(
        "thumbnail",
        summary,
        sample=sample_rows,
        errors_prewritten=True,
        errors_count=failed_count,
    )


def run_embedding(ctx: RunContext) -> None:
    quality_path = ctx.path("data", "quality_results.jsonl")
    include_statuses = set(ctx.config.embedding.include_quality_statuses)
    batch_size = max(1, ctx.config.embedding.batch_size)
    provider = build_embedding_provider(ctx.config.embedding)

    embedding_dir = ctx.path("embeddings")
    embedding_dir.mkdir(parents=True, exist_ok=True)
    vector_path = embedding_dir / "embeddings.npy"
    vector_tmp_path = embedding_dir / "embeddings.tmp.npy"
    id_path = embedding_dir / "ids.json"
    id_tmp_path = embedding_dir / "ids.tmp.json"
    refs_path = embedding_dir / "embedding_refs.jsonl"
    refs_tmp_path = embedding_dir / "embedding_refs.tmp.jsonl"
    sample_rows: list[dict[str, Any]] = []

    scan_key = "embedding_candidate_scan"
    total_rows = 0
    candidate_count = 0
    skipped_non_local = 0
    last_reported = 0
    ctx.report_progress("task_started", key=scan_key, description="scan embedding candidates", total=None)
    ctx.update_stage_progress("embedding", processed_items=0, progress_phase="candidate_scan")
    for row in _iter_jsonl_rows(quality_path):
        total_rows += 1
        if total_rows % PROGRESS_INTERVAL == 0:
            ctx.report_progress("task_advance", key=scan_key, advance=total_rows - last_reported)
            ctx.update_stage_progress(
                "embedding",
                processed_items=total_rows,
                matched_items=candidate_count,
                progress_phase="candidate_scan",
            )
            last_reported = total_rows
        if row["quality_status"] not in include_statuses:
            continue
        if not row.get("local_path"):
            skipped_non_local += 1
            continue
        candidate_count += 1
    if total_rows > last_reported:
        ctx.report_progress("task_advance", key=scan_key, advance=total_rows - last_reported)
    ctx.report_progress(
        "task_completed",
        key=scan_key,
        description=f"scan embedding candidates ({candidate_count} images)",
        completed=total_rows,
    )

    encode_key = "embedding_images"
    ctx.report_progress("task_started", key=encode_key, description="embed images", total=candidate_count)
    ctx.update_stage_progress(
        "embedding",
        processed_items=0,
        total_items=candidate_count,
        matched_items=candidate_count,
        progress_phase="embed_images",
    )

    batch_rows: list[dict[str, Any]] = []
    batch_paths: list[Path] = []
    vector_memmap: np.memmap | None = None
    vector_dimension = 0
    output_index = 0
    shard_count = 0
    last_encoded_reported = 0

    def flush_batch(ids_f: TextIO, refs_f: TextIO, *, final: bool = False) -> None:
        nonlocal batch_rows, batch_paths, vector_memmap, vector_dimension, output_index, shard_count, last_encoded_reported
        if not batch_rows:
            return
        vectors = provider.encode(batch_paths)
        if vectors.ndim != 2:
            raise RuntimeError(f"embedding provider returned non-2D vectors with shape={vectors.shape}")
        if int(vectors.shape[0]) != len(batch_rows):
            raise RuntimeError(
                f"embedding provider returned {vectors.shape[0]} vectors for {len(batch_rows)} input images"
            )
        if vector_memmap is None:
            vector_dimension = int(vectors.shape[1])
            vector_memmap = np.lib.format.open_memmap(
                vector_tmp_path,
                mode="w+",
                dtype=np.float32,
                shape=(candidate_count, vector_dimension),
            )
        end_index = output_index + int(vectors.shape[0])
        vector_memmap[output_index:end_index] = vectors.astype(np.float32)
        for row_idx, row in enumerate(batch_rows, start=output_index):
            if row_idx:
                ids_f.write(",")
            ids_f.write(json.dumps(row["image_id"], ensure_ascii=False))
            ref_row = {
                "image_id": row["image_id"],
                "uri": row["uri"],
                "local_path": row["local_path"],
                "quality_status": row["quality_status"],
                "row_index": row_idx,
                "embedding_uri": ctx.uri(vector_path),
            }
            refs_f.write(json.dumps(ref_row, ensure_ascii=False, sort_keys=True))
            refs_f.write("\n")
            if len(sample_rows) < ctx.config.runtime.sample_limit:
                sample_rows.append(ref_row)
        output_index = end_index
        shard_count += 1
        if output_index - last_encoded_reported >= PROGRESS_INTERVAL or final:
            ctx.report_progress("task_advance", key=encode_key, advance=output_index - last_encoded_reported)
            ctx.update_stage_progress(
                "embedding",
                processed_items=output_index,
                total_items=candidate_count,
                progress_phase="embed_images",
            )
            last_encoded_reported = output_index
        batch_rows = []
        batch_paths = []

    if candidate_count == 0:
        np.save(vector_tmp_path, np.zeros((0, 0), dtype=np.float32))
        id_tmp_path.write_text("[]\n", encoding="utf-8")
        refs_tmp_path.write_text("", encoding="utf-8")
    else:
        with id_tmp_path.open("w", encoding="utf-8") as ids_f, refs_tmp_path.open("w", encoding="utf-8") as refs_f:
            ids_f.write("[")
            for row in _iter_jsonl_rows(quality_path):
                if row["quality_status"] not in include_statuses or not row.get("local_path"):
                    continue
                batch_rows.append(row)
                batch_paths.append(Path(row["local_path"]))
                if len(batch_rows) >= batch_size:
                    flush_batch(ids_f, refs_f)
            flush_batch(ids_f, refs_f, final=True)
            ids_f.write("]\n")
        if vector_memmap is not None:
            vector_memmap.flush()
            del vector_memmap

    vector_tmp_path.replace(vector_path)
    id_tmp_path.replace(id_path)
    refs_tmp_path.replace(refs_path)

    ctx.report_progress(
        "task_completed",
        key=encode_key,
        description=f"embed images ({output_index} vectors)",
        completed=output_index,
        total=candidate_count,
    )

    record_data_artifact(ctx, "embeddings", "embedding", vector_path, output_index)
    record_data_artifact(ctx, "embedding_ids", "embedding", id_path, output_index)
    record_data_artifact(ctx, "embedding_refs", "embedding", refs_path, output_index)

    summary = {
        "stage": "embedding",
        "provider": ctx.config.embedding.provider,
        "model_name": ctx.config.embedding.model_name,
        "model_version": ctx.config.embedding.model_version,
        "dimension": vector_dimension,
        "input_count": total_rows,
        "candidate_count": candidate_count,
        "success_count": output_index,
        "failed_count": 0,
        "skipped_non_local_count": skipped_non_local,
        "batch_size": batch_size,
        "shard_count": shard_count,
        "embedding_uri": ctx.uri(vector_path),
    }
    ctx.write_stage_outputs("embedding", summary, sample=sample_rows, errors=[])


def run_vector_index(ctx: RunContext) -> None:
    vectors, ids = _load_vectors(ctx)
    index_dir = ctx.path("indexes")
    index_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = index_dir / "hnsw_candidates.jsonl"
    metadata_path = index_dir / "index_metadata.json"

    if len(ids) == 0:
        write_jsonl(candidates_path, [])
        metadata = {"backend": "none", "vector_count": 0, "dimension": 0}
        write_json(metadata_path, metadata)
        index_file_uri = None
    else:
        top_k = min(ctx.config.vector_index.top_k, len(ids))
        try:
            import faiss  # type: ignore

            index = faiss.IndexHNSWFlat(
                int(vectors.shape[1]),
                int(ctx.config.vector_index.hnsw.M),
                faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = int(ctx.config.vector_index.hnsw.ef_construction)
            index.hnsw.efSearch = int(ctx.config.vector_index.hnsw.ef_search)
            index.add(vectors.astype(np.float32))
            distances, indices = index.search(vectors.astype(np.float32), top_k)
            index_path = index_dir / "faiss.index"
            if ctx.config.vector_index.save_index:
                faiss.write_index(index, str(index_path))
                record_data_artifact(ctx, "faiss_index", "vector_index", index_path, len(ids))
                index_file_uri = ctx.uri(index_path)
            else:
                index_file_uri = None
            backend = "faiss_hnsw_flat"
        except Exception as exc:  # noqa: BLE001
            if not ctx.config.vector_index.allow_exact_fallback:
                raise
            similarities = vectors @ vectors.T
            indices = np.argsort(-similarities, axis=1)[:, :top_k]
            distances = np.take_along_axis(similarities, indices, axis=1)
            backend = "exact_numpy_fallback"
            index_file_uri = None
            ctx.logger.log(
                "vector_index",
                "faiss_fallback",
                f"FAISS unavailable or failed, using exact numpy fallback: {exc}",
                level="WARNING",
                error_code=exc.__class__.__name__,
            )

        candidate_rows = []
        for row_idx, image_id in enumerate(ids):
            candidate_rows.append(
                {
                    "image_id": image_id,
                    "candidates": [
                        {"image_id": ids[int(idx)], "score": float(score)}
                        for idx, score in zip(indices[row_idx], distances[row_idx], strict=False)
                        if int(idx) >= 0
                    ],
                }
            )
        write_jsonl(candidates_path, candidate_rows)
        metadata = {
            "backend": backend,
            "metric": ctx.config.vector_index.metric,
            "index_type": ctx.config.vector_index.index_type,
            "vector_count": len(ids),
            "dimension": int(vectors.shape[1]),
            "top_k": top_k,
            "hnsw": ctx.config.vector_index.hnsw.model_dump(mode="json"),
            "index_file_uri": index_file_uri,
        }
        write_json(metadata_path, metadata)

    record_data_artifact(ctx, "hnsw_candidates", "vector_index", candidates_path, len(ids))
    record_data_artifact(ctx, "index_metadata", "vector_index", metadata_path, 1)
    summary = {"stage": "vector_index", **read_json(metadata_path)}
    ctx.write_stage_outputs("vector_index", summary, sample=read_jsonl(candidates_path)[: ctx.config.runtime.sample_limit], errors=[])


def run_similarity(ctx: RunContext) -> None:
    vectors, ids = _load_vectors(ctx)
    candidates = {row["image_id"]: row["candidates"] for row in read_jsonl(ctx.path("indexes", "hnsw_candidates.jsonl"))}
    id_to_index = {image_id: idx for idx, image_id in enumerate(ids)}
    final_top_k = min(ctx.config.vector_index.rerank.final_top_k, max(0, len(ids) - 1))
    topk_rows: list[dict[str, Any]] = []
    duplicate_edges: list[tuple[str, str, float]] = []

    for image_id in ids:
        candidate_ids = [item["image_id"] for item in candidates.get(image_id, []) if item["image_id"] != image_id]
        candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id in id_to_index]
        if ctx.config.vector_index.rerank.enabled:
            base_vec = vectors[id_to_index[image_id]]
            scored = [(candidate_id, float(base_vec @ vectors[id_to_index[candidate_id]])) for candidate_id in candidate_ids]
            scored.sort(key=lambda item: item[1], reverse=True)
        else:
            scored = [(item["image_id"], float(item["score"])) for item in candidates.get(image_id, []) if item["image_id"] != image_id]
        neighbors = [{"image_id": item_id, "score": score} for item_id, score in scored[:final_top_k]]
        for neighbor in neighbors:
            if (
                ctx.config.clustering.duplicate_detection.enabled
                and neighbor["score"] >= ctx.config.clustering.duplicate_detection.embedding_threshold
            ):
                duplicate_edges.append((image_id, neighbor["image_id"], neighbor["score"]))
        topk_rows.append({"image_id": image_id, "neighbors": neighbors})

    groups = _duplicate_groups(ids, duplicate_edges)
    similarity_dir = ctx.path("similarity")
    topk_path = similarity_dir / "topk.jsonl"
    groups_path = similarity_dir / "duplicate_groups.jsonl"
    write_jsonl(topk_path, topk_rows)
    write_jsonl(groups_path, groups)
    record_data_artifact(ctx, "similarity_topk", "similarity", topk_path, len(topk_rows))
    record_data_artifact(ctx, "duplicate_groups", "similarity", groups_path, len(groups))

    summary = {
        "stage": "similarity",
        "image_count": len(ids),
        "final_top_k": final_top_k,
        "duplicate_group_count": len(groups),
        "duplicate_edge_count": len(duplicate_edges),
        "embedding_threshold": ctx.config.clustering.duplicate_detection.embedding_threshold,
    }
    ctx.write_stage_outputs("similarity", summary, sample=groups[: ctx.config.runtime.sample_limit], errors=[])


def run_clustering(ctx: RunContext) -> None:
    vectors, ids = _load_vectors(ctx)
    cluster_dir = ctx.path("clusters")
    cluster_rows_path = cluster_dir / "clusters.jsonl"
    cluster_summary_path = cluster_dir / "cluster_summary.json"

    if not ctx.config.clustering.enabled or len(ids) == 0:
        write_jsonl(cluster_rows_path, [])
        write_json(cluster_summary_path, {"enabled": ctx.config.clustering.enabled, "cluster_count": 0})
        record_data_artifact(ctx, "clusters", "clustering", cluster_rows_path, 0)
        summary = {"stage": "clustering", "enabled": ctx.config.clustering.enabled, "cluster_count": 0, "image_count": len(ids)}
        ctx.write_stage_outputs("clustering", summary, sample=[], errors=[])
        return

    cluster_count, requested_count = _effective_cluster_count(ctx, len(ids))
    if cluster_count <= 1:
        labels = np.zeros(len(ids), dtype=int)
        centers = np.mean(vectors, axis=0, keepdims=True)
    elif ctx.config.clustering.algorithm == "kmeans":
        model = KMeans(
            n_clusters=cluster_count,
            max_iter=ctx.config.clustering.max_iter,
            random_state=ctx.config.clustering.random_state,
            n_init="auto",
        )
        labels = model.fit_predict(vectors)
        centers = model.cluster_centers_
    else:
        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            batch_size=min(ctx.config.clustering.batch_size, max(cluster_count, len(ids))),
            max_iter=ctx.config.clustering.max_iter,
            random_state=ctx.config.clustering.random_state,
            n_init="auto",
        )
        labels = model.fit_predict(vectors)
        centers = model.cluster_centers_

    distances = np.linalg.norm(vectors - centers[labels], axis=1)
    threshold = (
        float(np.percentile(distances, ctx.config.clustering.outlier.percentile_threshold))
        if ctx.config.clustering.outlier.enabled and len(distances)
        else float("inf")
    )
    cluster_to_items: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for image_id, label, distance in zip(ids, labels, distances, strict=False):
        cluster_to_items[int(label)].append((image_id, float(distance)))

    representatives = {
        str(cluster_id): [
            image_id
            for image_id, _ in sorted(items, key=lambda item: item[1])[: ctx.config.clustering.representative.samples_per_cluster]
        ]
        for cluster_id, items in cluster_to_items.items()
    }
    cluster_rows = [
        {
            "image_id": image_id,
            "cluster_id": f"cluster_{int(label):05d}",
            "distance_to_centroid": float(distance),
            "is_outlier": bool(distance >= threshold) if math.isfinite(threshold) else False,
            "is_representative": image_id in representatives.get(str(int(label)), []),
        }
        for image_id, label, distance in zip(ids, labels, distances, strict=False)
    ]
    cluster_sizes = Counter(row["cluster_id"] for row in cluster_rows)
    summary_payload = {
        "algorithm": ctx.config.clustering.algorithm,
        "requested_cluster_count": requested_count,
        "effective_cluster_count": cluster_count,
        "image_count": len(ids),
        "cluster_sizes": dict(cluster_sizes),
        "outlier_count": sum(1 for row in cluster_rows if row["is_outlier"]),
        "representatives": {f"cluster_{int(k):05d}": value for k, value in representatives.items()},
    }
    write_jsonl(cluster_rows_path, cluster_rows)
    write_json(cluster_summary_path, summary_payload)
    record_data_artifact(ctx, "clusters", "clustering", cluster_rows_path, len(cluster_rows))
    record_data_artifact(ctx, "cluster_summary", "clustering", cluster_summary_path, 1)
    ctx.write_stage_outputs(
        "clustering",
        {"stage": "clustering", **summary_payload},
        sample=cluster_rows[: ctx.config.runtime.sample_limit],
        errors=[],
    )


def run_prelabel(ctx: RunContext) -> None:
    refs = read_jsonl(ctx.path("embeddings", "embedding_refs.jsonl"))
    provider = build_prelabel_provider(ctx.config.prelabel)
    outputs = provider.predict(refs, ctx.config.label_schema.labels, ctx.config.prelabel.top_k)
    prelabel_path = ctx.path("prelabels", "prelabels.jsonl")
    write_jsonl(prelabel_path, outputs)
    record_data_artifact(ctx, "prelabels", "prelabel", prelabel_path, len(outputs))

    label_counts = Counter()
    confidence_buckets = Counter()
    errors = []
    for row in outputs:
        if row["status"] != "succeeded":
            errors.append(_stage_error(row, row.get("error_code") or "prelabel_failed", "prelabel provider failed"))
            continue
        if row["candidates"]:
            top = row["candidates"][0]
            label_counts[top["label"]] += 1
            confidence_buckets[_confidence_bucket(top["confidence"])] += 1
    summary = {
        "stage": "prelabel",
        "provider": ctx.config.prelabel.provider,
        "model": ctx.config.prelabel.provider_config.get("model"),
        "input_count": len(refs),
        "success_count": sum(1 for row in outputs if row["status"] == "succeeded"),
        "failed_count": sum(1 for row in outputs if row["status"] != "succeeded"),
        "label_counts": dict(label_counts),
        "confidence_buckets": dict(confidence_buckets),
    }
    ctx.write_stage_outputs("prelabel", summary, sample=outputs[: ctx.config.runtime.sample_limit], errors=errors)


def run_auto_decision(ctx: RunContext) -> None:
    quality_path = ctx.path("data", "quality_results.jsonl")
    _ensure_in_memory_row_limit(ctx, "auto_decision", quality_path)
    quality_rows = {row["image_id"]: row for row in read_jsonl(quality_path)}
    prelabels = {row["image_id"]: row for row in read_jsonl(ctx.path("prelabels", "prelabels.jsonl"))}
    clusters = {row["image_id"]: row for row in read_jsonl(ctx.path("clusters", "clusters.jsonl"))}
    duplicate_groups = read_jsonl(ctx.path("similarity", "duplicate_groups.jsonl"))
    high_risk = {label.id for label in ctx.config.label_schema.labels if label.high_risk}
    cluster_label_stats = _cluster_label_stats(prelabels, clusters)
    conflict_images = _duplicate_conflict_images(prelabels, duplicate_groups)

    decisions: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    auto_accept_count = 0

    for image_id, quality in quality_rows.items():
        prelabel = prelabels.get(image_id)
        candidates = prelabel.get("candidates", []) if prelabel else []
        reasons = _auto_decision_reasons(ctx, quality, candidates, clusters.get(image_id), cluster_label_stats, conflict_images, high_risk)
        top_label = candidates[0]["label"] if candidates else None
        status = "auto_accepted" if not reasons else _initial_non_auto_status(quality)
        if status == "auto_accepted":
            auto_accept_count += 1
        reason_counts.update(reasons or ["auto_accepted"])
        decision = {
            "image_id": image_id,
            "decision_status": status,
            "recommended_label": top_label,
            "final_label": top_label if status == "auto_accepted" else None,
            "source": "auto_decision" if status == "auto_accepted" else "pending_review",
            "reasons": reasons,
        }
        decisions.append(decision)
        final_rows.append(
            {
                "run_id": ctx.run_id,
                "image_id": image_id,
                "uri": quality["uri"],
                "quality_status": quality["quality_status"],
                "quality_reasons": quality["quality_reasons"],
                "cluster": clusters.get(image_id),
                "prelabel_candidates": candidates,
                "recommended_label": top_label,
                "final_label": decision["final_label"],
                "final_status": status,
                "source": decision["source"],
                "config_snapshot_uri": ctx.state["config_snapshot_uri"],
            }
        )

    decision_path = ctx.path("decisions", "auto_decisions.jsonl")
    final_path = ctx.path("decisions", "final_annotations.jsonl")
    write_jsonl(decision_path, decisions)
    write_jsonl(final_path, final_rows)
    record_data_artifact(ctx, "auto_decisions", "auto_decision", decision_path, len(decisions))
    record_data_artifact(ctx, "final_annotations_working", "auto_decision", final_path, len(final_rows))
    summary = {
        "stage": "auto_decision",
        "input_count": len(quality_rows),
        "auto_accept_count": auto_accept_count,
        "pending_or_invalid_count": len(final_rows) - auto_accept_count,
        "reason_counts": dict(reason_counts),
        "config": ctx.config.auto_decision.model_dump(mode="json"),
    }
    ctx.write_stage_outputs("auto_decision", summary, sample=decisions[: ctx.config.runtime.sample_limit], errors=[])


def run_review_queue(ctx: RunContext) -> None:
    final_rows = read_jsonl(ctx.path("decisions", "final_annotations.jsonl"))
    duplicate_groups = read_jsonl(ctx.path("similarity", "duplicate_groups.jsonl"))
    duplicate_images = {image_id for group in duplicate_groups for image_id in group.get("image_ids", [])}
    queue_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    auto_sample_ratio = ctx.config.review.sampling.auto_accept_qa_ratio

    for row in final_rows:
        reasons = _review_reasons(ctx, row, duplicate_images)
        if row["final_status"] == "auto_accepted" and _stable_sample(row["image_id"], auto_sample_ratio):
            reasons.append("auto_accept_sample")
        if not reasons:
            continue
        reason_counts.update(reasons)
        queue_rows.append(
            {
                "review_task_id": stable_image_id(ctx.run_id, row["image_id"]),
                "run_id": ctx.run_id,
                "image_id": row["image_id"],
                "uri": row["uri"],
                "thumbnail_uri": _thumbnail_uri(ctx, row["image_id"]),
                "recommended_label": row["recommended_label"],
                "quality_status": row["quality_status"],
                "final_status": row["final_status"],
                "priority": _review_priority(reasons),
                "reasons": reasons,
            }
        )

    queue_rows.sort(key=lambda row: (row["priority"], row["image_id"]))
    queue_path = ctx.path("review", "review_queue.jsonl")
    write_jsonl(queue_path, queue_rows)
    record_data_artifact(ctx, "review_queue", "review_queue", queue_path, len(queue_rows))
    summary = {
        "stage": "review_queue",
        "queue_count": len(queue_rows),
        "reason_counts": dict(reason_counts),
        "auto_accept_sample_ratio": auto_sample_ratio,
    }
    ctx.write_stage_outputs("review_queue", summary, sample=queue_rows[: ctx.config.runtime.sample_limit], errors=[])


def run_report(ctx: RunContext) -> None:
    summaries = {}
    for summary_path in sorted(ctx.path("summaries").glob("*/stage_summary.json")):
        summaries[summary_path.parent.name] = read_json(summary_path)
    report_dir = ctx.path("reports")
    overview_json = report_dir / "overview.json"
    overview_md = report_dir / "overview.md"
    write_json(overview_json, {"run_id": ctx.run_id, "status": ctx.state["status"], "stages": summaries})
    overview_lines = [f"# Run {ctx.run_id}", ""]
    overview_lines.append(f"- Project: {ctx.config.project.id}")
    overview_lines.append(f"- Dataset: {ctx.config.project.dataset_id}")
    overview_lines.append(f"- Status: {ctx.state['status']}")
    for stage, summary in summaries.items():
        compact = _compact_summary(summary)
        overview_lines.append(f"- {stage}: {compact}")
    overview_md.parent.mkdir(parents=True, exist_ok=True)
    overview_md.write_text("\n".join(overview_lines) + "\n", encoding="utf-8")
    record_data_artifact(ctx, "overview_report_json", "report", overview_json, 1)
    record_data_artifact(ctx, "overview_report_md", "report", overview_md, 1)
    ctx.write_stage_outputs(
        "report",
        {"stage": "report", "summary_count": len(summaries), "overview_md_uri": ctx.uri(overview_md)},
        sample=[{"overview_md_uri": ctx.uri(overview_md)}],
        errors=[],
    )


def run_export(ctx: RunContext) -> None:
    working_rows = _apply_review_decisions(ctx)
    export_dir = ctx.path("exports")
    full_path = export_dir / ctx.config.output.full_file
    valid_path = export_dir / ctx.config.output.valid_only_file
    summary_path = export_dir / "final_annotations.summary.json"
    valid_statuses = {"auto_accepted", "human_accepted", "corrected"}
    valid_rows = [row for row in working_rows if row.get("final_status") in valid_statuses and row.get("final_label")]
    write_jsonl(full_path, working_rows)
    write_jsonl(valid_path, valid_rows)
    status_counts = Counter(row.get("final_status") for row in working_rows)
    label_counts = Counter(row.get("final_label") for row in valid_rows)
    summary = {
        "stage": "export",
        "run_id": ctx.run_id,
        "full_count": len(working_rows),
        "valid_count": len(valid_rows),
        "pending_count": status_counts.get("pending", 0),
        "status_counts": dict(status_counts),
        "label_counts": dict(label_counts),
        "full_annotations_uri": ctx.uri(full_path),
        "valid_annotations_uri": ctx.uri(valid_path),
        "human_confirmed_required": True,
    }
    write_json(summary_path, summary)
    record_data_artifact(ctx, "export_full_annotations", "export", full_path, len(working_rows))
    record_data_artifact(ctx, "export_valid_annotations", "export", valid_path, len(valid_rows))
    record_data_artifact(ctx, "export_summary", "export", summary_path, 1)
    ctx.write_stage_outputs("export", summary, sample=working_rows[: ctx.config.runtime.sample_limit], errors=[])


def run_final_report(ctx: RunContext) -> None:
    export_summary = read_json(ctx.path("exports", "final_annotations.summary.json"), default={}) or {}
    report_path = ctx.path("exports", "final_report.json")
    write_json(
        report_path,
        {
            "run_id": ctx.run_id,
            "export": export_summary,
            "acceptance": ctx.config.acceptance.model_dump(mode="json"),
            "note": "Acceptance targets are report-only in v1; export is manually triggered by the user.",
        },
    )
    record_data_artifact(ctx, "final_report", "final_report", report_path, 1)
    ctx.write_stage_outputs("final_report", {"stage": "final_report", "report_uri": ctx.uri(report_path)}, sample=[], errors=[])


def stable_image_id(dataset_id: str, uri: str) -> str:
    return hashlib.sha1(f"{dataset_id}:{uri}".encode("utf-8")).hexdigest()[:16]


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _local_path_uri(path: Path) -> str:
    if not path.is_absolute():
        path = path.resolve()
    return path.as_uri()


def _load_input_rows(ctx: RunContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_config = ctx.config.input
    errors: list[dict[str, Any]] = []
    if input_config.type == "manifest":
        manifest_path = uri_to_path(input_config.manifest_uri or "")
        rows = read_jsonl(manifest_path)
        normalized = []
        for idx, row in enumerate(rows):
            uri = row.get("uri")
            if not uri:
                errors.append({"row_index": idx, "error_code": "missing_uri", "message": "manifest row missing uri"})
                continue
            normalized.append(
                {
                    "image_id": row.get("image_id"),
                    "uri": ensure_local_uri(uri) if is_local_uri(uri) else uri,
                    "metadata": row.get("metadata") or {},
                }
            )
        return normalized, errors
    if input_config.type == "path_list":
        list_path = uri_to_path(input_config.path_list_uri or "")
        rows = []
        for line_number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append({"uri": ensure_local_uri(line), "metadata": {"path_list_line": line_number}})
            except Exception as exc:  # noqa: BLE001
                errors.append({"row_index": line_number, "error_code": exc.__class__.__name__, "message": str(exc)})
        return rows, errors
    if input_config.type in {"local_dir", "object_prefix"}:
        raw_dir = input_config.local_dir if input_config.type == "local_dir" else input_config.object_prefix_uri
        root = uri_to_path(raw_dir or "").expanduser()
        if not root.is_absolute():
            root = root.resolve()
        globber = root.rglob("*") if input_config.recursive else root.glob("*")
        allowed = {ext.lower() for ext in (input_config.extensions or IMAGE_EXTENSIONS)}
        paths: list[Path] = []
        scanned = 0
        last_reported = 0
        progress_key = "input_scan"
        ctx.report_progress("task_started", key=progress_key, description="scan input files", total=None)
        ctx.update_stage_progress("ingest", processed_items=0, matched_items=0, progress_phase="input_scan")
        for path in globber:
            scanned += 1
            if path.is_file() and path.suffix.lower() in allowed:
                paths.append(path)
            if scanned % PROGRESS_INTERVAL == 0:
                ctx.report_progress(
                    "task_advance",
                    key=progress_key,
                    advance=scanned - last_reported,
                    description=f"scan input files ({len(paths)} images)",
                )
                ctx.update_stage_progress("ingest", processed_items=scanned, matched_items=len(paths))
                last_reported = scanned
        if scanned > last_reported:
            ctx.report_progress(
                "task_advance",
                key=progress_key,
                advance=scanned - last_reported,
                description=f"scan input files ({len(paths)} images)",
            )
        ctx.report_progress(
            "task_completed",
            key=progress_key,
            description=f"scan input files ({len(paths)} images)",
            completed=scanned,
        )
        ctx.update_stage_progress(
            "ingest",
            processed_items=scanned,
            matched_items=len(paths),
            progress_phase="input_prepare",
        )
        rows = []
        prepare_key = "input_prepare"
        last_reported = 0
        ctx.report_progress("task_started", key=prepare_key, description="prepare input rows", total=len(paths))
        for idx, path in enumerate(sorted(paths), start=1):
            rows.append({"uri": _local_path_uri(path), "metadata": {"relative_path": str(path.relative_to(root))}})
            if idx % PROGRESS_INTERVAL == 0 or idx == len(paths):
                ctx.report_progress("task_advance", key=prepare_key, advance=idx - last_reported)
                ctx.update_stage_progress(
                    "ingest",
                    processed_items=idx,
                    matched_items=len(paths),
                    progress_phase="input_prepare",
                )
                last_reported = idx
        ctx.report_progress(
            "task_completed",
            key=prepare_key,
            description=f"prepare input rows ({len(rows)} images)",
            completed=len(rows),
            total=len(paths),
        )
        return rows, errors
    raise ValueError(f"Unsupported input type: {input_config.type}")


def _quality_for_row(ctx: RunContext, row: dict[str, Any]) -> dict[str, Any]:
    return _quality_for_row_with_config(ctx.config.quality, row)


def _quality_for_row_with_config(quality_config: QualityConfig, row: dict[str, Any]) -> dict[str, Any]:
    result = {
        **row,
        "quality_status": "keep",
        "quality_reasons": [],
        "width": None,
        "height": None,
        "format": None,
        "perceptual_hash": None,
        "blur_score": None,
        "channel_stddev": None,
    }
    path = Path(row["local_path"]) if row.get("local_path") else None
    actions: list[str] = []
    reasons: list[str] = []
    if not path or not path.exists():
        return {**result, "quality_status": "drop", "quality_reasons": ["missing_local_file"], "message": "file does not exist"}

    file_size = path.stat().st_size
    if file_size < quality_config.min_file_size_bytes:
        actions.append("needs_review")
        reasons.append("file_too_small")
    if quality_config.max_file_size_bytes and file_size > quality_config.max_file_size_bytes:
        actions.append("quarantine")
        reasons.append("file_too_large")

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            fmt = (image.format or "").upper()
            result["width"] = width
            result["height"] = height
            result["format"] = fmt
            result["perceptual_hash"] = _dhash(image)
            if fmt not in quality_config.allowed_formats:
                actions.append("drop")
                reasons.append("unsupported_format")
            if width < quality_config.min_width or height < quality_config.min_height:
                actions.append("drop")
                reasons.append("image_too_small")
            if width * height < quality_config.min_pixels:
                actions.append("drop")
                reasons.append("too_few_pixels")
            aspect = max(width / max(height, 1), height / max(width, 1))
            if aspect > quality_config.max_aspect_ratio:
                actions.append("needs_review")
                reasons.append("aspect_ratio_extreme")
            if quality_config.blur.enabled:
                blur_score = _blur_score(image)
                result["blur_score"] = blur_score
                if blur_score < quality_config.blur.edge_variance_threshold:
                    actions.append(quality_config.blur.action)
                    reasons.append("blurry")
            if quality_config.low_information.enabled:
                stddev = _channel_stddev(image)
                result["channel_stddev"] = stddev
                if stddev < quality_config.low_information.channel_stddev_threshold:
                    actions.append(quality_config.low_information.action)
                    reasons.append("low_information")
    except (UnidentifiedImageError, OSError) as exc:
        return {**result, "quality_status": "drop", "quality_reasons": ["decode_failed"], "message": str(exc)}

    result["quality_status"] = _highest_priority(actions) if actions else "keep"
    result["quality_reasons"] = reasons
    return result


def _highest_priority(actions: list[str]) -> str:
    return max(actions, key=lambda action: ACTION_PRIORITY[action]) if actions else "keep"


def _dhash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8))
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = "".join("1" if item else "0" for item in diff.flatten())
    return f"{int(bits, 2):016x}"


def _blur_score(image: Image.Image) -> float:
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).var[0])


def _channel_stddev(image: Image.Image) -> float:
    stat = ImageStat.Stat(image.convert("RGB").resize((64, 64)))
    return float(sum(stat.stddev) / len(stat.stddev))


def _load_vectors(ctx: RunContext) -> tuple[np.ndarray, list[str]]:
    vector_path = ctx.path("embeddings", "embeddings.npy")
    id_path = ctx.path("embeddings", "ids.json")
    vectors = np.load(vector_path).astype(np.float32) if vector_path.exists() else np.zeros((0, 0), dtype=np.float32)
    ids = read_json(id_path, default=[]) or []
    if len(ids) and ctx.config.embedding.normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
    return vectors, ids


def _duplicate_groups(ids: list[str], edges: list[tuple[str, str, float]]) -> list[dict[str, Any]]:
    parent = {image_id: image_id for image_id in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    edge_scores: dict[tuple[str, str], float] = {}
    for left, right, score in edges:
        union(left, right)
        edge_scores[tuple(sorted((left, right)))] = max(score, edge_scores.get(tuple(sorted((left, right))), 0.0))

    groups: dict[str, list[str]] = defaultdict(list)
    for image_id in ids:
        groups[find(image_id)].append(image_id)
    output = []
    for index, members in enumerate(groups.values(), start=1):
        if len(members) <= 1:
            continue
        scores = [
            score
            for pair, score in edge_scores.items()
            if pair[0] in members and pair[1] in members
        ]
        output.append(
            {
                "duplicate_group_id": f"dup_{index:05d}",
                "image_ids": sorted(members),
                "size": len(members),
                "max_similarity": max(scores) if scores else None,
            }
        )
    return output


def _effective_cluster_count(ctx: RunContext, valid_image_count: int) -> tuple[int, int]:
    configured = ctx.config.clustering.num_clusters
    if configured == "auto":
        requested = min(10000, max(100, math.ceil(math.sqrt(valid_image_count))))
    else:
        requested = int(configured)
    return max(1, min(valid_image_count, requested)), requested


def _cluster_label_stats(
    prelabels: dict[str, dict[str, Any]],
    clusters: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for image_id, cluster in clusters.items():
        candidates = prelabels.get(image_id, {}).get("candidates", [])
        if candidates:
            counts[cluster["cluster_id"]][candidates[0]["label"]] += 1
    stats = {}
    for cluster_id, counter in counts.items():
        total = sum(counter.values())
        label, count = counter.most_common(1)[0]
        stats[cluster_id] = {"main_label": label, "main_label_ratio": count / total if total else 0.0, "label_counts": dict(counter)}
    return stats


def _duplicate_conflict_images(prelabels: dict[str, dict[str, Any]], groups: list[dict[str, Any]]) -> set[str]:
    conflict_images = set()
    for group in groups:
        labels = {
            prelabels.get(image_id, {}).get("candidates", [{}])[0].get("label")
            for image_id in group.get("image_ids", [])
            if prelabels.get(image_id, {}).get("candidates")
        }
        labels.discard(None)
        if len(labels) > 1:
            conflict_images.update(group.get("image_ids", []))
    return conflict_images


def _auto_decision_reasons(
    ctx: RunContext,
    quality: dict[str, Any],
    candidates: list[dict[str, Any]],
    cluster: dict[str, Any] | None,
    cluster_label_stats: dict[str, dict[str, Any]],
    conflict_images: set[str],
    high_risk: set[str],
) -> list[str]:
    reasons: list[str] = []
    if quality["quality_status"] != "keep":
        reasons.append(f"quality_{quality['quality_status']}")
    if not candidates:
        reasons.append("missing_prelabel")
        return reasons
    top = candidates[0]
    second_conf = candidates[1]["confidence"] if len(candidates) > 1 else 0.0
    if top["confidence"] < ctx.config.auto_decision.min_confidence:
        reasons.append("low_confidence")
    if top["confidence"] - second_conf < ctx.config.auto_decision.min_margin:
        reasons.append("low_margin")
    if ctx.config.auto_decision.block_high_risk_labels and top["label"] in high_risk:
        reasons.append("high_risk_label")
    if quality["image_id"] in conflict_images:
        reasons.append("duplicate_label_conflict")
    if ctx.config.auto_decision.require_cluster_consistency and cluster:
        stats = cluster_label_stats.get(cluster["cluster_id"])
        if stats and (
            stats["main_label"] != top["label"]
            or stats["main_label_ratio"] < ctx.config.auto_decision.min_cluster_main_label_ratio
        ):
            reasons.append("cluster_label_inconsistent")
    return reasons


def _initial_non_auto_status(quality: dict[str, Any]) -> str:
    if quality["quality_status"] == "drop":
        return "invalid"
    return "pending"


def _review_reasons(ctx: RunContext, row: dict[str, Any], duplicate_images: set[str]) -> list[str]:
    reasons: list[str] = []
    if row["final_status"] == "pending":
        reasons.append("pending")
    if row["quality_status"] != "keep":
        reasons.append("quality_risk")
    candidates = row.get("prelabel_candidates") or []
    if ctx.config.review.queue.include_low_confidence and candidates and candidates[0]["confidence"] < ctx.config.auto_decision.min_confidence:
        reasons.append("low_confidence")
    cluster = row.get("cluster") or {}
    if ctx.config.review.queue.include_outliers and cluster.get("is_outlier"):
        reasons.append("outlier")
    if ctx.config.review.queue.include_cluster_representatives and cluster.get("is_representative"):
        reasons.append("cluster_representative")
    if ctx.config.review.queue.include_duplicate_conflicts and row["image_id"] in duplicate_images:
        reasons.append("duplicate_candidate")
    return list(dict.fromkeys(reasons))


def _review_priority(reasons: list[str]) -> int:
    order = {
        "pending": 10,
        "low_confidence": 20,
        "quality_risk": 30,
        "duplicate_candidate": 40,
        "outlier": 50,
        "cluster_representative": 60,
        "auto_accept_sample": 70,
    }
    return min(order.get(reason, 99) for reason in reasons)


def _stable_sample(image_id: str, ratio: float) -> bool:
    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    value = int(hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < ratio


def _thumbnail_uri(ctx: RunContext, image_id: str) -> str | None:
    path = ctx.path("thumbnails", f"{image_id}.jpg")
    return ctx.uri(path) if path.exists() else None


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.95:
        return "0.95-1.00"
    if confidence >= 0.9:
        return "0.90-0.95"
    if confidence >= 0.8:
        return "0.80-0.90"
    if confidence >= 0.6:
        return "0.60-0.80"
    return "0.00-0.60"


def _stage_error(row: dict[str, Any], error_code: str, message: str) -> dict[str, Any]:
    return {
        "image_id": row.get("image_id"),
        "uri": row.get("uri"),
        "error_code": error_code,
        "message": message,
    }


def _compact_summary(summary: dict[str, Any]) -> str:
    keys = [
        "output_count",
        "success_count",
        "failed_count",
        "auto_accept_count",
        "queue_count",
        "valid_count",
        "cluster_count",
        "effective_cluster_count",
    ]
    parts = [f"{key}={summary[key]}" for key in keys if key in summary]
    return ", ".join(parts) if parts else "ok"


def _apply_review_decisions(ctx: RunContext) -> list[dict[str, Any]]:
    rows = read_jsonl(ctx.path("decisions", "final_annotations.jsonl"))
    decisions = read_jsonl(ctx.path("review", "review_decisions.jsonl"))
    by_id = {row["image_id"]: row for row in rows}
    for decision in decisions:
        row = by_id.get(decision.get("image_id"))
        if not row:
            continue
        action = decision.get("action_type")
        if action in {"accept", "human_accepted"}:
            row["final_status"] = "human_accepted"
            row["final_label"] = decision.get("final_label") or row.get("recommended_label")
            row["source"] = "human_review"
        elif action in {"correct", "corrected"}:
            row["final_status"] = "corrected"
            row["final_label"] = decision.get("final_label")
            row["source"] = "human_review"
        elif action in {"reject", "rejected"}:
            row["final_status"] = "rejected"
            row["final_label"] = None
            row["source"] = "human_review"
        elif action == "invalid":
            row["final_status"] = "invalid"
            row["final_label"] = None
            row["source"] = "human_review"
        row["review_note"] = decision.get("note")
        row["reviewed_at"] = decision.get("created_at")
    write_jsonl(ctx.path("decisions", "final_annotations.jsonl"), rows)
    return rows
