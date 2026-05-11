from __future__ import annotations

from typing import Callable

from .logging_utils import duration_ms, utc_now
from .stages import (
    run_auto_decision,
    run_clustering,
    run_embedding,
    run_export,
    run_final_report,
    run_ingest,
    run_prelabel,
    run_quality_check,
    run_report,
    run_review_queue,
    run_similarity,
    run_thumbnail,
    run_vector_index,
)
from .storage import RunContext, create_run, load_run


StageFn = Callable[[RunContext], None]


AUTO_STAGES: list[str] = [
    "ingest",
    "quality_check",
    "thumbnail",
    "embedding",
    "vector_index",
    "similarity",
    "clustering",
    "prelabel",
    "auto_decision",
    "review_queue",
    "report",
]

EXPORT_STAGES: list[str] = ["export", "final_report"]

STAGE_FUNCTIONS: dict[str, StageFn] = {
    "ingest": run_ingest,
    "quality_check": run_quality_check,
    "thumbnail": run_thumbnail,
    "embedding": run_embedding,
    "vector_index": run_vector_index,
    "similarity": run_similarity,
    "clustering": run_clustering,
    "prelabel": run_prelabel,
    "auto_decision": run_auto_decision,
    "review_queue": run_review_queue,
    "report": run_report,
    "export": run_export,
    "final_report": run_final_report,
}


def start_run(
    config,
    *,
    artifact_root: str | None = None,
    until: str | None = None,
    from_stage: str | None = None,
    force: bool = False,
) -> RunContext:
    ctx = create_run(config, artifact_root=artifact_root)
    run_stages(ctx, AUTO_STAGES, until=until, from_stage=from_stage, force=force)
    return ctx


def resume_run(run_id: str, *, artifact_root: str | None = None, force: bool = False) -> RunContext:
    ctx = load_run(run_id, artifact_root=artifact_root)
    run_stages(ctx, AUTO_STAGES, force=force, resume=True)
    return ctx


def run_single_stage(run_id: str, stage: str, *, artifact_root: str | None = None, force: bool = False) -> RunContext:
    ctx = load_run(run_id, artifact_root=artifact_root)
    if stage not in STAGE_FUNCTIONS:
        raise ValueError(f"Unknown stage: {stage}")
    _run_stage(ctx, stage, force=force)
    if stage in AUTO_STAGES and ctx.state.get("status") not in {"failed", "review_ready"}:
        ctx.state["status"] = "paused"
    ctx.save_state()
    return ctx


def export_run(run_id: str, *, artifact_root: str | None = None, force: bool = False) -> RunContext:
    ctx = load_run(run_id, artifact_root=artifact_root)
    ctx.state["status"] = "exporting"
    ctx.save_state()
    for stage in EXPORT_STAGES:
        _run_stage(ctx, stage, force=force)
    ctx.state["status"] = "completed"
    ctx.state["finished_at"] = utc_now()
    ctx.state["current_stage"] = "final_report"
    ctx.save_state()
    return ctx


def run_stages(
    ctx: RunContext,
    stages: list[str],
    *,
    until: str | None = None,
    from_stage: str | None = None,
    force: bool = False,
    resume: bool = False,
) -> RunContext:
    selected = _select_stages(stages, until=until, from_stage=from_stage)
    ctx.state["status"] = "running"
    ctx.state["started_at"] = ctx.state.get("started_at") or utc_now()
    ctx.save_state()
    for stage in selected:
        current_status = ctx.state.get("stages", {}).get(stage, {}).get("status")
        if resume and current_status == "succeeded" and not force:
            continue
        _run_stage(ctx, stage, force=force)
    if until:
        ctx.state["status"] = "paused" if until != stages[-1] else "review_ready"
    else:
        ctx.state["status"] = "review_ready" if stages == AUTO_STAGES else ctx.state["status"]
    ctx.state["current_stage"] = selected[-1] if selected else ctx.state.get("current_stage")
    ctx.save_state()
    return ctx


def _select_stages(stages: list[str], *, until: str | None, from_stage: str | None) -> list[str]:
    for stage in [until, from_stage]:
        if stage and stage not in stages:
            raise ValueError(f"Stage '{stage}' is not valid for this command")
    start = stages.index(from_stage) if from_stage else 0
    end = stages.index(until) + 1 if until else len(stages)
    if start > end:
        raise ValueError("--from-stage must be before or equal to --until")
    return stages[start:end]


def _run_stage(ctx: RunContext, stage: str, *, force: bool = False) -> None:
    stage_state = ctx.state.setdefault("stages", {}).setdefault(stage, {})
    if stage_state.get("status") == "succeeded" and not force:
        ctx.logger.log(stage, "stage_skipped", "stage already succeeded; use --force to rerun")
        return
    if force:
        _mark_downstream_needs_recompute(ctx, stage)
    start = utc_now()
    ctx.state["current_stage"] = stage
    stage_state.update(
        {
            "stage_name": stage,
            "status": "running",
            "started_at": start,
            "finished_at": None,
            "artifact_uri": None,
            "metrics_json": {},
            "error_json": None,
        }
    )
    ctx.save_state()
    ctx.logger.log(stage, "stage_started", f"{stage} started")
    try:
        STAGE_FUNCTIONS[stage](ctx)
        end = utc_now()
        summary = _load_stage_summary(ctx, stage)
        stage_state.update(
            {
                "status": "succeeded",
                "total_items": _summary_count(summary, "input_count", "image_count", "full_count"),
                "succeeded_items": _summary_count(summary, "success_count", "output_count", "valid_count", "auto_accept_count"),
                "failed_items": _summary_count(summary, "failed_count"),
                "artifact_uri": ctx.uri(ctx.summary_path(stage)),
                "metrics_json": summary,
                "finished_at": end,
                "duration_ms": duration_ms(start, end),
            }
        )
        ctx.logger.log(stage, "stage_completed", f"{stage} completed", metrics={"duration_ms": stage_state["duration_ms"]})
    except Exception as exc:  # noqa: BLE001
        end = utc_now()
        stage_state.update(
            {
                "status": "failed",
                "failed_items": stage_state.get("total_items"),
                "finished_at": end,
                "duration_ms": duration_ms(start, end),
                "error_json": {"error_code": exc.__class__.__name__, "message": str(exc)},
            }
        )
        ctx.state["status"] = "failed"
        ctx.state["error_json"] = {"stage": stage, "error_code": exc.__class__.__name__, "message": str(exc)}
        ctx.logger.log(stage, "stage_failed", str(exc), level="ERROR", error_code=exc.__class__.__name__)
        ctx.save_state()
        raise
    ctx.save_state()


def _mark_downstream_needs_recompute(ctx: RunContext, stage: str) -> None:
    all_stages = AUTO_STAGES + EXPORT_STAGES
    if stage not in all_stages:
        return
    for downstream in all_stages[all_stages.index(stage) + 1 :]:
        existing = ctx.state.get("stages", {}).get(downstream)
        if existing and existing.get("status") == "succeeded":
            existing["status"] = "needs_recompute"


def _load_stage_summary(ctx: RunContext, stage: str) -> dict:
    from .jsonio import read_json

    return read_json(ctx.summary_path(stage), default={}) or {}


def _summary_count(summary: dict, *keys: str) -> int | None:
    for key in keys:
        if isinstance(summary.get(key), int):
            return int(summary[key])
    return None
