from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .jsonio import append_jsonl, read_json, read_jsonl
from .logging_utils import utc_now
from .storage import list_runs, load_run


class ReviewDecisionIn(BaseModel):
    image_id: str
    action_type: str
    final_label: str | None = None
    note: str | None = None


def create_app(artifact_root: str | None = None) -> FastAPI:
    app = FastAPI(title="DataCleanTools Review API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runs")
    def runs() -> list[dict[str, Any]]:
        return list_runs(artifact_root=artifact_root)

    @app.get("/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict[str, Any]:
        ctx = _load(run_id, artifact_root)
        summaries = {
            path.parent.name: read_json(path)
            for path in sorted(ctx.path("summaries").glob("*/stage_summary.json"))
        }
        return {"state": ctx.state, "summaries": summaries}

    @app.get("/runs/{run_id}/review/tasks")
    def review_tasks(run_id: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        ctx = _load(run_id, artifact_root)
        rows = read_jsonl(ctx.path("review", "review_queue.jsonl"))
        return {"total": len(rows), "items": rows[offset : offset + limit]}

    @app.get("/runs/{run_id}/images/{image_id}")
    def image_detail(run_id: str, image_id: str) -> dict[str, Any]:
        ctx = _load(run_id, artifact_root)
        final_rows = {row["image_id"]: row for row in read_jsonl(ctx.path("decisions", "final_annotations.jsonl"))}
        if image_id not in final_rows:
            raise HTTPException(status_code=404, detail="image not found")
        similar = None
        for row in read_jsonl(ctx.path("similarity", "topk.jsonl")):
            if row["image_id"] == image_id:
                similar = row
                break
        return {"annotation": final_rows[image_id], "similarity": similar}

    @app.post("/runs/{run_id}/review/decisions")
    def save_review_decision(run_id: str, decision: ReviewDecisionIn) -> dict[str, Any]:
        ctx = _load(run_id, artifact_root)
        valid_actions = {"accept", "human_accepted", "correct", "corrected", "reject", "rejected", "invalid"}
        if decision.action_type not in valid_actions:
            raise HTTPException(status_code=400, detail=f"invalid action_type: {decision.action_type}")
        row = decision.model_dump()
        row.update({"run_id": run_id, "created_at": utc_now()})
        append_jsonl(ctx.path("review", "review_decisions.jsonl"), row)
        return {"status": "saved", "decision": row}

    @app.get("/runs/{run_id}/exports")
    def export_status(run_id: str) -> dict[str, Any]:
        ctx = _load(run_id, artifact_root)
        summary_path = ctx.path("exports", "final_annotations.summary.json")
        return read_json(summary_path, default={"status": "not_exported"}) or {"status": "not_exported"}

    return app


def _load(run_id: str, artifact_root: str | None):
    try:
        return load_run(run_id, artifact_root=artifact_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
