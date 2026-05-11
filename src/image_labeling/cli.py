from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from .api import create_app
from .config import load_config
from .jsonio import read_json, read_jsonl
from .pipeline import AUTO_STAGES, export_run, resume_run, run_single_stage, start_run
from .storage import copy_report_bundle, list_runs, load_run, read_artifact_rows


console = Console()
app = typer.Typer(help="Image classification cleaning and pre-labeling pipeline.")
config_app = typer.Typer(help="Configuration commands.")
run_app = typer.Typer(help="Run lifecycle commands.")
stage_app = typer.Typer(help="Stage commands.")
artifacts_app = typer.Typer(help="Artifact inspection commands.")
report_app = typer.Typer(help="Report commands.")
api_app = typer.Typer(help="API commands.")

app.add_typer(config_app, name="config")
app.add_typer(run_app, name="run")
app.add_typer(stage_app, name="stage")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(report_app, name="report")
app.add_typer(api_app, name="api")


@config_app.command("validate")
def validate_config(config: Path = typer.Option(..., "-c", "--config", exists=True, readable=True)) -> None:
    loaded = load_config(config)
    console.print("[green]config valid[/green]")
    console.print(JSON.from_data(loaded.model_dump(mode="json")))


@run_app.command("start")
def start(
    config: Path = typer.Option(..., "-c", "--config", exists=True, readable=True),
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root", help="Override local artifact root URI/path."),
    until: Optional[str] = typer.Option(None, "--until", help=f"Stop after a stage: {', '.join(AUTO_STAGES)}"),
    from_stage: Optional[str] = typer.Option(None, "--from-stage", help="Start from a stage in the new run."),
    force: bool = typer.Option(False, "--force", help="Rerun stages even if they already succeeded."),
) -> None:
    ctx = start_run(load_config(config), artifact_root=artifact_root, until=until, from_stage=from_stage, force=force)
    console.print(f"run_id: [bold]{ctx.run_id}[/bold]")
    console.print(f"status: {ctx.state['status']}")
    console.print(f"run_dir: {ctx.run_dir}")
    console.print(f"review_api_url: http://localhost:8000/runs/{ctx.run_id}/review/tasks")


@run_app.command("resume")
def resume(
    run_id: str = typer.Option(..., "--run-id"),
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    ctx = resume_run(run_id, artifact_root=artifact_root, force=force)
    console.print(f"run_id: [bold]{ctx.run_id}[/bold]")
    console.print(f"status: {ctx.state['status']}")


@run_app.command("list")
def list_run_command(artifact_root: Optional[str] = typer.Option(None, "--artifact-root")) -> None:
    rows = list_runs(artifact_root=artifact_root)
    table = Table("run_id", "project", "dataset", "status", "current_stage", "run_dir")
    for row in rows:
        table.add_row(
            row.get("run_id", ""),
            row.get("project_id", ""),
            row.get("dataset_id", ""),
            row.get("status") or "",
            row.get("current_stage") or "",
            row.get("run_dir", ""),
        )
    console.print(table)


@run_app.command("status")
def status(run_id: str = typer.Option(..., "--run-id"), artifact_root: Optional[str] = typer.Option(None, "--artifact-root")) -> None:
    ctx = load_run(run_id, artifact_root=artifact_root)
    state = ctx.state
    table = Table("field", "value")
    for key in ["run_id", "status", "current_stage", "created_at", "started_at", "finished_at", "config_snapshot_uri", "input_manifest_uri"]:
        table.add_row(key, str(state.get(key)))
    console.print(table)
    stage_table = Table("stage", "status", "total", "succeeded", "failed", "duration_ms")
    for stage, item in state.get("stages", {}).items():
        stage_table.add_row(
            stage,
            str(item.get("status")),
            str(item.get("total_items")),
            str(item.get("succeeded_items")),
            str(item.get("failed_items")),
            str(item.get("duration_ms")),
        )
    console.print(stage_table)


@run_app.command("summary")
def summary(run_id: str = typer.Option(..., "--run-id"), artifact_root: Optional[str] = typer.Option(None, "--artifact-root")) -> None:
    ctx = load_run(run_id, artifact_root=artifact_root)
    payload = {
        "run_id": ctx.run_id,
        "status": ctx.state.get("status"),
        "summaries": {
            path.parent.name: read_json(path)
            for path in sorted(ctx.path("summaries").glob("*/stage_summary.json"))
        },
    }
    console.print(JSON.from_data(payload))


@run_app.command("logs")
def logs(
    run_id: str = typer.Option(..., "--run-id"),
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root"),
    stage: Optional[str] = typer.Option(None, "--stage"),
    level: Optional[str] = typer.Option(None, "--level"),
    follow: bool = typer.Option(False, "--follow", "-f"),
) -> None:
    ctx = load_run(run_id, artifact_root=artifact_root)
    log_path = ctx.path("logs", "pipeline.jsonl") if not stage else ctx.path("logs", stage, "worker-0.jsonl")
    seen = 0
    while True:
        rows = read_jsonl(log_path)
        for row in rows[seen:]:
            if level and row.get("level") != level:
                continue
            console.print(json.dumps(row, ensure_ascii=False))
        seen = len(rows)
        if not follow:
            break
        time.sleep(1)


@stage_app.command("run")
def stage_run(
    run_id: str = typer.Option(..., "--run-id"),
    stage: str = typer.Option(..., "--stage"),
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    ctx = run_single_stage(run_id, stage, artifact_root=artifact_root, force=force)
    console.print(f"run_id: [bold]{ctx.run_id}[/bold]")
    console.print(f"stage: {stage}")
    console.print(f"status: {ctx.state['stages'][stage]['status']}")


@stage_app.command("summary")
def stage_summary(
    run_id: str = typer.Option(..., "--run-id"),
    stage: str = typer.Option(..., "--stage"),
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root"),
) -> None:
    ctx = load_run(run_id, artifact_root=artifact_root)
    summary_path = ctx.summary_path(stage)
    if not summary_path.exists():
        raise typer.BadParameter(f"stage summary not found: {stage}")
    console.print(JSON.from_data(read_json(summary_path)))


@artifacts_app.command("list")
def artifacts_list(
    run_id: str = typer.Option(..., "--run-id"),
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root"),
) -> None:
    ctx = load_run(run_id, artifact_root=artifact_root)
    table = Table("name", "stage", "records", "checksum", "uri")
    for item in ctx.state.get("artifacts", []):
        table.add_row(
            item.get("name", ""),
            item.get("stage", ""),
            str(item.get("records")),
            str(item.get("checksum") or ""),
            item.get("uri", ""),
        )
    console.print(table)


@artifacts_app.command("sample")
def artifacts_sample(
    run_id: str = typer.Option(..., "--run-id"),
    artifact: str = typer.Option(..., "--artifact"),
    limit: int = typer.Option(20, "--limit"),
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root"),
) -> None:
    ctx = load_run(run_id, artifact_root=artifact_root)
    rows = read_artifact_rows(ctx, artifact, limit)
    console.print(JSON.from_data(rows))


@report_app.command("bundle")
def report_bundle(
    run_id: str = typer.Option(..., "--run-id"),
    output: Path = typer.Option(..., "-o", "--output"),
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root"),
) -> None:
    ctx = load_run(run_id, artifact_root=artifact_root)
    copy_report_bundle(ctx, output)
    console.print(f"report bundle written: {output}")


@app.command("export")
def export(
    run_id: str = typer.Option(..., "--run-id"),
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root"),
    yes: bool = typer.Option(False, "--yes", help="Confirm manual review decision and export without prompt."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    ctx = load_run(run_id, artifact_root=artifact_root)
    queue_summary = read_json(ctx.summary_path("review_queue"), default={}) or {}
    auto_summary = read_json(ctx.summary_path("auto_decision"), default={}) or {}
    console.print("[bold]Export pre-check[/bold]")
    console.print(JSON.from_data({"review_queue": queue_summary, "auto_decision": auto_summary}))
    if not yes and not typer.confirm("人工已根据摘要/审核结果确认可以导出？"):
        raise typer.Abort()
    ctx = export_run(run_id, artifact_root=artifact_root, force=force)
    export_summary = read_json(ctx.path("exports", "final_annotations.summary.json"))
    console.print(JSON.from_data(export_summary))


@api_app.command("serve")
def serve(
    artifact_root: Optional[str] = typer.Option(None, "--artifact-root"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    uvicorn.run(create_app(artifact_root=artifact_root), host=host, port=port)


if __name__ == "__main__":
    app()
