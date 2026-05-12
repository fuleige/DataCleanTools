from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig, dump_config_yaml, load_config
from .jsonio import read_json, read_jsonl, write_json, write_jsonl
from .logging_utils import PipelineLogger, utc_now
from .uri import uri_to_path


RUN_INDEX_DIR = "_runs"


def local_artifact_root(config: PipelineConfig | None = None, artifact_root: str | None = None) -> Path:
    if artifact_root:
        return uri_to_path(artifact_root)
    if config:
        return uri_to_path(config.storage.artifact_store.root_uri)
    return Path("/data/dataclean-artifacts")


def generate_run_id() -> str:
    import secrets
    from datetime import datetime, timezone

    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunContext:
    config: PipelineConfig
    run_id: str
    artifact_root: Path
    run_dir: Path
    state: dict[str, Any]
    logger: PipelineLogger
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def summary_path(self, stage: str) -> Path:
        return self.path("summaries", stage, "stage_summary.json")

    def sample_path(self, stage: str) -> Path:
        return self.path("summaries", stage, "sample.jsonl")

    def errors_path(self, stage: str) -> Path:
        return self.path("summaries", stage, "errors.jsonl")

    def uri(self, path: Path) -> str:
        return path.resolve().as_uri()

    def save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        write_json(self.path("state.json"), self.state)

    def report_progress(self, event: str, **payload: Any) -> None:
        if self.progress_callback:
            self.progress_callback({"event": event, **payload})

    def update_stage_progress(
        self,
        stage: str,
        *,
        processed_items: int | None = None,
        total_items: int | None = None,
        **extra: Any,
    ) -> None:
        stage_state = self.state.setdefault("stages", {}).setdefault(stage, {})
        if processed_items is not None:
            stage_state["processed_items"] = processed_items
        if total_items is not None:
            stage_state["total_items"] = total_items
        stage_state.update(extra)
        self.save_state()

    def add_artifact(
        self,
        *,
        name: str,
        stage: str,
        path: Path,
        records: int | None = None,
        artifact_type: str = "file",
    ) -> None:
        artifact = {
            "name": name,
            "stage": stage,
            "type": artifact_type,
            "uri": self.uri(path),
            "records": records,
            "checksum": sha256_file(path),
            "created_at": utc_now(),
        }
        artifacts = [item for item in self.state.get("artifacts", []) if item.get("name") != name]
        artifacts.append(artifact)
        self.state["artifacts"] = artifacts
        write_json(self.path("artifact_index.json"), artifacts)
        self.save_state()

    def write_stage_outputs(
        self,
        stage: str,
        summary: dict[str, Any],
        *,
        sample: list[dict[str, Any]] | None = None,
        errors: list[dict[str, Any]] | None = None,
        errors_prewritten: bool = False,
        errors_count: int | None = None,
    ) -> None:
        write_json(self.summary_path(stage), summary)
        write_jsonl(self.sample_path(stage), sample or [])
        if not errors_prewritten:
            write_jsonl(self.errors_path(stage), errors or [])
        elif not self.errors_path(stage).exists():
            write_jsonl(self.errors_path(stage), [])
        self.add_artifact(name=f"{stage}_summary", stage=stage, path=self.summary_path(stage), records=1)
        self.add_artifact(name=f"{stage}_sample", stage=stage, path=self.sample_path(stage), records=len(sample or []))
        self.add_artifact(
            name=f"{stage}_errors",
            stage=stage,
            path=self.errors_path(stage),
            records=errors_count if errors_count is not None else len(errors or []),
        )


def _run_dir(root: Path, config: PipelineConfig, run_id: str) -> Path:
    return root / config.project.id / config.project.dataset_id / run_id


def create_run(config: PipelineConfig, artifact_root: str | None = None, run_id: str | None = None) -> RunContext:
    root = local_artifact_root(config, artifact_root)
    run_id = run_id or generate_run_id()
    run_dir = _run_dir(root, config, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    dump_config_yaml(config, run_dir / "resolved_config.yaml")
    state = {
        "run_id": run_id,
        "project": config.project.model_dump(mode="json"),
        "dataset_id": config.project.dataset_id,
        "label_schema_id": config.label_schema.schema_id,
        "status": "created",
        "current_stage": None,
        "config_snapshot_uri": (run_dir / "resolved_config.yaml").resolve().as_uri(),
        "input_manifest_uri": None,
        "output_prefix": (run_dir / "exports").resolve().as_uri(),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "error_json": None,
        "stages": {},
        "artifacts": [],
    }
    write_json(run_dir / "state.json", state)
    root.joinpath(RUN_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    write_json(
        root / RUN_INDEX_DIR / f"{run_id}.json",
        {
            "run_id": run_id,
            "run_dir": run_dir.resolve().as_uri(),
            "project_id": config.project.id,
            "dataset_id": config.project.dataset_id,
            "created_at": state["created_at"],
        },
    )
    return RunContext(config, run_id, root, run_dir, state, PipelineLogger(run_id, run_dir))


def load_run(run_id: str, artifact_root: str | None = None) -> RunContext:
    root = local_artifact_root(artifact_root=artifact_root)
    index_path = root / RUN_INDEX_DIR / f"{run_id}.json"
    if not index_path.exists():
        matches = list(root.glob(f"*/*/{run_id}/state.json"))
        if not matches:
            raise FileNotFoundError(f"run not found: {run_id}. Use --artifact-root if it was created elsewhere.")
        run_dir = matches[0].parent
    else:
        run_dir = uri_to_path(read_json(index_path)["run_dir"])
    config = load_config(run_dir / "resolved_config.yaml")
    state = read_json(run_dir / "state.json")
    return RunContext(config, run_id, root, run_dir, state, PipelineLogger(run_id, run_dir))


def list_runs(artifact_root: str | None = None) -> list[dict[str, Any]]:
    root = local_artifact_root(artifact_root=artifact_root)
    rows: list[dict[str, Any]] = []
    for index_path in sorted((root / RUN_INDEX_DIR).glob("*.json")):
        item = read_json(index_path)
        state_path = uri_to_path(item["run_dir"]) / "state.json"
        state = read_json(state_path, default={}) or {}
        rows.append({**item, "status": state.get("status"), "current_stage": state.get("current_stage")})
    return rows


def record_data_artifact(ctx: RunContext, name: str, stage: str, path: Path, records: int | None = None) -> None:
    ctx.add_artifact(name=name, stage=stage, path=path, records=records)


def copy_report_bundle(ctx: RunContext, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for rel in ["artifact_index.json", "state.json"]:
        src = ctx.path(rel)
        if src.exists():
            shutil.copy2(src, output_dir / rel)
    for src_dir_name in ["summaries", "logs", "reports"]:
        src_dir = ctx.path(src_dir_name)
        if src_dir.exists():
            dst_dir = output_dir / src_dir_name
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)


def read_artifact_rows(ctx: RunContext, artifact_name: str, limit: int) -> list[dict[str, Any]]:
    artifacts = ctx.state.get("artifacts", [])
    by_name = {item["name"]: item for item in artifacts}
    artifact = by_name.get(artifact_name)
    if artifact is None:
        sample_name = f"{artifact_name}_sample"
        artifact = by_name.get(sample_name)
    if artifact is None:
        raise FileNotFoundError(f"artifact not found: {artifact_name}")
    path = uri_to_path(artifact["uri"])
    if path.suffix == ".jsonl":
        return read_jsonl(path)[:limit]
    payload = read_json(path)
    return [payload] if isinstance(payload, dict) else payload[:limit]
