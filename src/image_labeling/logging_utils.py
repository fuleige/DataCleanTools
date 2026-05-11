from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .jsonio import append_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class PipelineLogger:
    run_id: str
    run_dir: Path

    def log(
        self,
        stage: str,
        event: str,
        message: str,
        *,
        level: str = "INFO",
        batch_id: str | None = None,
        provider: str | None = None,
        metrics: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> None:
        row = {
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "stage": stage,
            "batch_id": batch_id,
            "level": level,
            "provider": provider,
            "event": event,
            "message": message,
            "metrics": metrics or {},
            "error_code": error_code,
        }
        append_jsonl(self.run_dir / "logs" / "pipeline.jsonl", row)
        append_jsonl(self.run_dir / "logs" / stage / "worker-0.jsonl", row)


def duration_ms(start_iso: str, end_iso: str) -> int:
    start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    return int((end - start).total_seconds() * 1000)
