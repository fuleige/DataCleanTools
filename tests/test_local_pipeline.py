from pathlib import Path

from PIL import Image

from image_labeling.config import load_config
from image_labeling.jsonio import read_json, read_jsonl
from image_labeling.pipeline import export_run, resume_run, run_single_stage, start_run


def test_local_pipeline_smoke(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for idx, color in enumerate([(255, 0, 0), (250, 10, 10), (0, 255, 0), (0, 0, 255)], start=1):
        Image.new("RGB", (96, 96), color=color).save(image_dir / f"{idx}.jpg")

    config = load_config("configs/example-local.yaml")
    assert config.runtime.quality_check_executor == "process"
    assert config.runtime.quality_check_workers == 8
    assert config.runtime.quality_check_shard_size == 10000
    assert config.runtime.quality_check_resume_shards is True
    assert config.runtime.max_in_memory_rows == 500000
    config.input.local_dir = str(image_dir)
    config.storage.artifact_store.root_uri = (tmp_path / "artifacts").as_uri()
    config.quality.low_information.enabled = False
    config.quality.blur.enabled = False
    config.runtime.quality_check_workers = 2
    config.runtime.quality_check_shard_size = 2
    config.review.sampling.auto_accept_qa_ratio = 1.0

    ctx = start_run(config)
    assert ctx.state["status"] == "review_ready"
    quality_metrics = ctx.state["stages"]["quality_check"]["metrics_json"]
    assert quality_metrics["executor"] == "process"
    assert quality_metrics["effective_executor"] == "process"
    assert quality_metrics["worker_count"] == 2
    assert quality_metrics["shard_count"] == 2
    assert ctx.path("data", "manifest.jsonl").exists()
    assert ctx.path("data", "quality_check_shards", "shard_000000.jsonl").exists()
    assert ctx.path("checkpoints", "quality_check_shards.jsonl").exists()
    assert ctx.path("review", "review_queue.jsonl").exists()

    exported = export_run(ctx.run_id, artifact_root=(tmp_path / "artifacts").as_uri())
    assert exported.state["status"] == "completed"
    assert exported.path("exports", "final_annotations.jsonl").exists()


def test_quality_check_reuses_completed_shards(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for idx, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (120, 120, 120), (30, 30, 30)], start=1):
        Image.new("RGB", (96, 96), color=color).save(image_dir / f"{idx}.jpg")

    config = load_config("configs/example-local.yaml")
    config.input.local_dir = str(image_dir)
    config.storage.artifact_store.root_uri = (tmp_path / "artifacts").as_uri()
    config.quality.low_information.enabled = False
    config.quality.blur.enabled = False
    config.runtime.quality_check_workers = 2
    config.runtime.quality_check_shard_size = 2

    ctx = start_run(config, until="quality_check")
    checkpoint_path = ctx.path("checkpoints", "quality_check_shards.jsonl")
    checkpoint_rows = read_jsonl(checkpoint_path)
    assert sum(1 for row in checkpoint_rows if row["status"] == "succeeded") == 3
    assert checkpoint_rows[0]["implementation_version"] == "quality_check_sharded_v1"

    ctx.state["stages"]["quality_check"]["status"] = "failed"
    ctx.save_state()
    ctx.path("data", "quality_results.jsonl").unlink()
    ctx.path("data", "quality_check_shards", "shard_000001.jsonl").unlink()
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write('{"shard_id":')

    resumed = run_single_stage(ctx.run_id, "quality_check", artifact_root=(tmp_path / "artifacts").as_uri())
    summary = read_json(resumed.summary_path("quality_check"))
    assert summary["output_count"] == 5
    assert summary["reused_shards"] == 2
    assert summary["implementation_version"] == "quality_check_sharded_v1"
    assert resumed.path("data", "quality_results.jsonl").exists()


def test_resume_respects_start_until_boundary(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for idx, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)], start=1):
        Image.new("RGB", (96, 96), color=color).save(image_dir / f"{idx}.jpg")

    config = load_config("configs/example-local.yaml")
    config.input.local_dir = str(image_dir)
    config.storage.artifact_store.root_uri = (tmp_path / "artifacts").as_uri()
    config.quality.low_information.enabled = False
    config.quality.blur.enabled = False
    config.runtime.quality_check_shard_size = 2

    ctx = start_run(config, until="quality_check")
    assert ctx.state["execution"]["until"] == "quality_check"
    ctx.state["stages"]["quality_check"]["status"] = "failed"
    ctx.state["status"] = "failed"
    ctx.state["finished_at"] = "2026-01-01T00:00:00Z"
    ctx.state["error_json"] = {"stage": "quality_check", "error_code": "SyntheticFailure", "message": "old error"}
    ctx.path("data", "quality_results.jsonl").unlink()
    ctx.save_state()

    resumed = resume_run(ctx.run_id, artifact_root=(tmp_path / "artifacts").as_uri())
    assert resumed.state["status"] == "paused"
    assert resumed.state["current_stage"] == "quality_check"
    assert resumed.state["finished_at"] != "2026-01-01T00:00:00Z"
    assert resumed.state["error_json"] is None
    assert "thumbnail" not in resumed.state["stages"]
