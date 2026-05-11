from pathlib import Path

from PIL import Image

from image_labeling.config import load_config
from image_labeling.pipeline import export_run, start_run


def test_local_pipeline_smoke(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for idx, color in enumerate([(255, 0, 0), (250, 10, 10), (0, 255, 0), (0, 0, 255)], start=1):
        Image.new("RGB", (96, 96), color=color).save(image_dir / f"{idx}.jpg")

    config = load_config("configs/example-local.yaml")
    config.input.local_dir = str(image_dir)
    config.storage.artifact_store.root_uri = (tmp_path / "artifacts").as_uri()
    config.quality.low_information.enabled = False
    config.quality.blur.enabled = False
    config.review.sampling.auto_accept_qa_ratio = 1.0

    ctx = start_run(config)
    assert ctx.state["status"] == "review_ready"
    assert ctx.path("data", "manifest.jsonl").exists()
    assert ctx.path("review", "review_queue.jsonl").exists()

    exported = export_run(ctx.run_id, artifact_root=(tmp_path / "artifacts").as_uri())
    assert exported.state["status"] == "completed"
    assert exported.path("exports", "final_annotations.jsonl").exists()
