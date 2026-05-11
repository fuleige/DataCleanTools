from image_labeling.config import load_config


def test_example_config_loads() -> None:
    config = load_config("configs/example-local.yaml")
    assert config.project.id == "local_image_classification"
    assert config.review.enabled is True
    assert config.embedding.provider == "simple_color"
