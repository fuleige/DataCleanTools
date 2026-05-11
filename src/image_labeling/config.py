from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return os.environ.get(key, match.group(0))

        return pattern.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectConfig(StrictModel):
    id: str
    name: str = ""
    dataset_id: str
    run_name: str = "default"


class InputConfig(StrictModel):
    type: Literal["manifest", "path_list", "object_prefix", "local_dir"]
    manifest_uri: str | None = None
    path_list_uri: str | None = None
    object_prefix_uri: str | None = None
    local_dir: str | None = None
    recursive: bool = True
    extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]
    )

    @model_validator(mode="after")
    def required_source(self) -> "InputConfig":
        required = {
            "manifest": self.manifest_uri,
            "path_list": self.path_list_uri,
            "object_prefix": self.object_prefix_uri,
            "local_dir": self.local_dir,
        }
        if not required[self.type]:
            raise ValueError(f"input.{self.type} requires its matching URI/path field")
        return self


class ArtifactStoreConfig(StrictModel):
    type: Literal["local", "s3"] = "local"
    root_uri: str = "file:///data/dataclean-artifacts"

    @model_validator(mode="after")
    def local_only_for_now(self) -> "ArtifactStoreConfig":
        if self.type != "local":
            raise ValueError("Current implementation supports local artifact_store only")
        return self


class StorageConfig(StrictModel):
    artifact_store: ArtifactStoreConfig = Field(default_factory=ArtifactStoreConfig)


class LabelConfig(StrictModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    high_risk: bool = False


class LabelSchemaConfig(StrictModel):
    source: Literal["yaml"] = "yaml"
    schema_id: str
    task_type: Literal["single_label"] = "single_label"
    labels: list[LabelConfig]
    allow_unknown: bool = True
    unknown_label: str = "unknown"

    @model_validator(mode="after")
    def validate_labels(self) -> "LabelSchemaConfig":
        ids = [label.id for label in self.labels]
        if len(ids) != len(set(ids)):
            raise ValueError("label_schema.labels contains duplicate label ids")
        if self.allow_unknown and self.unknown_label not in ids:
            self.labels.append(LabelConfig(id=self.unknown_label, name="Unknown"))
        return self


class BlurQualityConfig(StrictModel):
    enabled: bool = True
    edge_variance_threshold: float = 18.0
    action: Literal["drop", "quarantine", "needs_review", "keep"] = "needs_review"


class LowInfoQualityConfig(StrictModel):
    enabled: bool = True
    channel_stddev_threshold: float = 4.0
    action: Literal["drop", "quarantine", "needs_review", "keep"] = "quarantine"


class QualityConfig(StrictModel):
    min_width: int = 64
    min_height: int = 64
    min_pixels: int = 4096
    max_aspect_ratio: float = 6.0
    min_file_size_bytes: int = 256
    max_file_size_bytes: int | None = None
    allowed_formats: list[str] = Field(default_factory=lambda: ["JPEG", "PNG", "WEBP", "BMP", "TIFF"])
    blur: BlurQualityConfig = Field(default_factory=BlurQualityConfig)
    low_information: LowInfoQualityConfig = Field(default_factory=LowInfoQualityConfig)

    @field_validator("allowed_formats")
    @classmethod
    def normalize_formats(cls, value: list[str]) -> list[str]:
        return [item.upper() for item in value]


class ThumbnailConfig(StrictModel):
    size: int = 256
    quality: int = 85


class EmbeddingConfig(StrictModel):
    provider: str = "simple_color"
    model_name: str = "simple-color-histogram"
    model_version: str = "simple-color-histogram-v1"
    batch_size: int = 128
    normalize: bool = True
    include_quality_statuses: list[str] = Field(default_factory=lambda: ["keep", "needs_review"])


class HnswConfig(StrictModel):
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 128


class RerankConfig(StrictModel):
    enabled: bool = True
    use_original_vectors: bool = True
    final_top_k: int = 20


class VectorIndexConfig(StrictModel):
    backend: Literal["faiss"] = "faiss"
    metric: Literal["cosine"] = "cosine"
    index_type: Literal["hnsw_flat"] = "hnsw_flat"
    top_k: int = 100
    hnsw: HnswConfig = Field(default_factory=HnswConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    save_index: bool = True
    allow_exact_fallback: bool = True


class DuplicateDetectionConfig(StrictModel):
    enabled: bool = True
    embedding_threshold: float = 0.985
    phash_hamming_threshold: int = 4


class OutlierConfig(StrictModel):
    enabled: bool = True
    method: Literal["distance_to_centroid"] = "distance_to_centroid"
    percentile_threshold: float = 99.0


class RepresentativeConfig(StrictModel):
    samples_per_cluster: int = 5


class ClusteringConfig(StrictModel):
    enabled: bool = True
    algorithm: Literal["minibatch_kmeans", "kmeans"] = "minibatch_kmeans"
    num_clusters: int | Literal["auto"] = "auto"
    batch_size: int = 10000
    max_iter: int = 100
    random_state: int = 42
    outlier: OutlierConfig = Field(default_factory=OutlierConfig)
    duplicate_detection: DuplicateDetectionConfig = Field(default_factory=DuplicateDetectionConfig)
    representative: RepresentativeConfig = Field(default_factory=RepresentativeConfig)


class PrelabelConfig(StrictModel):
    provider: str = "multimodal_api"
    provider_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "api_style": "openai_compatible",
            "model": "${MULTIMODAL_API_MODEL}",
            "timeout_seconds": 60,
            "max_retries": 3,
            "rate_limit_qps": 2,
            "mock": True,
        }
    )
    top_k: int = 3
    prompt_version: str | None = None


class AutoDecisionConfig(StrictModel):
    min_confidence: float = 0.9
    min_margin: float = 0.15
    min_cluster_main_label_ratio: float = 0.8
    require_cluster_consistency: bool = True
    block_high_risk_labels: bool = True


class ReviewQueueConfig(StrictModel):
    include_low_confidence: bool = True
    include_outliers: bool = True
    include_cluster_representatives: bool = True
    include_duplicate_conflicts: bool = True


class ReviewSamplingConfig(StrictModel):
    auto_accept_qa_ratio: float = 0.02
    per_cluster_representatives: int = 3


class ReviewBatchActionsConfig(StrictModel):
    allow_duplicate_group_accept: bool = True
    allow_cluster_accept: bool = False
    max_batch_size: int = 500


class ReviewConfig(StrictModel):
    enabled: bool = True
    queue: ReviewQueueConfig = Field(default_factory=ReviewQueueConfig)
    sampling: ReviewSamplingConfig = Field(default_factory=ReviewSamplingConfig)
    batch_actions: ReviewBatchActionsConfig = Field(default_factory=ReviewBatchActionsConfig)

    @model_validator(mode="after")
    def must_be_enabled(self) -> "ReviewConfig":
        if not self.enabled:
            raise ValueError("First version requires review.enabled=true")
        return self


class AcceptanceConfig(StrictModel):
    auto_accept_precision_target: float = 0.98
    final_precision_target: float = 0.99
    min_samples_per_core_label: int = 50
    report_only: bool = True


class OutputConfig(StrictModel):
    valid_only_file: str = "valid_annotations.jsonl"
    full_file: str = "final_annotations.jsonl"


class RuntimeConfig(StrictModel):
    sample_limit: int = 20
    log_level: str = "INFO"


class PipelineConfig(StrictModel):
    project: ProjectConfig
    input: InputConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    label_schema: LabelSchemaConfig
    quality: QualityConfig = Field(default_factory=QualityConfig)
    thumbnail: ThumbnailConfig = Field(default_factory=ThumbnailConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    prelabel: PrelabelConfig = Field(default_factory=PrelabelConfig)
    auto_decision: AutoDecisionConfig = Field(default_factory=AutoDecisionConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return PipelineConfig.model_validate(_expand_env(raw))


def dump_config_yaml(config: PipelineConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, allow_unicode=True, sort_keys=False)
