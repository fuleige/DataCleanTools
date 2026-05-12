# 数据与配置设计

## 1. 配置来源

第一版以 YAML 为主配置来源，Pydantic 负责校验、默认值填充和错误提示。每次运行都会生成 resolved config，并归档到对象存储。

配置加载顺序：

1. 系统默认配置。
2. 业务模板配置。
3. 任务配置。
4. 运行时覆盖参数。

后者覆盖前者。运行开始后，resolved config 不允许修改；配置变化必须创建新的 run。

## 2. YAML 顶层结构

```yaml
project: {}
input: {}
storage: {}
label_schema: {}
quality: {}
thumbnail: {}
embedding: {}
vector_index: {}
clustering: {}
prelabel: {}
auto_decision: {}
review: {}
acceptance: {}
output: {}
runtime:
  sample_limit: 20
  log_level: INFO
  quality_check_executor: process
  quality_check_workers: 8
  quality_check_shard_size: 10000
  quality_check_resume_shards: true
  max_in_memory_rows: 500000
```

## 3. 关键配置

### project 与 input

项目、数据集和运行名称通过 YAML 指定。第一版不在 Web 页面维护项目配置。

```yaml
project:
  id: ecommerce_product_classification
  name: 电商商品分类
  dataset_id: product_202605
  run_name: siglip_hnsw_kmeans_v1

input:
  type: manifest
  manifest_uri: s3://bucket/image-labeling/datasets/product_202605/input/v1/manifest.jsonl
```

`project.id` 表示业务项目，`project.dataset_id` 表示输入图片集合，`project.run_name` 只作为可读名称。系统创建运行时生成唯一 `run_id`，不能依赖 `run_name` 做唯一键。

输入支持四种形态：

- `manifest`：生产推荐方式。
- `path_list`：本地路径清单文件，一行一个图片路径。
- `object_prefix`：扫描对象存储前缀。
- `local_dir`：仅用于开发和小规模验证。

`path_list` 示例：

```yaml
input:
  type: path_list
  path_list_uri: file:///data/images/list.txt
```

对象存储是生产推荐存储。第一版实现可以同时支持 `local` artifact store，用于单机开发和小规模验证；无论输入来源是对象存储还是本地文件，运行时都要生成标准 manifest 并绑定 `run_id`。

输入 manifest 每行一张图，推荐 JSONL 格式：

```json
{"image_id":"img_001","uri":"s3://bucket/images/001.jpg","metadata":{"source":"batch_a"}}
```

字段约定：

- `uri`：必填，支持对象存储 URI 或本地 `file://` URI。
- `image_id`：可选；未提供时由 `dataset_id + uri` 生成稳定 hash。
- `metadata`：可选，用于保存业务字段、来源、已有弱标签等信息。

本地路径清单每行一个图片路径，空行忽略：

```text
/data/images/001.jpg
/data/images/002.jpg
```

### storage

生产推荐对象存储，开发和小规模验证允许本地 artifact store。

```yaml
storage:
  artifact_store:
    type: s3
    root_uri: s3://bucket/image-labeling
```

```yaml
storage:
  artifact_store:
    type: local
    root_uri: file:///data/image-labeling-artifacts
```

### label_schema

标签体系第一版只从 YAML 维护，运行时导入 PostgreSQL 并绑定 run。

```yaml
label_schema:
  source: yaml
  schema_id: product_category_v1
  task_type: single_label
  labels:
    - id: shoes
      name: 鞋
      aliases: [运动鞋, 皮鞋]
      description: 图片主体是鞋类商品
      high_risk: false
  allow_unknown: true
  unknown_label: unknown
```

数据库中的 `label_schemas` 只用于查询、展示和结果追踪，不作为第一版人工维护入口。

### vector_index

```yaml
vector_index:
  backend: faiss
  metric: cosine
  index_type: hnsw_flat
  top_k: 100
  hnsw:
    M: 32
    ef_construction: 200
    ef_search: 128
  rerank:
    enabled: true
    use_original_vectors: true
    final_top_k: 20
  save_index: true
```

### embedding

第一版一个 run 只允许一个 embedding provider。模型对比通过创建多个 run 完成。

```yaml
embedding:
  provider: siglip
  model_name: siglip-so400m
  model_version: siglip-so400m-v1
  batch_size: 128
  normalize: true
```

### prelabel

第一版默认使用多模态 API，接口优先兼容 OpenAI 风格；也支持零样本视觉语言模型。规则 provider 不作为预标注路径。

```yaml
prelabel:
  provider: multimodal_api
  provider_config:
    api_style: openai_compatible
    model: ${MULTIMODAL_API_MODEL}
    timeout_seconds: 60
    max_retries: 3
    rate_limit_qps: 2
    budget:
      max_requests: 100000
      max_cost_usd: 100
```

### clustering

```yaml
clustering:
  enabled: true
  algorithm: minibatch_kmeans
  num_clusters: auto
  batch_size: 10000
  max_iter: 100
  random_state: 42
  outlier:
    enabled: true
    method: distance_to_centroid
    percentile_threshold: 99.0
  duplicate_detection:
    enabled: true
    embedding_threshold: 0.985
    phash_hamming_threshold: 4
  representative:
    samples_per_cluster: 5
```

`num_clusters: auto` 使用 `ceil(sqrt(valid_image_count))`，并限制在 `[100, 10000]`。

### review

```yaml
review:
  enabled: true
  queue:
    include_low_confidence: true
    include_outliers: true
    include_cluster_representatives: true
    include_duplicate_conflicts: true
  sampling:
    auto_accept_qa_ratio: 0.02
    per_cluster_representatives: 3
  batch_actions:
    allow_duplicate_group_accept: true
    allow_cluster_accept: false
    max_batch_size: 500
```

第一版 `review.enabled` 必须为 `true`，不支持关闭审核流程。审核完成由用户根据摘要和审核台状态人工判断，系统不自动定义任务完成。第一版不配置账号、角色、SSO 或 operator。

### acceptance

```yaml
acceptance:
  auto_accept_precision_target: 0.98
  final_precision_target: 0.99
  min_samples_per_core_label: 50
  report_only: true
```

验收指标只用于报告和阈值建议。抽样比例统一由 `review.sampling.auto_accept_qa_ratio` 控制。

## 4. PostgreSQL 逻辑模型

以下是第一版逻辑表，不是最终迁移脚本。

### datasets

保存图片集合。

关键字段：

- `id`
- `name`
- `source_type`
- `source_uri`
- `created_at`

### label_schemas

保存从 YAML 导入的标签体系快照。

关键字段：

- `id`
- `name`
- `version`
- `task_type`
- `labels_json`
- `prompt_text`
- `created_at`

### image_assets

保存图片稳定元数据。

关键字段：

- `id`
- `dataset_id`
- `uri`
- `content_hash`
- `perceptual_hash`
- `width`
- `height`
- `format`
- `file_size`
- `metadata_json`

### pipeline_runs

保存一次运行。

关键字段：

- `id`
- `dataset_id`
- `schema_id`
- `status`
- `config_snapshot_uri`
- `input_manifest_uri`
- `output_prefix`
- `started_at`
- `finished_at`
- `error_json`

`status` 建议枚举：

- `created`
- `running`
- `paused`
- `review_ready`
- `exporting`
- `completed`
- `failed`
- `cancelled`

`paused` 用于 `--until` 正常停在指定阶段后的状态，不表示失败。

### pipeline_stage_runs

保存阶段级状态和指标。

关键字段：

- `id`
- `run_id`
- `stage_name`
- `status`
- `total_items`
- `succeeded_items`
- `failed_items`
- `artifact_uri`
- `metrics_json`
- `started_at`
- `finished_at`

`status` 建议枚举：

- `pending`
- `running`
- `succeeded`
- `failed`
- `skipped`
- `blocked`
- `needs_recompute`

`blocked` 表示上游产物缺失或失败，当前阶段不能执行。`needs_recompute` 表示上游被 `--force` 重跑后，当前阶段依赖的旧产物不再可信。

### pipeline_stage_batches

保存 batch/shard 状态，用于断点续跑。

关键字段：

- `id`
- `run_id`
- `stage_name`
- `batch_index`
- `input_uri`
- `input_start`
- `input_end`
- `status`
- `artifact_uri`
- `checksum`
- `retry_count`
- `error_json`
- `started_at`
- `finished_at`

### artifact_shards

保存对象存储大产物的分片索引。

关键字段：

- `id`
- `run_id`
- `artifact_type`
- `provider_name`
- `shard_index`
- `uri`
- `item_count`
- `checksum`
- `metadata_json`
- `created_at`

`artifact_type` 示例：

- `embeddings`
- `nearest_neighbors`
- `cluster_assignments`
- `annotation_candidates`
- `final_export`

### quality_results

保存图片质量摘要。

关键字段：

- `run_id`
- `image_id`
- `status`
- `reasons_json`
- `width`
- `height`
- `blur_score`
- `entropy_score`
- `dominant_color_ratio`
- `thumbnail_uri`

主键建议：`(run_id, image_id)`。

### embedding_summaries

保存 embedding provider 的汇总信息，不重复记录每个 shard。

关键字段：

- `id`
- `run_id`
- `provider_name`
- `model_version`
- `dimension`
- `normalized`
- `artifact_type`
- `artifact_count`
- `total_items`
- `created_at`

具体 shard 统一由 `artifact_shards` 管理。

### cluster_assignments

保存聚类和相似组摘要。

关键字段：

- `run_id`
- `image_id`
- `cluster_id`
- `duplicate_group_id`
- `is_outlier`
- `cluster_confidence`
- `nearest_summary_json`
- `nearest_artifact_uri`

完整 top-k nearest neighbors 写对象存储，不写入 PostgreSQL。

### annotation_candidates

保存预标注候选摘要。

关键字段：

- `id`
- `run_id`
- `image_id`
- `provider_name`
- `model_version`
- `prompt_version`
- `label`
- `rank`
- `confidence`
- `evidence`
- `raw_output_uri`

### review_tasks

保存审核队列。

关键字段：

- `id`
- `run_id`
- `target_type`
- `target_id`
- `priority`
- `reason`
- `status`
- `context_json`
- `created_at`
- `completed_at`

### review_decisions

保存人工修改或接受动作。第一版批量操作在后端展开成多条单图 decision，不设计 batch decision 表。

关键字段：

- `id`
- `run_id`
- `image_id`
- `review_task_id`
- `decision`
- `final_label`
- `source`
- `comment`
- `created_at`

### final_annotations

保存最终标签。

关键字段：

- `run_id`
- `image_id`
- `final_label`
- `final_status`
- `confidence`
- `source`
- `decision_trace_json`
- `updated_at`

主键建议：`(run_id, image_id)`。

## 5. 对象存储目录

推荐目录：

```text
s3://bucket/image-labeling/
  datasets/{dataset_id}/
    input/{manifest_version}/manifest.jsonl
  runs/{run_id}/
    config/resolved_config.yaml
    ingest/run_input_manifest.jsonl
    summaries/{stage}/stage_summary.json
    summaries/{stage}/sample.jsonl
    summaries/{stage}/errors.jsonl
    logs/pipeline.jsonl
    logs/{stage}/worker-{worker_id}.jsonl
    thumbnails/256/{image_id}.jpg
    quality/quality_results.jsonl
    embeddings/{provider_name}/shard-{shard_id}.parquet
    indexes/{provider_name}/faiss.index
    indexes/{provider_name}/id_map.parquet
    indexes/{provider_name}/raw_vector_refs.parquet
    similarity/nearest_neighbors/shard-{shard_id}.parquet
    clusters/cluster_assignments.jsonl
    prelabel/{provider_name}/candidates.jsonl
    review/review_queue.jsonl
    exports/final_annotations.jsonl
    reports/run_report.json
```

`summaries` 和 `logs` 是第一版排查问题的核心入口。CLI 优先读取 PostgreSQL 中的状态和指标；需要样例、错误明细或完整中间结果时，再读取对象存储中的这些产物。

## 6. 对象存储提交语义

为避免半写产物被当成成功结果，所有大产物按以下顺序提交：

1. 写入临时路径：`runs/{run_id}/_tmp/{stage}/{batch_id}/...`。
2. 计算记录数和 checksum。
3. 将文件移动或复制到正式路径。
4. 写入或更新 `artifact_shards`。
5. 更新 `pipeline_stage_batches.status = succeeded`。
6. stage 汇总所有 batch 成功后，更新 `pipeline_stage_runs.status = succeeded`。

读取方只读取已登记到 `artifact_shards` 且 batch 状态为 `succeeded` 的产物。

## 7. 最终 JSONL 契约

`final_annotations.jsonl` 每行一张图：

```json
{
  "run_id": "run_001",
  "dataset_id": "dataset_product",
  "image_id": "img_001",
  "uri": "s3://bucket/path/img_001.jpg",
  "quality": {
    "status": "keep",
    "reasons": []
  },
  "cluster": {
    "cluster_id": "cluster_12",
    "duplicate_group_id": null,
    "is_outlier": false
  },
  "prediction": {
    "label": "shoes",
    "confidence": 0.97,
    "provider": "ensemble_v1",
    "model_version": "siglip-v1",
    "prompt_version": "taxonomy-v1",
    "candidates": []
  },
  "review": {
    "status": "auto_accepted",
    "final_label": "shoes",
    "source": "ai",
    "completion_confirmed_by_user": true,
    "updated_at": "2026-05-11T00:00:00Z"
  },
  "trace": {
    "config_snapshot_uri": "s3://bucket/image-labeling/runs/run_001/config/resolved_config.yaml",
    "decision_reasons": ["quality_keep", "confidence_pass", "cluster_consistency_pass"]
  }
}
```

## 8. 索引建议

建议 PostgreSQL 索引：

- `pipeline_stage_batches(run_id, stage_name, status)`
- `artifact_shards(run_id, artifact_type)`
- `quality_results(run_id, status)`
- `cluster_assignments(run_id, cluster_id)`
- `cluster_assignments(run_id, duplicate_group_id)`
- `annotation_candidates(run_id, image_id)`
- `review_tasks(run_id, status, priority)`
- `final_annotations(run_id, final_status)`
- `final_annotations(run_id, final_label)`
