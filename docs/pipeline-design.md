# 流水线设计

## 1. 总览

一次 `pipeline_run` 从输入图片集合开始，最终产出 JSONL 标注结果和报告。每个阶段读取上游稳定产物，写入对象存储和 PostgreSQL 状态。

默认自动阶段运行到 `review_ready`：

1. 数据接入
2. 图片质量检测
3. 缩略图生成
4. embedding 生成
5. FAISS HNSW 索引构建
6. 相似图处理
7. KMeans 类聚类
8. 图片预标注
9. 自动采用决策
10. 审核队列生成
11. 阶段报告生成

人工审核完成后，最终导出由 CLI 手动触发：

1. 结果导出
2. 最终报告生成

## 2. 数据接入

第一版支持：

- JSONL manifest。
- 本地路径清单文件，一行一个图片路径。
- 对象存储前缀扫描。
- 本地目录扫描，主要用于开发和小规模验证。

接入规则：

- 每张图必须有稳定 `image_id`。输入未提供时，用 `dataset_id + uri` 生成 hash。
- 读取对象大小、etag、last_modified 和 content type。
- 重复 `image_id` 或重复 URI 只保留一条，重复项写入接入报告。
- 输入 manifest 归档到对象存储并绑定 `run_id`。
- 本地路径清单和本地目录扫描也必须先规范化为运行 manifest，再进入后续阶段。

## 3. 图片质量检测

质量检测不物理删除原图，只给当前任务生成状态：

- `keep`：进入后续自动流程。
- `needs_review`：可继续处理，但默认进入审核或抽样查看。
- `quarantine`：隔离，不自动采用。
- `drop`：当前任务剔除，不删除原图。

默认检测项：

- 解码失败。
- 格式不支持。
- 图片过小。
- 总像素过少。
- 长宽比异常。
- 文件过小或过大。
- 模糊。
- 低信息量/近纯色。
- EXIF 方向异常。

多条规则命中时默认动作优先级：

```text
drop > quarantine > needs_review > keep
```

## 4. Embedding Provider

Embedding provider 输入图片，输出固定维度向量。第一版抽象支持：

- DINOv2。
- SigLIP。
- CLIP。
- 自定义 PyTorch/ONNX 模型。

第一版一个 run 只启用一个 embedding provider。需要比较多个 embedding 模型时，创建多个 run。

执行规则：

- 按 batch 批推理。
- 默认 L2 normalize。
- 向量按 shard 写对象存储。
- PostgreSQL 只保存产物摘要和 shard 引用。

缓存键：

```text
image_hash + model_version + preprocess_config_hash
```

## 5. FAISS HNSW 索引

第一版默认使用 FAISS `IndexHNSWFlat`。这个选择基于当前假设：机器 CPU/内存资源充足，索引主要服务静态离线 run。

默认参数：

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
```

规则：

- HNSW 只负责召回候选。
- 所有用于近重复、相似图展示和自动采用的相似度，都必须基于原始 embedding 精排。
- `IndexFlatIP` 只作为小规模精确基线和调试方案。
- IVF/PQ 只作为内存受限或更大规模时的备选。
- HNSW 不作为频繁删除/在线增量索引方案。

产物：

- `faiss.index`
- `id_map.parquet`
- `raw_vector_refs.parquet`
- 索引构建报告。

## 6. 相似图处理

相似图处理分两类：

- 近重复检测：用于去重和批量接受。
- 语义相似检索：用于审核界面辅助判断。

近重复默认信号：

- 感知 hash 相似。
- 精排后的 embedding cosine similarity 超过阈值。
- 文件 hash、尺寸和格式辅助判断。

完整 top-k 相似结果写对象存储。PostgreSQL 只保存少量摘要和对象存储 URI。

## 7. 聚类

第一版只采用 KMeans 类算法：

- 百万级默认 `MiniBatchKMeans`。
- 中小规模可用普通 `KMeans`。
- 可按业务字段、数据来源或初步预测标签先分桶，再在桶内聚类。

`num_clusters: auto` 时默认：

```text
ceil(sqrt(valid_image_count))
```

并限制在 `[100, 10000]` 范围内。

离群点不依赖密度聚类，默认使用：

- 到所属中心点的距离。
- top-k 近邻相似度。
- 图片质量状态。

聚类产物：

- `cluster_id`
- 簇大小。
- 簇代表样本。
- 主预测标签占比。
- 簇内预测分布。
- 离群候选。

## 8. 预标注 Provider

预标注 provider 输出候选标签，不直接写最终标签。

支持类型：

- 多模态 API。
- 零样本视觉语言模型。
- 本地分类模型。
- ensemble provider。

第一版默认 provider 顺序：

1. 多模态 API，优先兼容 OpenAI 风格接口。
2. 零样本视觉语言模型。

规则 provider 不作为预标注路径。需要硬编码业务排除条件时，应放在质量过滤、自动采用规则或导出过滤中，不作为预标注 provider。

统一输出：

```json
{
  "image_id": "img_001",
  "provider": "local_classifier",
  "model_version": "v1",
  "prompt_version": null,
  "candidates": [
    {
      "label": "shoes",
      "confidence": 0.96,
      "rank": 1
    }
  ],
  "status": "succeeded",
  "error_code": null
}
```

多模态 API provider 必须配置超时、重试、限流和预算。AI 二审保留为可选 provider 能力，不作为默认主流程。

## 9. 自动采用

自动采用默认要求同时满足：

- 质量状态为 `keep`。
- top-1 置信度达到阈值。
- top-1 和 top-2 margin 达到阈值。
- 近重复组内标签不冲突。
- 所属簇主标签占比达到阈值。
- 标签不属于配置中的高风险标签。

任一条件不满足，则进入审核队列或按配置标记为 `pending`、`rejected`、`invalid`。

## 10. 审核队列

队列优先级默认：

1. 低置信度样本。
2. 近重复组冲突样本。
3. 簇代表样本。
4. 离群候选。
5. 自动采用抽样查看样本。

第一版审核队列只服务可视化界面，不做多人分配、仲裁或复杂质检。

## 11. 中间结果与日志

第一版不依赖 Web 查看中间结果。每个阶段必须产出可供 CLI 查询的摘要、样例和日志。

阶段通用产物：

- `stage_summary.json`：阶段级摘要，包含输入数量、输出数量、跳过数量、失败数量、耗时、主要参数和关键指标。
- `sample.jsonl`：少量代表样例，用于命令行抽样查看，不作为完整结果。
- `errors.jsonl`：失败样本、错误码、错误信息、provider、batch 和重试次数。
- `logs/{stage}/*.jsonl`：结构化运行日志。
- 本地报告包：由 CLI 从已提交的摘要、样例、错误和日志索引生成，用于离线排查，不参与流水线状态判断。

各阶段摘要至少包含：

- 接入：输入数量、去重数量、无效 URI 数量、manifest 归档路径。
- 质量检测：`keep`、`drop`、`quarantine`、`needs_review` 数量，各质量原因 top-k 和样例图。
- 缩略图：成功数量、失败数量、缩略图路径前缀。
- embedding：provider、模型版本、维度、成功数量、失败数量、shard 数量、吞吐。
- HNSW 索引：向量数量、维度、`M`、`ef_construction`、`ef_search`、索引文件 URI。
- 相似图：top-k、近重复组数量、冲突组数量、样例组。
- 聚类：算法、簇数量、簇大小分布、离群数量、代表样本。
- 预标注：provider、成功率、标签分布、置信度分桶、失败原因。
- 自动采用：自动采用数量、待审核数量、规则命中和未通过原因分布。
- 审核队列：队列数量、原因分布、抽样查看数量。
- 导出：最终有效样本数、无效/拒绝样本数、导出文件 URI。

日志字段至少包含：

```json
{
  "timestamp": "2026-05-11T00:00:00Z",
  "run_id": "run_001",
  "stage": "embedding",
  "batch_id": "batch_0001",
  "level": "INFO",
  "provider": "siglip",
  "event": "batch_completed",
  "message": "embedding batch completed",
  "metrics": {
    "items": 512,
    "duration_ms": 1200
  },
  "error_code": null
}
```

CLI 只负责展示摘要和抽样内容。完整明细仍以对象存储产物为准。

## 12. 结果导出

导出由用户人工确认审核完成后手动触发。系统不根据待审核数量、抽样修改率或验收指标自动判定任务完成，只在导出前展示这些摘要并写入导出报告。

默认导出：

- `final_annotations.jsonl`
- `final_annotations.summary.json`
- `quality_report.json`
- `cluster_report.json`
- `model_report.json`
- `review_report.json`

最终 JSONL 每行一张图，必须包含：

- `run_id`
- `image_id`
- 原图 URI。
- 质量状态。
- 聚类摘要。
- 预标注候选摘要。
- 最终标签。
- 最终状态。
- 来源：自动采用或人工修改。
- 配置快照 URI。

## 13. 验收指标

验收指标是配置项，默认值用于报告和阈值调整建议，不默认阻断导出。

默认值：

- 自动采用样本抽样查看 precision 目标：`0.98`。
- 最终交付 precision 目标：`0.99`。
- 每个核心标签最小抽样查看样本数：`50`。
- 自动采用抽样比例：使用 `review.sampling.auto_accept_qa_ratio`，默认 `0.02`。

第一版 precision 只基于抽样查看、人工修改和已确认子集估计，不代表全量真实精度。

## 14. 失败处理

- 每个 batch 记录状态：`pending`、`running`、`succeeded`、`failed`、`skipped`。
- 失败 batch 可重试，不影响已成功 batch。
- Provider 失败记录 provider、输入、错误码、重试次数和最终动作。
- 多模态 API 调用必须支持超时、指数退避、限流和预算中止。
- 恢复时优先读取已提交的 manifest 和 artifact shard，不重复计算已成功批次。
- `run resume` 默认只处理失败或未完成 batch。
- 已成功阶段默认不可重复执行；需要重跑时必须显式使用 `--force`。
- `--force` 重跑成功阶段时，必须记录操作者意图、重跑原因、旧 artifact URI 和新 artifact URI。
- 下游产物依赖被重跑阶段时，必须标记为需要重算或生成新的 run，不能静默复用旧下游结果。
