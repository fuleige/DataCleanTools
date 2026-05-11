# 流水线设计

## 1. 总览

一次 `pipeline_run` 从输入图片集合开始，最终产出 JSONL 标注结果和报告。每个阶段读取上游稳定产物，写入对象存储和 PostgreSQL 状态。

默认阶段：

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
11. 结果导出
12. 报告生成

## 2. 数据接入

第一版支持：

- JSONL manifest。
- 对象存储前缀扫描。
- 本地目录，仅用于开发和小规模验证。

接入规则：

- 每张图必须有稳定 `image_id`。输入未提供时，用 `dataset_id + uri` 生成 hash。
- 读取对象大小、etag、last_modified 和 content type。
- 重复 `image_id` 或重复 URI 只保留一条，重复项写入接入报告。
- 输入 manifest 归档到对象存储并绑定 `run_id`。

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

- 本地分类模型。
- 零样本视觉语言模型。
- 多模态 API。
- 规则 provider。
- ensemble provider。

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

## 11. 结果导出

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

## 12. 验收指标

验收指标是配置项，默认值用于报告和阈值调整建议，不默认阻断导出。

默认值：

- 自动采用样本抽样查看 precision 目标：`0.98`。
- 最终交付 precision 目标：`0.99`。
- 每个核心标签最小抽样查看样本数：`50`。
- 自动采用抽样比例：使用 `review.sampling.auto_accept_qa_ratio`，默认 `0.02`。

第一版 precision 只基于抽样查看、人工修改和已确认子集估计，不代表全量真实精度。

## 13. 失败处理

- 每个 batch 记录状态：`pending`、`running`、`succeeded`、`failed`、`skipped`。
- 失败 batch 可重试，不影响已成功 batch。
- Provider 失败记录 provider、输入、错误码、重试次数和最终动作。
- 多模态 API 调用必须支持超时、指数退避、限流和预算中止。
- 恢复时优先读取已提交的 manifest 和 artifact shard，不重复计算已成功批次。

