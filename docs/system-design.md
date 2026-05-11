# 总体系统设计

## 1. 系统定位

本系统是一套配置驱动的图像分类数据处理流水线，用于从原始图片集合产出可复现、可追踪、可交付的单标签分类结果。

第一版重点是把百万级图片的离线处理链路跑通：清洗、特征、相似检索、聚类、预标注、自动采用、可视化人工修正和结果导出。非核心的平台能力先延后，避免拖慢落地。

## 2. 核心原则

- 配置优先：质量阈值、模型、聚类、标签体系、自动采用规则和审核抽样都通过 YAML 配置表达。
- 保守自动化：只有质量、置信度、相似组一致性和簇一致性都达标的样本才自动采用。
- 大结果进对象存储：embedding、FAISS index、top-k 相似结果和导出 JSONL 不塞入 PostgreSQL。
- PostgreSQL 管状态：数据库保存任务状态、批次状态、摘要、最终标签和审核修改。
- 可恢复：所有重任务按 batch/shard 执行，失败后只重跑未完成或失败批次。
- 最小审核界面：第一版只做可视化改标签和导出，不做复杂审核系统。

## 3. 模块边界

### API 服务

FastAPI 服务负责：

- 创建任务。
- 查询 run 状态。
- 查询审核队列。
- 保存审核修改。
- 触发导出。

API 不承担大规模模型推理和聚类计算。

### Prefect 流水线

Prefect flow 是离线任务入口，负责阶段编排、重试、状态更新和恢复。重计算由 worker 执行。

默认阶段：

1. `ingest`
2. `quality_check`
3. `thumbnail`
4. `embedding`
5. `vector_index`
6. `similarity`
7. `clustering`
8. `prelabel`
9. `auto_decision`
10. `review_queue`
11. `export`
12. `report`

AI 二审不是默认阶段，只作为后续可选 provider 能力保留。

### Worker

Worker 负责图片解码、质量检测、embedding、FAISS、聚类、预标注和导出。第一版采用静态 worker 配置：

- CPU worker：接入、质量检测、缩略图、导出、报告。
- GPU worker：embedding、本地模型预标注。
- API worker：云端多模态 API 调用，按预算和限流执行。

### 存储

PostgreSQL 保存：

- 数据集、图片、run、stage、batch 状态。
- 质量结果摘要。
- 聚类和近重复摘要。
- 预标注候选摘要。
- 审核队列和审核修改。
- 最终标签。

对象存储保存：

- 原图。
- 输入 manifest。
- 缩略图。
- embedding shard。
- FAISS index。
- 原始向量引用。
- 完整 top-k 相似结果。
- 聚类明细。
- 预标注明细。
- 最终 JSONL 和报告。
- resolved config。

## 4. 数据流

1. 用户提交任务配置和输入 manifest。
2. 系统校验 YAML，生成 resolved config，并归档到对象存储。
3. 接入阶段生成标准 `ImageAsset` 清单。
4. 质量阶段生成 `keep`、`drop`、`quarantine`、`needs_review` 状态。
5. embedding 阶段按 shard 写入向量产物。
6. 索引阶段构建 FAISS HNSW，并保存 id map 和原始向量引用。
7. 相似阶段用 HNSW 召回 top-k，再用原始向量精排。
8. 聚类阶段使用 MiniBatchKMeans/KMeans 生成语义簇。
9. 预标注阶段生成 top-k 标签候选。
10. 自动决策阶段写入自动采用或待审核状态。
11. 审核界面处理待审核和抽样查看样本。
12. 导出阶段生成最终 JSONL 和简要报告。

## 5. 部署基线

第一版采用裸机脚本部署，不默认 Docker Compose 或 Kubernetes。

外部已有服务：

- PostgreSQL。
- S3/MinIO/OSS 等对象存储。
- Prefect Server。

项目内进程：

- FastAPI API。
- React + Vite 审核前端。
- Prefect worker。

配置和密钥通过环境变量注入。明文密钥不写入 YAML，不写入 resolved config。

## 6. 第一版非目标

- 不做复杂权限、登录、SSO、多租户。
- 不做复杂质检工作流。
- 不做自动训练闭环。
- 不做在线实时推理服务。
- 不做在线向量数据库服务。
- 不把审核台扩展成通用标注平台。

