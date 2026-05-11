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

- 查询 run 状态和阶段摘要。
- 查询审核队列。
- 查询图片详情、相似图摘要和簇摘要。
- 保存审核修改。
- 查询导出状态和结果路径。

API 不承担大规模模型推理和聚类计算。

### Prefect 流水线

Prefect flow 是离线任务入口，负责阶段编排、重试、状态更新和恢复。重计算由 worker 执行。

默认自动阶段运行到 `review_ready`：

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
11. `report`

最终导出由人工确认审核完成后，通过 CLI 手动触发：

1. `export`
2. `final_report`

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

1. 用户提交任务配置和输入源，输入源可以是 manifest、路径清单、本地目录或对象存储前缀。
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
12. 用户根据审核摘要人工判断任务完成后，通过 CLI 触发导出。
13. 导出阶段生成最终 JSONL 和简要报告。

## 5. 运行与交互方式

第一版采用“YAML 配置指定项目，CLI 控制运行，Web UI 查看状态并完成人工审核”的方式。

### 项目指定

业务项目、数据集和单次运行分开表达：

- `project`：业务项目，例如商品分类、缺陷分类或内容审核分类。
- `dataset`：一批图片集合，例如某月商品图、某条产线缺陷图。
- `run`：一次具体流水线运行。更换模型、阈值、标签体系或输入集合时，应创建新的 run。

项目主要通过 YAML 指定。命令行负责选择配置文件、启动任务、查询状态、恢复失败任务和触发导出。第一版不在 Web UI 暴露配置上传、配置编辑或启动 run 的入口。

### CLI 入口

第一版提供以下命令语义：

```bash
image-labeling config validate -c configs/product_classification.yaml
image-labeling run start -c configs/product_classification.yaml
image-labeling run start -c configs/product_classification.yaml --until quality_check
image-labeling run start -c configs/product_classification.yaml --from-stage embedding --until clustering
image-labeling stage run --run-id run_20260511_001 --stage prelabel
image-labeling run status --run-id run_20260511_001
image-labeling run summary --run-id run_20260511_001
image-labeling stage summary --run-id run_20260511_001 --stage quality_check
image-labeling run logs --run-id run_20260511_001 --follow
image-labeling artifacts list --run-id run_20260511_001
image-labeling artifacts sample --run-id run_20260511_001 --artifact quality_results --limit 20
image-labeling report bundle --run-id run_20260511_001 -o reports/run_20260511_001
image-labeling run resume --run-id run_20260511_001
image-labeling export --run-id run_20260511_001
```

`run start` 会校验配置、生成 resolved config、创建 `pipeline_run`、提交 Prefect flow，并返回：

```text
run_id: run_20260511_001
review_url: http://localhost:5173/runs/run_20260511_001/review
```

允许少量运行时覆盖参数，例如 `run_name`、`output_prefix` 或只执行指定阶段；所有覆盖后的结果必须写入 resolved config，保证 run 可追踪。

### 自动运行与手动控制

第一版默认是一键式自动流水线，同时保留阶段级手动控制。

默认执行：

```bash
image-labeling run start -c configs/product_classification.yaml
```

默认从接入开始自动执行到 `review_ready`，包括质量检测、缩略图、embedding、HNSW、相似图、聚类、预标注、自动采用和审核队列生成。默认不自动执行最终导出。

人工审核完成后，用户手动触发导出：

```bash
image-labeling export --run-id run_20260511_001
```

审核完成由用户人工判断。系统只提供待审核数量、抽样查看修改率、每标签分布、簇摘要和质量摘要，不通过程序自动判定任务是否完成。导出命令应展示这些摘要并记录本次导出是人工确认后的结果；即使仍存在待审核样本，也由用户决定是否继续导出。

手动控制用于开发调试、阶段验证、成本控制和失败排查：

- `--until <stage>`：从头运行到指定阶段后停止。
- `--from-stage <stage>`：从指定阶段继续执行，要求上游产物已成功提交。
- `stage run`：只运行单个阶段，主要用于调试和小规模验证。
- `run resume`：恢复失败或未完成的 batch，默认不重复计算已成功 batch。
- `--force`：显式重跑已成功阶段。第一版必须提示会覆盖或生成新的下游产物，并要求操作者确认。

运行控制规则：

- 常规业务使用一条 `run start` 跑到 `review_ready`。
- 失败恢复优先使用 `run resume`。
- 配置、标签体系、模型或输入集合变化时创建新 run，不在旧 run 上修改 resolved config。
- 同一 run 内重跑成功阶段必须显式使用 `--force`，并记录到 run 日志和 stage summary。
- 最终导出必须手动触发，避免审核未完成时误生成交付结果。

### 中间结果与日志查看

第一版必须支持非 Web 的中间结果查看能力，主要通过 CLI 和文件化产物完成。

CLI 查看能力：

- `run status`：展示 run 当前状态、当前阶段、成功/失败 batch 数和最近错误。
- `run summary`：展示全局摘要，包括总图片数、质量分布、embedding 进度、聚类数量、预标注成功率、自动采用率和待审核数量。
- `stage summary`：展示单阶段摘要和关键样例，例如质量失败原因、近重复冲突组、簇代表样本、低置信度标签分布。
- `run logs`：查看结构化日志，支持按阶段、level、provider、batch 过滤，支持 `--follow` 追踪运行中日志。
- `artifacts list`：列出当前 run 已提交的中间产物 URI、记录数、checksum 和创建时间。
- `artifacts sample`：抽样查看中间产物内容，避免在终端直接输出完整百万级结果。
- `report bundle`：把阶段摘要、错误样例、抽样结果、artifact 索引和日志索引下载或渲染到本地目录，便于用 `less`、`jq` 或编辑器查看。

中间结果展示原则：

- 终端只展示摘要、路径和少量样例。
- 完整中间结果写入对象存储，由 `artifact_shards` 记录。
- 日志使用 JSONL 结构化格式，同时保留人可读 message。
- embedding、FAISS index、完整 top-k 相似结果等大文件不直接打印，只展示元数据和对象存储 URI。
- 每个阶段结束后必须生成 `stage_summary.json`，失败时也要尽量生成失败摘要。

本地报告包推荐结构：

```text
reports/{run_id}/
  overview.md
  artifact_index.json
  stages/{stage}/stage_summary.json
  stages/{stage}/sample.jsonl
  stages/{stage}/errors.jsonl
  logs/pipeline.jsonl
  logs/{stage}.jsonl
```

### API 与 UI

FastAPI 是审核 UI 的后端，提供状态查询、队列查询、图片详情、标签修改、批量操作、报告查询和导出结果查询接口。

第一版任务配置和运行控制都以 CLI 为主。Web UI 不负责创建任务，只承担只读状态查看、可视化审核和结果路径查看。

审核 UI 启动方式：

```bash
image-labeling api serve
image-labeling ui serve
```

用户打开审核台后完成：

- 查看任务列表和 run 状态。
- 查看阶段进度、质量摘要、聚类摘要、预标注摘要和自动采用摘要。
- 筛选待审核、低置信度、近重复冲突、离群候选和抽样查看样本。
- 接受推荐标签、修改标签、标记无效或拒绝样本。
- 批量接受近重复组。
- 查看报告、导出状态和最终结果路径。

### Run 生命周期

推荐状态流转：

```text
created -> running -> review_ready -> exporting -> completed
created -> running -> paused -> running
```

异常状态：

```text
failed -> running
cancelled
```

流水线可以先自动跑到 `review_ready`。用户根据审核台和 CLI 摘要人工判断任务完成后，通过 CLI 触发最终导出。Web UI 只展示导出状态和结果路径。

配置变化不修改已有 run。配置、标签体系、模型或输入集合发生变化时，创建新的 run。

## 6. 部署基线

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

## 7. 第一版非目标

- 不做复杂权限、登录、SSO、多租户。
- 不做复杂质检工作流。
- 不做自动训练闭环。
- 不做在线实时推理服务。
- 不做在线向量数据库服务。
- 不把审核台扩展成通用标注平台。
