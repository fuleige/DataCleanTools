# 实现状态记录

本文档用于记录当前代码实现到什么程度、哪些设计已经落地、哪些能力仍未完成。后续每次推进核心能力、改变运行方式或调整技术边界时，都应同步维护本文件。

## 2026-05-11 第一版本地 MVP

### 本版本定位

这一版把设计文档推进成了本地可运行的 CLI MVP，目标是验证第一版离线链路、配置模型、产物结构和命令交互。

当前版本可以作为本地端到端原型使用，但还不是面向百万级图片的生产部署版本。

### 已完成

- 建立 Python 项目骨架、`pyproject.toml` 和命令行入口 `image-labeling`。
- 在 `/data/envs/dataclean-tools` 创建项目专用虚拟环境。
- 安装依赖时采用“去掉代理变量 + 国内镜像”的方式，避免直连 PyPI 下载过慢。
- 已安装并验证 `faiss-cpu 1.13.2`。
- 实现 YAML 配置加载、默认值填充和 Pydantic 校验。
- 实现本地 artifact store、`state.json`、`artifact_index.json` 和结构化 JSONL 日志。
- 实现 CLI 命令：
  - `config validate`
  - `run start`
  - `run resume`
  - `run list`
  - `run status`
  - `run summary`
  - `run logs`
  - `stage run`
  - `stage summary`
  - `artifacts list`
  - `artifacts sample`
  - `report bundle`
  - `export`
  - `api serve`
- 实现本地输入接入：
  - `local_dir`
  - `path_list`
  - 本地 `manifest`
- 实现图片质量检测：
  - 解码失败
  - 格式不支持
  - 图片过小
  - 总像素过少
  - 长宽比异常
  - 文件大小异常
  - 模糊图
  - 低信息量/近纯色图
- 实现缩略图生成。
- 实现本地验证用 embedding provider：`simple_color`。
- 实现 FAISS HNSW 索引构建；当 FAISS 不可用且配置允许时，可退回 numpy 精确检索。
- 实现 top-k 相似图和近重复组产物。
- 实现 MiniBatchKMeans/KMeans 聚类。
- 实现 mock 多模态预标注 provider。
- 实现 OpenAI-compatible 多模态 API provider 的基础接口。
- 实现自动采用决策。
- 实现审核队列生成。
- 实现阶段摘要、样例、错误明细和日志产物。
- 实现最终 JSONL 导出。
- 实现 FastAPI 状态查询、审核队列、图片详情、审核决策写入和导出状态接口。
- 补充本地示例配置 `configs/example-local.yaml`。
- 补充基础测试：
  - 配置加载测试。
  - 本地端到端流水线 smoke test。

### 已验证

- `image-labeling config validate -c configs/example-local.yaml` 可正常校验配置。
- `pytest` 结果为 `2 passed`。
- 临时本地图片目录 smoke 流程已跑通：
  - `run start` 成功运行到 `review_ready`。
  - `run status` 可查看阶段状态。
  - `stage summary` 可查看阶段摘要。
  - `artifacts list` 可查看中间产物。
  - `export --yes` 可生成最终导出文件。

### 当前限制

- 没有接入 Prefect，当前阶段编排由本地 Python 代码顺序执行。
- 没有 PostgreSQL，当前状态和摘要通过本地 JSON/JSONL 文件保存。
- 没有真实对象存储读写；`storage.artifact_store.type=s3` 当前会被拒绝。
- 没有实现真实 DINOv2、SigLIP、CLIP embedding provider。
- 当前 `simple_color` embedding 只适合本地小规模功能验证，不适合作为真实语义特征。
- 没有实现完整 Web 审核前端；当前只有 FastAPI 薄接口。
- 多模态 API provider 只有基础调用接口，尚未做生产级 prompt、预算统计、批量限流和错误治理。
- 没有批量 worker、并发调度、batch checkpoint 和生产级失败恢复。
- 没有真实人工审核 UI 的图片网格、单图详情、近重复组视图和簇视图。
- 没有数据库 migration、部署脚本、服务进程管理。
- 没有百万级性能测试和资源占用评估。

### 下一步建议

- 优先把真实 embedding provider 接口落地，先支持一个可用的 SigLIP 或 CLIP 路径。
- 增加 PostgreSQL 状态层或明确本地 JSON 状态到数据库状态的迁移边界。
- 补最小审核前端，先覆盖审核队列、单图确认/修改、导出状态查看。
- 增加批处理粒度和断点恢复，避免大数据集单阶段失败后整体重跑。
- 增加对象存储适配，至少支持 S3/MinIO artifact store。
- 进行更大规模的本地性能压测，确认 HNSW、聚类、日志和中间产物体积。
