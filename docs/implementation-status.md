# 实现状态记录

本文档用于记录当前代码实现到什么程度、哪些设计已经落地、哪些能力仍未完成。后续每次推进核心能力、改变运行方式或调整技术边界时，都应同步维护本文件。

## 2026-05-14 示例配置注释完善

### 已完成

- 补全 `configs/example-local.yaml` 的配置说明注释，覆盖输入源、产物存储、标签体系、质量检测、缩略图、embedding、向量索引、聚类、预标注、自动决策、审核、验收、导出和运行时参数。
- 在注释中补充主要枚举候选值、建议范围、阈值含义、动作优先级和工程使用建议。
- 将部分原本依赖代码默认值的常用字段显式写入示例配置，例如 `input.extensions`、`input.compute_content_hash`、`embedding.include_quality_statuses` 和 `prelabel.prompt_version`，方便复制后直接修改。

### 已验证

- `/data/envs/dataclean-tools/bin/image-labeling config validate -c configs/example-local.yaml` 通过。

## 2026-05-14 百万级 embedding 流式写入

### 已完成

- `embedding` 阶段改为两遍流式处理 `quality_results.jsonl`：先统计可嵌入候选，再按 `embedding.batch_size` 批量编码。
- embedding 向量改为通过 numpy memmap 直接写入 `.npy`，`ids.json` 和 `embedding_refs.jsonl` 也改为流式写入，避免一次性加载百万级质量结果。
- `embedding` 阶段增加候选扫描和编码进度上报，可直接在跳过 `thumbnail` 的情况下从已完成 `quality_check` 的 run 上执行。
- 补充回归测试，验证 `runtime.max_in_memory_rows` 小于质量结果行数时 embedding 仍可运行。

### 已验证

- `/data/envs/dataclean-tools/bin/python -m pytest` 通过，结果为 `5 passed`。
- `/data/envs/dataclean-tools/bin/image-labeling config validate -c configs/example-local.yaml` 通过。
- `git diff --check` 通过。

## 2026-05-19 提交前检查

### 已验证

- 当前检查未运行 CUDA/GPU 程序；Python 验证命令均显式设置 `CUDA_VISIBLE_DEVICES=`，避免影响机器上已有 GPU 任务。
- `git diff --check` 通过。
- `tools/build_faiss_gpu_index.py`、`tools/extract_siglip2_embeddings.py`、`tools/benchmark_siglip2_embedding.py` 通过 `py_compile` 语法检查。
- `/data/envs/dataclean-tools/bin/image-labeling config validate -c configs/example-local.yaml` 通过。
- `/data/envs/dataclean-tools/bin/python -m pytest` 通过，结果为 `5 passed`，存在 3 个 faiss/swig 相关既有 warning。

## 2026-05-14 SigLIP2 深度学习 embedding provider

### 已完成

- 新增 `embedding.provider=siglip2`，通过 Hugging Face `AutoImageProcessor` 和 `AutoModel` 加载本地或远程 SigLIP/SigLIP2 模型。
- `EmbeddingConfig` 增加 `provider_config`，支持配置 `model_path`、`device`、`dtype`、`local_files_only` 和 `trust_remote_code`。
- 新增 `deep` 可选依赖组，包含 `torch`、`transformers` 和 `safetensors`。
- `configs/example-local.yaml` 增加 SigLIP2 provider 的配置示例、字段含义和建议批大小。
- `tools/benchmark_siglip2_embedding.py` 支持 SigLIP/SigLIP2 多进程多 GPU 压测，可选择 full model 或 vision-only model，并可重复指定同一 GPU 以测试每卡多 worker 并发。
- `tools/extract_siglip2_embeddings.py` 支持从 clean manifest 执行 SigLIP2 vision-only 多进程多 GPU 向量提取，直接写出兼容的 `embeddings.npy`、`ids.json` 和 `embedding_refs.jsonl`。
- `tools/build_faiss_gpu_index.py` 支持使用 FAISS GPU 构建 IVF-Flat 索引，默认只建索引并保存 `faiss_ivf_flat.index` 和 `index_metadata.json`；可选 `--write-candidates` 时再按 batch 搜索并流式写候选 JSONL。
- `pyproject.toml` 增加 `faiss-gpu` 可选依赖组，当前 `/data/envs/dl` 已安装并验证 `faiss-gpu-cu12==1.13.2`。

### 已验证

- `/data/envs/dl` 已验证可用：`torch 2.11.0+cu126`、`transformers 5.8.0`、`safetensors`，CUDA 可见 4 张 NVIDIA GeForce RTX 4090 D。
- 本地 `/data/imgsrch/models/siglip2-base` 最小推理通过，8 张图片输出归一化 float32 向量，shape 为 `[8, 768]`。
- 使用 clean manifest 抽样完成 4 卡性能基准；full model 单进程/卡在线程池限制为 1 且 batch/GPU=512 时，8192 张样本吞吐约 1382 img/s。
- vision-only 多进程压测中，32 workers 总数、每卡 8 进程、每 worker batch=128、CPU 数值线程池限制为 1 时，1048576 张样本吞吐约 8268 img/s，不含模型加载的 2591127 张 clean 图片预计纯提取约 5.2 分钟；压测期间 4 张 RTX 4090 D 均可达到 100% GPU utilization。
- 已对比 batch=256 和 48 workers：batch=256 未提升吞吐，48 workers 相比 32 workers 没有明确收益且单 worker 吞吐明显下降。当前推荐压测配置为 32 workers、每卡 8 进程、batch=128。
- 已对 MEP-3M clean manifest 执行全量 SigLIP2 embedding 提取：2591127 张图片全部成功，失败 0，输出向量 shape `(2591127, 768)`，保存为 float32 归一化向量；32 workers、每卡 8 进程、batch=128 的端到端耗时约 443 秒，包含向量和元数据写入，吞吐约 5846 img/s。
- FAISS GPU 环境已验证：`/data/envs/dl` 中 `faiss-gpu-cu12==1.13.2` 可见 4 张 GPU，提供 `StandardGpuResources`、`index_cpu_to_gpu_multiple_py` 和 `index_gpu_to_cpu`。
- 已用 4096 条真实 embedding 做 GPU IVF-Flat 烟测：单 GPU、`nlist=128`、`nprobe=16` 可完成训练、添加、索引保存和候选 JSONL 流式输出。
- 注意：当前正式 `embedding` stage 已支持流式写入，但尚未把多进程多 GPU 调度和 vision-only 快速路径完整合并到 `stage run embedding`；本次全量提取通过 `tools/extract_siglip2_embeddings.py` 落盘，并已手动同步 run state、summary 和 artifact index。
- 注意：正式 `vector_index` stage 当前仍是 CPU HNSW 路径；GPU IVF-Flat 索引构建先通过 `tools/build_faiss_gpu_index.py` 执行。后续应把 GPU IVF 配置合并进 `VectorIndexConfig` 和 `run_vector_index`，并将“建索引”和“候选检索”拆成可恢复的两个子阶段。
- 如需把深度学习栈同步到 `/data/envs/dataclean-tools`，安装命令必须继续使用 `env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy -u ALL_PROXY -u all_proxy ...`，非 PyTorch 包优先走国内镜像；PyTorch CUDA wheel 仍主要依赖官方 CUDA wheel 源或复用已有 `/data/envs/dl`。

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

## 2026-05-12 CLI 运行进度可视化

### 已完成

- `run start`、`run resume`、`stage run` 增加默认开启的 Rich 实时进度显示，可通过 `--no-progress` 关闭。
- `ingest` 阶段在递归扫描本地目录和构建 manifest 时上报进度。
- `quality_check` 阶段按已处理图片数上报进度。
- `state.json` 中的阶段状态增加 `processed_items`，`run status` 表格增加 `processed` 列，便于另一个终端查看当前处理到哪里。
- 本地目录输入在已是绝对路径时直接生成 `file://` URI，避免百万级文件逐个 `resolve()` 带来的额外开销。
- `input.compute_content_hash` 默认关闭；需要内容哈希时可显式开启，避免初筛任务在质量检查前额外完整读取所有图片文件。
- `ingest` 写 manifest、`quality_check` 写 quality results 改为流式写 JSONL，降低百万级图片运行时的内存占用。
- `quality_check` 支持 `runtime.quality_check_executor` 选择 `process` 或 `thread`，默认 `process`；`runtime.quality_check_workers` 表示进程/线程数量，默认 8；设置 workers 为 1 时退回单进程顺序处理。
- `process` 模式优先使用 `forkserver` 进程上下文，不可用时退到 `spawn`，避免在已有线程的父进程中直接 `fork`。
- `quality_check` 支持本地分片 checkpoint：`runtime.quality_check_shard_size` 默认 10000，`runtime.quality_check_resume_shards` 默认开启。恢复时会复用已成功且校验通过的 shard，只重跑缺失、失败、配置不匹配或行数不匹配的 shard，最后合并为原有 `data/quality_results.jsonl`。
- `run resume` 默认沿用原始 `run start` 的 `--until/--from-stage` 边界，也支持在 resume 命令上显式覆盖。
- `run resume` 进入 running 时会清理旧的 run-level `finished_at/error_json`，正常停在 `paused/review_ready` 时重新写入新的完成时间，避免中断恢复后状态页显示旧错误时间。
- checkpoint 读取会跳过损坏 JSONL 行，避免异常中断后最后一条半写记录阻断恢复。
- `quality_check` 的 shard 复用增加轻量实现版本常量，防止质量检查代码变更后误用旧 shard。
- `quality_check` 和 `thumbnail` 的错误输出改为流式写入；`embedding`、`auto_decision` 对仍需内存加载的输入增加 `runtime.max_in_memory_rows` 风险保护。
- `KeyboardInterrupt` 中断运行时会把当前阶段和 run 标记为 `aborted`，避免状态文件长期显示 `running`。

### 已验证

- `/data/envs/dataclean-tools/bin/python -m pytest` 通过，结果为 `4 passed`。
- `configs/example-local.yaml` 通过 `image-labeling config validate`。
- MEP-3M 商品分类初筛 run 已验证 `ingest` 可完成递归扫描；`quality_check` 手动中断后状态会落为 `aborted`，已成功 shard 可由 checkpoint 复用。
