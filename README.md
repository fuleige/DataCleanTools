# DataCleanTools

DataCleanTools 是一个面向图像分类数据集的自动化清洗、预标注和少量人工审核系统设计项目。

当前仓库已经开始第一版实现，重点先跑通本地可执行的离线批处理链路：图片接入、质量过滤、embedding、相似检索、聚类、预标注、自动采用、审核队列和最终结果导出。

## 第一版定位

- 任务形态：CLI 一键自动流水线 + 阶段级手动控制 + 最小审核界面。
- 分类任务：第一版只做单标签分类。
- 检索索引：默认 FAISS HNSW，召回后用原始 embedding 精排。
- 聚类算法：默认 MiniBatchKMeans/KMeans 类算法。
- 预标注：默认多模态 API，兼容 OpenAI 风格接口，同时支持零样本视觉语言模型。
- 审核方式：Web UI 只做状态摘要、图片查看、标签确认/修改、近重复组批量接受和结果路径查看。
- 存储方式：PostgreSQL 保存状态和摘要，对象存储保存 embedding、索引、相似结果、导出文件等大产物；开发场景允许本地 artifact store。

## 运行模型

项目虚拟环境固定放在：

```bash
/data/envs/dataclean-tools
```

首次安装依赖时去掉代理变量：

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy -u ALL_PROXY -u all_proxy \
  /data/envs/dataclean-tools/bin/python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -e ".[dev,faiss]"
```

如果当前机器暂时无法安装 `faiss-cpu`，可以先安装核心依赖：

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy -u ALL_PROXY -u all_proxy \
  /data/envs/dataclean-tools/bin/python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -e ".[dev]"
```

代码会在 FAISS 不可用且 `vector_index.allow_exact_fallback=true` 时退回到 numpy 精确检索，用于本地小规模验证。

默认使用一条命令跑到 `review_ready`：

```bash
image-labeling run start -c configs/product_classification.yaml
```

流水线默认自动完成接入、质量检测、缩略图、embedding、HNSW、相似图、聚类、预标注、自动采用和审核队列生成。

人工根据审核摘要和审核界面判断任务完成后，再手动导出：

```bash
image-labeling export --run-id run_20260511_001
```

第一版不在 Web 页面创建任务或编辑配置。项目、数据集、模型、阈值、标签体系和输入源都以 YAML 配置为准。

本地示例配置在 [configs/example-local.yaml](configs/example-local.yaml)。可以将其中 `input.local_dir` 改成自己的图片目录后运行：

```bash
/data/envs/dataclean-tools/bin/image-labeling config validate -c configs/example-local.yaml
/data/envs/dataclean-tools/bin/image-labeling run start -c configs/example-local.yaml
```

查看中间结果：

```bash
/data/envs/dataclean-tools/bin/image-labeling run status --run-id <run_id>
/data/envs/dataclean-tools/bin/image-labeling run summary --run-id <run_id>
/data/envs/dataclean-tools/bin/image-labeling stage summary --run-id <run_id> --stage quality_check
/data/envs/dataclean-tools/bin/image-labeling artifacts list --run-id <run_id>
/data/envs/dataclean-tools/bin/image-labeling artifacts sample --run-id <run_id> --artifact quality_results --limit 20
/data/envs/dataclean-tools/bin/image-labeling run logs --run-id <run_id>
```

人工确认审核完成后导出：

```bash
/data/envs/dataclean-tools/bin/image-labeling export --run-id <run_id>
```

## 文档入口

- [系统设计文档总览](docs/README.md)
- [总体系统设计](docs/system-design.md)
- [流水线设计](docs/pipeline-design.md)
- [数据与配置设计](docs/data-and-config-design.md)
- [审核界面设计](docs/review-ui-design.md)
- [实现状态记录](docs/implementation-status.md)

## 当前非目标

- 不做登录、账号、复杂权限、SSO 或多租户。
- 不做复杂质检工作流、审核员绩效、多人分配或仲裁。
- 不做自动训练闭环和模型自动上线。
- 不做在线实时标注或在线向量检索服务。
- 不把审核台扩展成通用标注平台。
