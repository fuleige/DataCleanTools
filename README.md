# DataCleanTools

DataCleanTools 是一个面向图像分类数据集的自动化清洗、预标注和少量人工审核系统设计项目。

当前仓库处于第一版系统设计阶段，重点是先把离线批处理链路设计清楚：图片接入、质量过滤、embedding、相似检索、聚类、预标注、自动采用、人工审核和最终结果导出。

## 第一版定位

- 任务形态：CLI 一键自动流水线 + 阶段级手动控制 + 最小审核界面。
- 分类任务：第一版只做单标签分类。
- 检索索引：默认 FAISS HNSW，召回后用原始 embedding 精排。
- 聚类算法：默认 MiniBatchKMeans/KMeans 类算法。
- 预标注：默认多模态 API，兼容 OpenAI 风格接口，同时支持零样本视觉语言模型。
- 审核方式：Web UI 只做状态摘要、图片查看、标签确认/修改、近重复组批量接受和结果路径查看。
- 存储方式：PostgreSQL 保存状态和摘要，对象存储保存 embedding、索引、相似结果、导出文件等大产物；开发场景允许本地 artifact store。

## 运行模型

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

## 文档入口

- [系统设计文档总览](docs/README.md)
- [总体系统设计](docs/system-design.md)
- [流水线设计](docs/pipeline-design.md)
- [数据与配置设计](docs/data-and-config-design.md)
- [审核界面设计](docs/review-ui-design.md)

## 当前非目标

- 不做登录、账号、复杂权限、SSO 或多租户。
- 不做复杂质检工作流、审核员绩效、多人分配或仲裁。
- 不做自动训练闭环和模型自动上线。
- 不做在线实时标注或在线向量检索服务。
- 不把审核台扩展成通用标注平台。
