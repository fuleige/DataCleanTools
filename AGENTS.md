# Project Notes

- 项目专用虚拟环境固定使用 `/data/envs/dataclean-tools`。
- 安装依赖时需要去掉代理环境变量，并优先使用国内镜像，例如：
  `env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy -u ALL_PROXY -u all_proxy /data/envs/dataclean-tools/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev,faiss]"`
- 如果可选 `faiss-cpu` 下载异常，先安装 `.[dev]` 跑核心链路；代码在 `vector_index.allow_exact_fallback=true` 时会退回 numpy 精确检索，后续再单独安装 `faiss-cpu`。
- 第一版实现重点是 CLI 离线流水线、本地 artifact/state、可检查中间结果和可替换 provider 接口。
- Web/API 只作为审核与状态查看的薄接口，不负责创建任务或编辑配置。
- 当前代码实现状态统一维护在 `docs/implementation-status.md`；新增核心能力、改变运行方式、调整已知限制或完成重要验证时，必须同步更新该文件。
- 为避免污染项目源码目录，手动数据清洗任务、临时配置、运行日志、操作记录和 run 恢复索引统一放在 `/data/codes/DataCleanRuns` 下。
- 每次执行新的数据清洗/初筛/标注任务前，先在 `/data/codes/DataCleanRuns` 创建独立任务文件夹，例如 `YYYYMMDD_HHMMSS_<task_name>`。
- 任务文件夹中应至少保留操作记录文档，记录输入图片目录、使用的配置文件、执行命令、生成的 `run_id`、artifact root、关键结果和后续恢复方式，方便后续用 `run status`、`run resume`、`stage summary` 等命令找回。
- 若需要为某次任务修改 YAML 配置，优先把任务专用配置放在对应的 `/data/codes/DataCleanRuns/<task>/` 目录中，不要把一次性运行配置堆到项目 `configs/` 目录。
- 未经用户明确允许，不执行 `git commit`、`git push`、`git tag` 或其他会改变 Git 历史/远端状态的操作。
