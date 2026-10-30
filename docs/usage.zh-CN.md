# 使用指南

[← PersonOS](../README.zh-CN.md) · [English](usage.md) · 简体中文 · [评测](evaluation.zh-CN.md)

PersonOS 通过 Python `Memory` API 或独立部署的 HTTP 服务处理文本、图片和视频记忆。本指南说明安装、模型配置、存储，以及写入和读取的流程。

[文本记忆](#文本记忆) · [视频记忆](#视频记忆) · [安装选项](#安装选项) · [模型权重](#模型权重) · [配置](#配置) · [核心 API](#核心-api) · [HTTP 服务](#http-服务)

## 文本记忆

以下文档以当前源码为准。源码安装包含本次新增的浏览接口与 `flash`；PyPI 已发布版本可能尚未包含这些能力。

需要 **Python 3.10+**，以及可用的对话模型和向量模型。下面使用包内默认的 OpenAI 模型配置，API key 需要能访问这两个模型。

```bash
git clone https://github.com/2049lab/PersonOS.git
cd PersonOS
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

export PERSONOS_LLM_API_KEY="your-api-key"
export PERSONOS_LLM_BASE_URL="https://api.openai.com/v1"
export PERSONOS_LLM_MODEL="gpt-4o-mini"
export PERSONOS_EMBEDDING_MODEL="text-embedding-3-small"
export PERSONOS_EMBEDDING_DIM=1536

personos doctor
```

Windows 用户可用 `.venv\Scripts\Activate.ps1` 激活环境，再通过 PowerShell 或 `.env` 文件设置相同变量。`personos doctor` 检查配置与已安装依赖，不会验证 API key、模型可用性或网络连通性。

使用其他服务商时，请明确设置对话模型、向量模型和向量维度。端点与凭证的继承规则见[模型服务配置](#模型服务配置)。

### 写入与检索

```python
from personos import Memory

with Memory() as memory:
    memory.add(
        "我在 2026 年 6 月从杭州搬到了上海。",
        user_id="quickstart-alice",
        session_id="first-conversation",
    )
    memory.end_session(
        user_id="quickstart-alice",
        session_id="first-conversation",
        sync=True,
    )

    result = memory.search("我住在哪？", user_id="quickstart-alice")
    print(result.ans.answer if result.ans else "没有返回答案。")
    print(result.ans.cited_cells if result.ans else [])
```

预期答案会提到上海，措辞与检索质量取决于所用模型。`add()` 把任务放入队列；`end_session(sync=True)` 关闭情景片段，并等待写入完成后再读取。示例会保留这段记忆，方便继续提问。

## 视频记忆

保留上面的文本配置，安装身份识别依赖，再配置支持当前后端 `video_url`、`image_url` 和 `input_audio` 内容格式的多模态服务。仅支持文本或图片的对话端点无法满足视频接入要求。

```bash
python -m pip install 'personos[identity]'
export PERSONOS_VIDEO_BACKEND=real
export PERSONOS_MLLM_API_KEY="your-multimodal-api-key"
export PERSONOS_MLLM_BASE_URL="https://your-provider.example/v1"
export PERSONOS_MLLM_MODEL="your-video-model"
export PERSONOS_MEDIA_BASE_URL="https://your-media-host.example/media"
```

将示例地址和模型名换成实际配置。使用本地存储时，需要把 `<PERSONOS_DATA_DIR>/media`（默认 `~/.personos/media`）中的文件托管到媒体 URL 下，让模型服务能够获取视频；也可以配置 [OSS 存储](../.env.example)。设置 URL 本身不会启动媒体服务器。身份识别依赖和模型权重还需要额外下载，详情见[模型权重](#模型权重)。

```python
from personos import Memory

clips = ["clip000.mp4", "clip001.mp4", "clip002.mp4"]  # 替换成你的连续视频片段

with Memory() as memory:
    for clip in clips:
        memory.add(
            [{"role": "user", "content": "", "video": clip}],
            user_id="robot",
            session_id="living-room",
        )

    memory.end_session(
        user_id="robot", session_id="living-room",
        sync=True, timeout_s=600 * len(clips) + 600,
    )
    result = memory.search("录像里出现了谁？他们做了什么？", user_id="robot")
    print(result.ans.answer if result.ans else "没有返回答案。")
    print(result.ans.cited_cells if result.ans else [])
```

视频按片段顺序处理，人物身份在会话关闭时提交。处理每段视频可能需要数分钟，具体取决于模型和录像。样例准备、媒体托管与数据来源说明见[视频示例](../examples/README.md#3-video-and-person-identity)。

## 安装选项

使用 `python -m pip install 'personos[EXTRA]'`；也可以用逗号组合安装，例如 `'personos[deep,image]'`。

| 安装 | 加上什么                        |
|---|-----------------------------|
| `personos` | 分层文本写入、快速检索、用户画像和本地 SQLite |
| `personos[deep]` | 多步深度召回 Agent            |
| `personos[image]` | 图片处理工具；图片理解还需要配置多模态模型服务 |
| `personos[identity]` | 视频 + 人脸 / 体态 / 声纹身份（约 2 GB） |
| `personos[mysql]` `personos[redis]` | 多进程部署                       |
| `personos[oss]` | 对象存储替代本地媒体文件                |
| `personos[anthropic]` | Anthropic 协议对话服务；向量模型需要单独配置 |
| `personos[observability]` | 可选的 Langfuse 追踪 |
| `personos[all]` | 全部可选能力 |

## 模型权重

真实身份识别后端使用以下模型：

| 能力 | 权重 | 位置 |
|---|---|---|
| 人脸识别 | InsightFace `buffalo_l`（约 300 MB） | `~/.insightface`（自动下载） |
| 声纹 | SpeechBrain `spkrec-ecapa-voxceleb` | HuggingFace 缓存；可用 `PERSONOS_ECAPA_MODEL` / `PERSONOS_ECAPA_DIR` 覆盖 |
| 多模态理解 | 无，走远程 API | `PERSONOS_MLLM_*` 指向兼容的多模态端点 |

模型权重与外部数据集各自适用其使用条款，不沿用本库许可证。

## 配置

### 模型服务配置

默认服务采用 OpenAI 兼容协议。对话与向量生成是两项独立能力，可以使用同一个端点，也可以分别配置。

| 配置项 | 默认值或继承规则 |
|---|---|
| `PERSONOS_LLM_BASE_URL` | `https://api.openai.com/v1` |
| `PERSONOS_LLM_MODEL` | `gpt-4o-mini` |
| `PERSONOS_LLM_API_KEY` | 默认对话服务需要配置 |
| `PERSONOS_EMBEDDING_BASE_URL` | 留空时沿用 `PERSONOS_LLM_BASE_URL` |
| `PERSONOS_EMBEDDING_API_KEY` | 留空时沿用 `PERSONOS_LLM_API_KEY` |
| `PERSONOS_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `PERSONOS_EMBEDDING_DIM` | `1536`，必须与向量模型实际返回的维度一致 |
| `PERSONOS_MLLM_BASE_URL` | 留空时沿用 `PERSONOS_LLM_BASE_URL` |
| `PERSONOS_MLLM_API_KEY` / `PERSONOS_MLLM_MODEL` | 启用多模态理解时需显式设置，不继承 key 或模型名 |
| `PERSONOS_MLLM_ENDPOINT` | 可选，覆盖多模态服务的完整 URL |

兼容对话接口的服务不一定支持向量生成。使用独立的向量服务时，需要设置 `PERSONOS_EMBEDDING_BASE_URL` 和 `PERSONOS_EMBEDDING_API_KEY`。修改配置中的向量维度，不会要求服务商改变模型实际输出的维度。

使用 Anthropic 协议对话服务时，安装 `personos[anthropic]`，设置 `PERSONOS_LLM_PROVIDER=anthropic`，并配置 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_MODEL`；`ANTHROPIC_BASE_URL` 可指定兼容网关。向量服务通过 `PERSONOS_EMBEDDING_*` 单独配置。其他服务实现可通过[服务注册表](../personos/providers/registry.py)选择。

### 存储与可选能力

| 能力 | 默认行为 | 配置方式 |
|---|---|---|
| 数据库 | SQLite 位于 `~/.personos/personos.db` | `PERSONOS_DB_URL` 指定 MySQL |
| 媒体存储 | 文件位于 `~/.personos/media` | `PERSONOS_MEDIA_BACKEND=oss` 指定对象存储 |
| 图片理解 | 关闭 | 配置多模态模型；未配置时保存原图，但不生成可检索的描述 |
| 视频人物识别 | 关闭 | 安装身份识别依赖，设置 `PERSONOS_VIDEO_BACKEND=real`，并配置兼容的多模态模型和可访问的媒体 URL |
| 重排序 | 关闭，保留检索融合后的顺序 | 设置 `PERSONOS_RERANK_API_KEY`、`PERSONOS_RERANK_MODEL`，按需指定端点 |
| 深度检索 | 关闭 | 安装 `personos[deep]` |
| 共享会话状态 | 保存在单个进程中 | 多个 worker 需要设置 `PERSONOS_REDIS_URL` |
| 追踪 | 关闭 | 安装 `personos[observability]` 并配置 Langfuse |

`PERSONOS_DATA_DIR` 可以调整本地数据库、媒体与默认日志的存放目录。模型调用使用配置的端点；本地保存数据不决定推理运行的位置。多个 worker 需要共享 MySQL 数据库和 Redis。

库会从当前目录或其父目录读取 `.env`，已导出的环境变量优先。可用 `PERSONOS_ENV_FILE` 指定配置文件。[`.env.example`](../.env.example) 列出了模型、存储、超时、worker 与追踪选项。

必要能力不可用时会抛出 `MissingCapability`。图片理解等可选能力不可用时，会给出警告并保留原始材料。`personos doctor` 提供配置状态与设置提示。

## 核心 API

以下调用使用已创建的 `Memory` 实例：

```python
memory.add(messages, user_id=..., session_id=...)
memory.end_session(user_id=..., session_id=..., sync=True)
memory.flush(user_id=..., session_id=...)
memory.search(query, user_id=..., mode="auto")  # flash | auto | fast | deep
memory.profile(user_id=...)                    # 提取后的用户画像
memory.trace(node_id, user_id=...)             # 双向证据溯源
memory.capabilities()                         # 配置与依赖检查
memory.reset(user_id=...)                      # 清理已保存记录，限制见下文
```

`add()` 接受字符串、消息字典或消息字典列表。消息可包含 `image`（字节或路径），或 `video`（路径、文件对象或 URL）。视频片段与对话消息应分别调用写入。完整输入示例见[文本、图片与视频示例](../examples/README.md)。

### 什么会成为记忆

文本提取关注用户的事实、经历、偏好、关系与计划。没有相关个人内容时可以不生成情景；情景也可以没有原子记忆，不会为了满足数量要求补造记忆。原始证据仍保留。情景提取失败时同样保留证据，不把整段原文直接替代为记忆。边界和行为变化见[框架设计](framework.zh-CN.md)。

不同消息中的相同文字会分别保存。队列消费重试沿用原有证据 ID，避免重复写入同一条原始记录；这不构成整条链路的“恰好处理一次”保证。

### 写入顺序

`add()` 和 `end_session()` 默认异步执行：将任务放入队列后立即返回 `AddReceipt`。同一会话内按 FIFO 顺序处理，不同会话之间公平调度，过载时通过背压限制提交。

传入 `sync=True`，或在入队后调用 `flush(user_id=..., session_id=...)`，可以等待该会话的任务处理完成。关闭会话才会完成末尾情景片段的写入并提交视频人物身份。`flush()` 只等待队列处理完成，不会自行关闭会话。用户画像整理可以继续在独立后台任务中进行。

`timeout_s` 控制同步操作的等待时间，超时会抛出 `TimeoutError`。队列满时抛出 `QueueBusy`，HTTP API 返回带 `Retry-After` 的 `503`。

### 召回与溯源

读取（`search`、`profile`、`trace`）是同步的。所有召回模式都要求非空查询、有效模式，以及 1–100 范围内的整数 `top_k`，不接受布尔值。`mode="fast"` 使用快速检索；安装深度检索依赖后，`mode="auto"` 可以升级到深度检索；`mode="deep"` 需要安装 `deep` 可选依赖。

```python
result = memory.search("谁吃了糖？", user_id="pebble")
print(result.ans.answer if result.ans else "没有返回答案。")
print(result.ans.cited_cells if result.ans else [])
```

| 结果字段 | 内容 |
|---|---|
| `result.ans` | 返回答案时，包含答案与引用的情景记录 |
| `result.rw.subject` | 解析后的问题主体 |
| `result.hits` | 按融合顺序排列的原子记忆 |
| `result.reviews` | 对答案草稿的复核结论 |
| `result.to_public()` | 普通字典形式的公开结果 |

通过引用的情景记录和 `trace()`，可以查看答案或身份匹配背后的原始证据。

### 为 Agent 提供材料

```python
result = memory.search("我的饮食偏好是什么？", user_id="alice", mode="flash")
material = result.to_public()
for item in material["materials"]:
    print(item["episode"], item["evidence_refs"])
```

`flash` 跳过查询改写、答案生成与答案复核，仍会对查询向量化并使用已配置的重排服务。`result.ans` 为 `None`，公开结果中的 `answer` 为空、`verdict` 为 `None`。返回的记忆是供接入方 Agent 使用的候选材料，不是已经复核的答案。该模式只接受文本查询；携带图片时请使用 `fast` 或 `auto`。`top_k` 接受 1–100 的整数。它直接使用原查询，不使用查询改写、画像注入或 `scenario`，也不要求对话模型凭证。

公开结果中的 `materials` 包含情景正文、时间范围、命中的 `atom_ids` 与证据引用；`memories` 保留原子记忆与内联证据。没有检索结果时，两者都是空列表，答案保持为空。

### 浏览已保存记忆

这些读取不调用模型，不要求配置模型凭证。调用需要一个尚未关闭的 `Memory` 实例，并始终限定在指定用户的记录内。

```python
page = memory.list_memories(
    user_id="alice", session_id="first-conversation", limit=20, offset=0,
)
for item in page["items"]:
    print(item["atom_id"], item["text"], item["evidence"])

chain_page = memory.list_chains(user_id="alice", limit=20, offset=0)
if chain_page["items"]:
    chain_id = chain_page["items"][0]["chain_id"]
    history = memory.get_chain(chain_id, user_id="alice", limit=20)
    if history and history["has_more"]:
        next_page = memory.get_chain(
            chain_id, user_id="alice", limit=20, cursor=history["next_cursor"],
        )
```

| 方法 | 过滤与分页参数 | 返回内容 |
|---|---|---|
| `list_memories` | `session_id`、`object_type`、`holder`、`kind`、`domain`、`query`、`limit`、`offset` | 原子记忆、当前链关联、情景上下文与本用户的原始证据 |
| `list_chains` | `limit`、`offset` | 链标题、条数、首尾 ID 与时间，不加载完整历史 |
| `get_chain` | `chain_id`、`limit`、`cursor` | 链元数据及按链顺序排列的一页记忆 |

`limit` 范围为 1–100，默认 20；`offset` 范围为 0–100000。列表返回 `items`、`limit`、`offset` 和 `has_more`，使用确定的 ID 顺序。`query` 做文字匹配，不做语义检索；多个过滤条件之间为 AND。`kind` 与 `domain` 使用 [`personos/models.py`](../personos/models.py) 中的编码。MySQL 上的 `kind`、`domain` 和 `query` 过滤需要 JSON 函数（MySQL 5.7+）；本次本地验证使用 SQLite。

链详情额外返回 `next_cursor` 和 `state`，状态包括 `complete`、`more`、`incomplete`、`changed`。继续读取时使用返回的游标；链发生变化后应从第一页重新读取。未知链或其他用户的链返回 `None`。无法在当前用户内解析的证据引用，不会从其他用户的数据中补齐。

### 重置的范围

`reset(user_id=...)` 清理已保存记录与部分会话状态，不提供覆盖待处理队列、外部媒体和模型服务端数据的完整擦除流程。调用方需要停止写入并另行处理这些数据生命周期步骤，详见[存储与部署](framework.zh-CN.md#存储与部署)。

## HTTP 服务

[`server/`](../server/README.md) 提供 HTTP 封装，有独立的依赖。它随源码仓库分发，不包含在库的 wheel 包中。完成上面的模型配置后，在仓库根目录运行：

```bash
python -m pip install -r server/requirements.txt
python -m pip install -e .
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

新增的读取接口沿用 `X-User-Token` 用户作用域和 `{code, data, msg}` 响应格式：

| 接口 | 用途 |
|---|---|
| `GET /api/v1/memories` | 按条件分页查看原子记忆；用 `caller` 和 `session_id` 选择相应调用方的会话 |
| `GET /api/v1/chains` | 分页读取记忆链元数据 |
| `GET /api/v1/chains/{chain_id}` | 用 `limit` 与 `cursor` 逐页读取链历史 |
| `POST /api/v1/recall`，`mode="flash"` | 返回检索材料，不生成答案 |

查询参数类型不符或分页超出范围时返回 `422`，无效过滤值或游标返回 `400`；未知或其他用户的链返回 `404`。请求仍需满足服务的签名和 token 要求。

服务沿用库的配置，并使用以下选项：

| 配置项 | 用途 |
|---|---|
| `PERSONOS_AKSK_MAP` | JSON 格式的访问 key 与签名 secret 映射 |
| `PERSONOS_DB_URL` | 多个 worker 共用的 MySQL 数据库 |
| `PERSONOS_REDIS_URL` | 共享写入队列、会话锁与片段状态 |
| `PERSONOS_ENV` | 共享 Redis 状态的命名空间 |

未配置 Redis 时，队列与会话状态只在单个进程内有效。增加 worker 前，需要配置共享 MySQL 和 Redis。线程池大小与队列限制见 [`.env.example`](../.env.example)，运行中的服务在 `/docs` 提供与当前路由一致的 OpenAPI 文档，原有 `/` 和 `/api-doc` 地址重定向至此。签名与用户 token 细节见 [HTTP 鉴权说明](../server/authentication.md)。

[← PersonOS](../README.zh-CN.md) · [English](usage.md) · [评测](evaluation.zh-CN.md)
