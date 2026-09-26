<div align="center">

# PersonOS

**Any in, memory out** —— 面向智能体、机器人和智能硬件的多模态长期记忆层。

你的智能体感知到的一切 —— 对话、图片、实时视频 —— 进;分层的、可追溯的
记忆出。它也是唯一一个能看视频、并且记住「视频里是谁」的开源记忆框架。

[![PyPI](https://img.shields.io/pypi/v/personos)](https://pypi.org/project/personos/)
[![Python](https://img.shields.io/pypi/pyversions/personos)](https://pypi.org/project/personos/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[English](README.md) · 简体中文

[快速开始](#快速开始) · [评测](#评测) · [设计](#为什么选-personos) · [安装](#安装) · [配置](#配置) · [API](#api) · [示例](examples/README.md)

</div>

## 快速开始

```bash
pip install personos
export PERSONOS_LLM_API_KEY=sk-...
export PERSONOS_LLM_BASE_URL=https://api.openai.com/v1   # 任何 OpenAI 兼容网关均可
```

```python
from personos import Memory

m = Memory()                      # 数据落在 ~/.personos 下的 SQLite;上面两个环境变量就是全部配置

m.add("我六月份从杭州搬到了上海", user_id="alice", session_id="s1")
m.end_session(user_id="alice", session_id="s1", sync=True)   # 等待队列消费完

print(m.search("我现在住在哪?", user_id="alice").ans.answer)
```

不确定当前配置能做什么?`personos doctor` 会读取配置,告诉你哪些能力可用、
哪些被关了、以及怎么打开。

## 评测

**[LoCoMo-10](https://github.com/snap-research/locomo)** —— 长期对话记忆基准
(10 段对话,每段约 300 轮,1,536 道可答题;对抗题按社区惯例不计入):

| 多跳 | 时序 | 开放域 | 单跳 | **总分** |
|---|---|---|---|---|
| 77.3% | 79.8% | 58.7% | 89.3% | **83.3%** |

**[LongMemEval-S](https://github.com/xiaowu0162/LongMemEval)** —— 长期记忆
基准(5 类问题 + abstention,共 500 题,每题 ~40 个 session 的合成用户)。
评测脚本严格沿用官方判分协议(`get_anscheck_prompt` 六个分支 +
abstention),可分多天断点续跑,详见
[scripts/bench/README.md](scripts/bench/README.md#longmemeval-s)。

| 知识更新 | 多会话 | 单会话 Assistant | 单会话 偏好 | 单会话 用户 | 时序推理 | **总分** |
|---|---|---|---|---|---|---|
| 91.7% | 76.9% | 98.2% | 43.3% | 89.1% | 76.4% | **80.6%** |

总分 = 全 500 题答对 403 题;细分:**可答题** 381/470 = 81.1%(判官
问「答对了吗」)、**abstention** 22/30 = 73.3%(判官问「识别为不可答
了吗」)。

**[M3-Bench-robot](https://github.com/bytedance-seed/m3-agent)** —— 机器人
视角的长视频记忆基准(100 个视频,1,276 道题),与该基准的参考 agent 对比:

| | 总分 | 跨模态推理 | 通用知识提取 | 多跳推理 | 多证据推理 | 人物理解 |
|---|---|---|---|---|---|---|
| **PersonOS** | **61.5%** | 59.0% | 48.0% | 63.5% | 62.8% | 73.5% |
| m3-agent(基线) | 37.4% | 37.0% | 28.4% | 42.4% | 37.1% | 50.9% |

**[Video-MME](https://github.com/BradyFU/Video-MME)**(长视频档,无字幕)
—— 通用视频理解:

| 总分 | 信息概要 | 物体识别 | 空间推理 | 物体推理 | 时序推理 | 动作识别 | 动作推理 | 时序感知 | 属性感知 | OCR | 计数 | 空间感知 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **87.0%** | 93.3% | 92.6% | 90.9% | 87.9% | 87.9% | 84.1% | 85.0% | 83.3% | 85.2% | 78.6% | 70.8% | 33.3% |

LoCoMo 的评测完全可复现:数据集下载、评测脚本、评分口径与判官配置都在
[scripts/bench](scripts/bench/README.md)。

## 为什么选 PersonOS

智能体正在走出对话框。机械臂、智能眼镜、桌面 copilot —— 它们持续不断地感知
世界,模态远不止文本,需要的记忆也得跟得上。多数框架把事实存成一个扁平列表
再检索。PersonOS 保存的是**分层的、只追加的记录**,并且拒绝在写入时消解矛盾:

```
evidence  ──►  memcell(情节)──►  atom  ──►  atom_chain
原始对话      用于回答的叙事单元    检索        同一事实随时间的多次出现,
从不修改                          锚点        成组保留,从不合并
```

这在实践中带来两个结果:

**矛盾会留下来。**「15 条鱼」→「其实是 13」→「其实是 11」会作为三个带时间戳的
原子互相关联地共存。回答层决定哪个是「现在」,记忆层从不静默覆盖过去。问
*「我有几条鱼」* 得到的是当前数量;问 *「这变过吗」* 历史依然都在。

**答案引用情节,而不是碎片。** atom 是检索索引 —— 短而自足、适合向量化的命题;
模型真正阅读的是叙事性的情节。检索粒度和回答粒度被刻意分开,因为好的搜索键
往往成不了好的答案。

### 召回过程完全可见

`search()` 返回答案,**也**返回它是怎么得到这个答案的:改写后的查询、检索到的
atom、排序后的材料、仲裁结论、是否升级到了深轨 agent。答案错了,你能看到是
哪一站出的错,而不是靠猜。

```python
out = m.search("我有几条鱼?", user_id="alice")
out.ans.answer        # 答案
out.rw.subject        # 问题被解析成了关于谁的
out.hits              # 检索到的 atom,按融合序
out.reviews           # 仲裁器对草稿的评判
out.to_public()       # ……或者只想要答案时,拿一个普通 dict
```

### 视频与人物身份

其他开源记忆框架要么只支持文本,要么在写入时把图片压缩成一段 caption。
PersonOS 直接接收**视频片段** —— 机器人和眼镜真正生活其中的那种数据流 ——
从人脸、全身照和声纹构建稳定的 *character* 实体:第 12 个片段里认出的人,
三个会话之后还是同一个人,全程无需任何人预先注册。

可选(`pip install personos[identity]`,约 2 GB 模型依赖)。文本核心不会
import 其中任何一行。

## 安装

核心刻意做小 —— 8 个依赖,没有 torch、没有 langchain、没有 web 框架。
所有重的东西都是按需开启的 extra:

| 安装 | 解锁 |
|---|---|
| `pip install personos` | 文本记忆:分层写入链路、快链 + 深轨召回、用户画像 |
| `personos[deep]` | 多步深轨召回 agent(推荐) |
| `personos[image]` | 图片写入 + 围绕照片召回 |
| `personos[identity]` | 视频:人脸 / 全身 / 声纹的人物身份(约 2 GB) |
| `personos[mysql]` `personos[redis]` | 多进程部署 |
| `personos[oss]` | 用对象存储替代本地媒体文件 |
| `personos[all]` | 全部 |

### 模型权重

可选能力会在首次使用时自动下载自己的权重,无需手动准备:

| 能力 | 权重 | 落盘位置 |
|---|---|---|
| 人脸识别 | InsightFace `buffalo_l`(约 300 MB) | `~/.insightface`(自动下载) |
| 声纹 | SpeechBrain `spkrec-ecapa-voxceleb` | HuggingFace 缓存;可用 `PERSONOS_ECAPA_MODEL` / `PERSONOS_ECAPA_DIR` 覆盖 |
| 多模态理解 | 无 —— 远端 API | `PERSONOS_MLLM_*` 指向任意 MLLM 端点 |

在墙内网络或离线机器上:HuggingFace 设 `HF_ENDPOINT=https://hf-mirror.com`,
或者预先下载后用 `*_DIR` 变量指向本地副本。

## 配置

除了模型端点,一切皆可缺省。没配置意味着某项能力关闭或降级 —— **绝不**意味着
文本链路会坏。

| | 默认 | 不配置时 |
|---|---|---|
| **Chat + embeddings** | — | **必填** |
| 数据库 | `~/.personos` 下的 SQLite | 设 `PERSONOS_DB_URL` 用 MySQL —— 只有多 worker 才需要 |
| 多模态模型 | 关 | 图片照存但不贡献检索;视频会被拒收并附操作指引 |
| 媒体存储 | 本地文件 | 设 `PERSONOS_MEDIA_BACKEND=oss` 用对象存储 |
| Reranker | 关 | 检索保持融合序 |
| 深轨召回 | 关 | `pip install personos[deep]` 启用多步 agent |
| Redis | 关 | 单进程;会话状态在内存里 |
| 链路追踪 | 关 | 每个 tracing 调用都是 no-op |

完整列表和说明见 [.env.example](.env.example)。

### 报错会告诉你怎么办

请求一个没配置的能力,你得到的是一句话,而不是静默:

```
MissingCapability: video understanding is unavailable: no multimodal model is configured
  To enable it: set PERSONOS_MLLM_API_KEY and PERSONOS_MLLM_MODEL
```

规则是:**完全做不到 → 抛异常;部分完成 → 照常返回并在 `result.warnings`
里明说**。缺一个可选能力永远不会让写入失败。

## API

```python
m.add(messages, user_id=..., session_id=...)   # 文本、带图的 dict、或视频片段
m.end_session(user_id=..., session_id=...)     # 闭合分段,构建记忆
m.search(query, user_id=..., mode="auto")      # "auto" | "fast" | "deep"
m.profile(user_id=...)                         # 蒸馏出的用户画像
m.trace(node_id, user_id=...)                  # 双向溯源
m.capabilities()                               # 当前配置能做什么
m.reset(user_id=...)                           # 删除某个用户的全部数据
```

**写入默认是异步的。** `add()`/`end_session()` 会进入按会话排序的队列
(与服务端部署用的是同一套机制 —— 会话内 FIFO、会话间公平调度、会话过载时
背压),并立即返回一个 `AddReceipt`,记忆写入不会阻塞你应用自己的工作:

```python
receipt = m.add(..., user_id=..., session_id=...)   # 立即返回
# ……你的代码继续跑;后台 dispatcher 在构建记忆……

m.flush(user_id=..., session_id=...)                # 或者:等到队列排空
m.end_session(..., sync=True)                       # 或者:闭合并等待,一次调用
```

需要「写后立刻读」时,给 `add()`/`end_session()` 传 `sync=True`,或者调
`flush()`。`queue_status()` 报告会话的队列深度和游标。队列满会抛
`QueueBusy` —— 与 HTTP API 的 `503 + Retry-After` 是同一份契约。

读(`search`、`profile`、`trace`)是同步的。目前还没有 `AsyncMemory`;
与其假装有,诚实的替代方案是 `await asyncio.to_thread(m.search, q)`。

## 示例

同一个故事,三个章节:
[一周的对话](examples/quickstart.py)(看着画像一天天自己长出来)·
[带一张照片](examples/images.py) ·
[带视频和人物身份](examples/video.py)。每个示例需要什么、真实输出长什么样,
见 [examples/README.md](examples/README.md)。

## 作为服务运行

[`server/`](server/README.md) 是一个 FastAPI 部署 —— 跨进程的有序写入、
背压、按 token 隔离的多租户。它**不在** pip 包里;有自己的依赖和生命周期。

如果你是把记忆嵌进应用里,不需要它。

## 状态

`0.1.0` —— 记忆管线已在生产部署中运行;外面的打包是新的,公开 API 在
`1.0` 之前仍可能微调。发版流程见 [RELEASE.md](RELEASE.md)。

## 参与贡献

欢迎到 [github.com/2049lab/personos](https://github.com/2049lab/personos)
提 issue 和 PR。测试套件用 `pytest` 跑,不依赖任何外部服务。

## 许可证

Apache-2.0,见 [LICENSE](LICENSE)。
