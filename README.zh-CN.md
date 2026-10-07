<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/brand/lockup-tagline-dark.svg">
  <img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/brand/lockup-tagline.svg" alt="PersonOS — Memory that knows who." width="420">
</picture>

<br>

**面向机器人、智能眼镜和 Agent 的多模态长期记忆。**<br>
输入视频、图片和对话，输出的是关于*人*的记忆——而且知道他们是谁。

<br>

[![PyPI](https://img.shields.io/pypi/v/personos?color=6C4CF1&label=pypi)](https://pypi.org/project/personos/)
[![Python](https://img.shields.io/pypi/pyversions/personos?color=3B82F6)](https://pypi.org/project/personos/)
[![License](https://img.shields.io/badge/license-Apache_2.0-10B981)](LICENSE)
[![M3-Bench-robot](https://img.shields.io/badge/M3--Bench--robot-61.5%25-F43F5E)](#基准测试)
[![Video-MME long](https://img.shields.io/badge/Video--MME_long-87.0%25-F59E0B)](#基准测试)

[English](https://github.com/2049lab/personos/blob/main/README.md) · 简体中文

[短片](#短片) · [为什么是身份](#为什么是身份) · [亮点](#亮点) · [基准测试](#基准测试) · [快速开始](#快速开始) · [工作原理](#工作原理) · [安装](#安装) · [路线图](#路线图)

</div>

<br>

## 短片

https://github.com/user-attachments/assets/7252ab1e-9fe8-4c83-b2b1-d3da03b83504

**谁偷吃了万圣节糖果？** 家用小机器人 Pebble 用它唯一的大镜头看着一场派对。
三个孩子以陌生人的身份进门，从彼此的称呼里慢慢得到名字，然后披上一模一样的
白床单消失在面具之下——停电的那一刻，其中一个换上了别人的床单，想把锅甩给他。

右边那本记事本就是 Pebble 的记忆，它的运作方式和 PersonOS 一样：**剧本**在事情
发生的当下就写下来，用的是中性代号（`P1`、`P2`……）；**谁是谁**单独维护，随着
证据到来不断修订，直到当晚记忆保存时才回填到剧本里。第二天早上，所有人都怪那个
床单的主人。Pebble 没有。

<table>
  <tr>
    <td width="33%"><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-1-names.jpg" alt="名字从对话中学到"></td>
    <td width="33%"><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-2-ghosts.jpg" alt="披上床单后名字仍在"></td>
    <td width="33%"><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-3-conflict.jpg" alt="Leo 同时在两个地方？"></td>
  </tr>
  <tr>
    <td><b>名字是挣来的。</b> 一句“哇……Mia，糖在那边！”让陌生人 <code>P1</code> 变成猜测 <i>Mia?</i>——再次被叫到、声音也对上，才确认。</td>
    <td><b>看不见脸，照样认得。</b> 床单之下，鞋子、身高、鞋带上的铃铛和声音，把每个幽灵牢牢连在对的孩子身上。</td>
    <td><b>矛盾会被抓住。</b> 番茄酱床单说这是 <i>Leo</i>；Leo 的声音却从厨房传来。一个人不可能同时在两个地方。</td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-4-fixed.jpg" alt="改认成 Mia"></td>
    <td><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-5-saved.jpg" alt="剧本回填真名"></td>
    <td><img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/demo/beat-6-recall.jpg" alt="带证据的回答"></td>
  </tr>
  <tr>
    <td><b>……然后被纠正。</b> 一串尖细的偷笑和一双粉色球鞋，把这个幽灵挪到了 Mia 的卡片上。记下的事实一个字不改，改的只是它指向谁。</td>
    <td><b>身份最后落定。</b> 当晚记忆保存时，代号在整份剧本里统一解析成真名：一份完整的记录，每个人都对。</td>
    <td><b>回答有据可查。</b> “到底是谁？”——是 Mia，附上确切的几行记录，以及那双出卖她的球鞋的快照。</td>
  </tr>
</table>

<sub>原创短片，用代码绘制（p5.js + p5.brush）；配音与音效来自 ElevenLabs。故事是编排的，机制是真实的——见<a href="#工作原理">工作原理</a>。</sub>

## 为什么是身份

一个能看懂视频、却不认识*人*的记忆系统，会这样记：

```diff
- [21:15] 一个披着番茄酱床单的幽灵走到糖碗边。
- [21:16] 一个幽灵拿了一颗糖。  [21:22] 一个幽灵拿了一颗糖。  [21:29] 一个幽灵拿了一颗糖。
```

哪个幽灵？问它“谁吃了糖”，它最多只能顺着床单找——直接找到错的那个孩子。
PersonOS 记的是：

```diff
+ [21:15] Mia——披着 Leo 那张番茄酱床单——走到糖碗边。
+ [21:16] Mia 拿了一颗糖。  [21:22] Mia 拿了一颗糖。  [21:29] Mia 拿了一颗糖。
```

每一行都指向一个稳定的**人物**，跨片段、跨会话持续存在。名字是**挣来的，不是
录入的**——她之所以是“Mia”，是因为别人这么叫她；在那之前，她是一个稳定的
句柄，你照样可以问关于她的问题。设备自己也是一个人物，因为“这是谁干的？”有时候
答案是“你自己”。

## 亮点

<table>
  <tr>
    <td width="50%" valign="top">
      <h4>认人，而不是认像素</h4>
      人脸、身形和声纹不断累积成每个人物的身份云。第 12 个片段里出现的人，三个
      会话之后仍然是同一个人——不需要录入步骤，也不需要维护人脸库。
    </td>
    <td width="50%" valign="top">
      <h4>会抓自己的错</h4>
      多模态模型会在身份上产生幻觉。PersonOS 用代码检查每一份剧本里物理上不可能
      的矛盾，交回模型修复，并以“做减法”的方式降级——宁可忘掉一个名字，也不学错
      一个名字。
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h4>事实先记，身份确定了再定</h4>
      发生了什么，看见的那一刻就记下；是谁做的，等证据够了再落定。纠正身份改变的
      是记录指向谁——从不改动观察到的内容。
    </td>
    <td valign="top">
      <h4>从不覆盖过去的记忆</h4>
      只追加的记录——证据 → 情节 → 原子 → 链。“15 条鱼 → 13 → 11”作为一条
      相连的历史保留下来；哪一个是当前值，由回答层决定。
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h4>回答给出依据</h4>
      每个回答都引用它来自哪个情节、哪个片段、哪个时间点，完整的检索与裁决轨迹只
      隔一个属性。
    </td>
    <td valign="top">
      <h4>为活在真实世界的设备而生</h4>
      片段通过按会话有序的队列异步流入，带背压；设备主循环不会被阻塞。文本、图片
      和视频进入同一个存储。
    </td>
  </tr>
</table>

## 基准测试

**[M3-Bench-robot](https://github.com/bytedance-seed/m3-agent)**——以机器人第一
视角的长视频记忆：100 段视频、1,276 个问题。身份能力就是为这个场景而做的。

| | **总体** | 人物理解 | 多跳推理 | 多证据 | 跨模态 | 通用知识 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PersonOS** | **61.5** | **73.5** | **63.5** | **62.8** | **59.0** | **48.0** |
| M3-Agent，同模型¹ | 37.4 | 50.9 | 42.4 | 37.1 | 37.0 | 28.4 |
| M3-Agent，论文结果 | 30.7 | 43.3 | 29.4 | 32.8 | 31.2 | 19.1 |

**[Video-MME](https://github.com/BradyFU/Video-MME)**，长视频、无字幕——*只凭记忆*
作答：视频先写入 PersonOS，回答模型从头到尾看不到视频。

| **总体** | 概要 | 物体识别 | 空间推理 | 物体推理 | 时序推理 | 动作识别 | 动作推理 | 时序感知 | 属性感知 | OCR | 计数 | 空间感知 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **87.0** | 93.3 | 92.6 | 90.9 | 87.9 | 87.9 | 84.1 | 85.0 | 83.3 | 85.2 | 78.6 | 70.8 | 33.3 |

**文本记忆不打折**——同一套存储在对话类基准上的表现：

| 基准 | 总体 | 分项 |
|---|:---:|---|
| [LoCoMo-10](https://github.com/snap-research/locomo) · 1,536 题，不含对抗类 | **83.3** | 单跳 89.3 · 时间 79.8 · 多跳 77.3 · 开放域 58.7 |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) · 500 题，官方评判提示词 | **80.6** | 知识更新 91.7 · 单会话-助手 98.2 · 单会话-用户 89.1 · 多会话 76.9 · 时间 76.4 · 偏好 43.3 · 拒答 73.3 |

<sub>数值为准确率（%）。视频记忆由 Qwen3.5-Omni-Plus 构建，由 GPT-5.5 凭记忆作答。¹ 与 PersonOS 使用相同的数据集、记忆模型和回答模型；提示词模板与检索预算不保证一致。文本基准可用 <a href="scripts/bench/README.md"><code>scripts/bench</code></a> 复现；视频评测代码随论文一起发布。</sub>

## 快速开始

```bash
pip install personos
export PERSONOS_LLM_API_KEY=sk-...
export PERSONOS_LLM_BASE_URL=https://api.openai.com/v1   # 任何 OpenAI 兼容网关
```

```python
from personos import Memory

m = Memory()   # SQLite 存在 ~/.personos——什么都不用部署

m.add("我六月份从杭州搬到了上海", user_id="alice", session_id="s1")
m.end_session(user_id="alice", session_id="s1", sync=True)

print(m.search("我住在哪？", user_id="alice").ans.answer)
# -> 上海
```

**给它一双眼睛。** 还是这三个调用，现在加上视频和人物身份：

```bash
pip install 'personos[identity]'
export PERSONOS_VIDEO_BACKEND=real
export PERSONOS_MLLM_API_KEY=sk-...  PERSONOS_MLLM_MODEL=...   # 任何接受视频输入的模型
```

```python
for clip in ["clip000.mp4", "clip001.mp4", "clip002.mp4"]:          # 连续片段
    m.add([{"role": "user", "content": "", "video": clip}],
          user_id="robot", session_id="living-room")

m.end_session(user_id="robot", session_id="living-room", sync=True)  # 身份在这里提交

out = m.search("桌边那个女孩叫什么名字？她学什么专业？", user_id="robot")
print(out.ans.answer)       # Alice ... 数学
print(out.ans.cited_cells)  # 回答所依据的情节（及片段）
```

不确定你的环境能做什么？运行 `personos doctor`。更多示例见
[examples/](examples/README.md)：一周的对话 · 一张照片 · 带人物身份的视频。

## 工作原理

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/architecture-light.svg">
  <img src="https://raw.githubusercontent.com/2049lab/personos/main/docs/assets/architecture.svg" alt="感知 → 身份解析 → 记忆 → 召回" width="100%">
</picture>

**① 感知。** 多模态模型观看每个片段并写出一份**剧本**：画面里有谁——用片段内的
局部代号，不用名字——谁说了什么，发生了什么，什么时候。

**② 身份解析。** 代号通过人脸 / 身形 / 声纹云与已知人物匹配。代码层面的检查
寻找物理上的矛盾（一个人在两个地方、两个人共用一个身份），并交回模型修复。在
一个会话内，**人物链**不断收集证据——第一次露面、一张更清楚的脸、一段更清晰的
声音、一个被喊出来的名字——会话结束时裁决成持久的人物。

**③ 记忆。** 剧本经过这些身份解析后，写入分层、只追加的存储：

```
evidence  ──►  memcell（情节）   ──►  atom       ──►  atom_chain
原始输入        用于回答的             检索锚点          同一事实随时间的
从不修改        叙事单元                                 变化，成组保留、从不合并
```

原子是便于嵌入的短命题——检索索引就建在上面。情节是模型实际阅读的内容。

**④ 召回。** `search()` 会改写问题、解析它在问*谁*、检索、重排、裁决草稿，必要
时升级到多步 Agent——并把整条路径一起返回：

```python
out = m.search("谁吃了糖？", user_id="pebble")
out.ans.answer   # 答案
out.rw.subject   # 问题被解析成在问谁
out.hits         # 检索到的原子，按融合顺序
out.reviews      # 裁决器对草稿的意见
out.to_public()  # ……或者一个普通 dict
```

## 适用场景

| | |
|---|---|
| **家用与服务机器人** | 记住家庭成员、客人以及每个人提出过的要求——跨越数天，无需录入。 |
| **智能眼镜与可穿戴设备** | “周二在展台遇到的那个人是谁？我们聊了什么？” |
| **桌面与陪伴型 Agent** | 聊天、截图、通话共用一份记忆，每个回答都能追溯到来源。 |

## 安装

核心只有 7 个依赖——没有 torch，没有 LangChain，没有 Web 框架。更重的能力都是
按需选装：

| 安装 | 解锁 |
|---|---|
| `pip install personos` | 文本记忆：分层写入、快速 + 深度召回、用户画像 |
| `personos[deep]` | 多步深度召回 Agent（推荐） |
| `personos[image]` | 图片写入与照片召回 |
| `personos[identity]` | 视频 + 人脸 / 身形 / 声纹身份（约 2 GB） |
| `personos[mysql]` `personos[redis]` | 多进程部署 |
| `personos[oss]` | 对象存储替代本地媒体文件 |
| `personos[all]` | 全部 |

<details>
<summary><b>模型权重</b></summary>

<br>

可选能力在首次使用时自行下载权重：

| 能力 | 权重 | 位置 |
|---|---|---|
| 人脸识别 | InsightFace `buffalo_l`（约 300 MB） | `~/.insightface`（自动下载） |
| 声纹 | SpeechBrain `spkrec-ecapa-voxceleb` | HuggingFace 缓存；可用 `PERSONOS_ECAPA_MODEL` / `PERSONOS_ECAPA_DIR` 覆盖 |
| 多模态理解 | 无——远程 API | `PERSONOS_MLLM_*` 指向任意 MLLM 端点 |

离线或在防火墙后：设置 `HF_ENDPOINT=https://hf-mirror.com`，或预先下载并把
`*_DIR` 变量指向本地副本。

</details>

<details>
<summary><b>配置</b></summary>

<br>

除模型端点以外都是可选的。未设置意味着某项能力关闭或降级——绝不会让文本链路
出错。

| | 默认 | 未设置意味着 |
|---|---|---|
| **对话 + 嵌入模型** | — | **必需** |
| 数据库 | `~/.personos` 下的 SQLite | 设置 `PERSONOS_DB_URL` 使用 MySQL——仅多 worker 时需要 |
| 多模态模型 | 关 | 图片会被存储但不被理解；视频会被拒绝并给出说明 |
| 媒体存储 | 本地文件 | 设置 `PERSONOS_MEDIA_BACKEND=oss` 使用对象存储 |
| 重排器 | 关 | 检索保持融合顺序 |
| 深度召回 | 关 | `pip install personos[deep]` |
| Redis | 关 | 单进程；会话状态放在内存里 |
| 链路追踪 | 关 | 所有追踪调用都是空操作 |

模型服务**通过 URL** 拉取视频片段，因此使用本地媒体存储时，请把
`PERSONOS_MEDIA_BASE_URL` 设成一个可公开访问的前缀（或改用 OSS）。
全部选项见 [.env.example](.env.example)。

调用一个未配置的能力，你得到的是一句话，而不是沉默：

```
MissingCapability: video understanding is unavailable: no multimodal model is configured
  To enable it: set PERSONOS_MLLM_API_KEY and PERSONOS_MLLM_MODEL
```

**完全做不了 → 抛出异常；只做了一部分 → 返回并在 `result.warnings` 里说明。**
缺少可选能力永远不会让一次写入失败。

</details>

<details>
<summary><b>API</b></summary>

<br>

```python
m.add(messages, user_id=..., session_id=...)   # 文本、带图片的 dict，或视频片段
m.end_session(user_id=..., session_id=...)     # 关闭片段；生成记忆并提交身份
m.search(query, user_id=..., mode="auto")      # "auto" | "fast" | "deep"
m.profile(user_id=...)                         # 提炼出的用户画像
m.trace(node_id, user_id=...)                  # 双向溯源
m.capabilities()                               # 当前配置能做什么
m.reset(user_id=...)                           # 删除某个用户的全部数据
```

写入**默认异步**：`add()` / `end_session()` 入队到按会话有序的队列（会话内 FIFO、
会话间公平、过载时背压），立即返回一个 `AddReceipt`。需要读到自己刚写的内容时，
传 `sync=True` 或调用 `m.flush(...)`。队列满时抛出 `QueueBusy`——对应 HTTP API
的 `503 + Retry-After`。

读取（`search`、`profile`、`trace`）是同步的；在异步代码里用
`await asyncio.to_thread(m.search, q)`。

</details>

<details>
<summary><b>作为服务运行</b></summary>

<br>

[`server/`](server/README.md) 是一个 FastAPI 部署——跨进程的有序写入、背压、
基于 token 的多租户。它不属于 pip 包；如果你是在应用里嵌入记忆，不需要它。

</details>

## 路线图

- [x] 分层只追加记忆，快速 + 深度召回
- [x] 图片写入与召回
- [x] 视频记忆：人脸 / 身形 / 声纹身份、自我修复、跨会话认人
- [x] 异步有序写入，HTTP 服务
- [ ] 记忆检查器 UI——时间线、人物画廊、回答轨迹
- [ ] 面向 Claude Code、Cursor 等客户端的 MCP 服务
- [ ] 集成：LangGraph、OpenAI Agents SDK、ROS 2
- [ ] `AsyncMemory`
- [ ] 面向实时摄像头流的流式写入
- [ ] 更轻量的身份安装（不依赖 torch 的人脸识别）
- [ ] 论文 + 完整视频评测代码

## 状态

`0.1.x`——记忆管线已在生产环境运行；外层的打包是新的，公开 API 在 `1.0` 之前
仍可能调整。发布流程见 [RELEASE.md](RELEASE.md)。

## 参与贡献

欢迎 Issue 和 PR——见 [CONTRIBUTING.md](CONTRIBUTING.md)。测试套件用 `pytest`
运行，不需要任何外部服务。

## 致谢

- 短片为原创作品，使用 [p5.js](https://p5js.org) 与
  [p5.brush](https://github.com/acamposuribe/p5.brush) 以代码绘制；配音与音效由
  [ElevenLabs](https://elevenlabs.io) 生成。
- M3-Bench-robot 基准以及示例中使用的视频样本来自
  [M3-Agent / M3-Bench](https://github.com/bytedance-seed/m3-agent)
  （ByteDance-Seed，CC BY-NC-SA 4.0）。样本按需下载，本仓库不作再分发。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
