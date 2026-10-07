<p align="center">
  <img src="docs/assets/banner.png" alt="PersonOS — Memory that knows who." width="100%">
</p>

<h1 align="center">PersonOS</h1>

<p align="center"><strong>擅长人物理解的多模态长期记忆框架</strong></p>

<p align="center">
  <a href="https://pypi.org/project/personos/"><img src="https://img.shields.io/pypi/v/personos?color=6C4CF1&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/personos/"><img src="https://img.shields.io/pypi/pyversions/personos?color=3B82F6" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-10B981" alt="License"></a>
  <a href="#评测"><img src="https://img.shields.io/badge/M3--Bench--robot-61.5%25-F43F5E" alt="M3-Bench-robot"></a>
  <a href="#评测"><img src="https://img.shields.io/badge/Video--MME_long-87.0%25-F59E0B" alt="Video-MME long"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · 简体中文
</p>

<p align="center">
  <a href="#演示视频">演示视频</a> ·
  <a href="#简介">简介</a> ·
  <a href="#核心特性">核心特性</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#评测">评测</a> ·
  <a href="#进展">进展</a>
</p>

PersonOS 是面向“会看会听”的设备的长期记忆框架，适用于家用机器人、智能眼镜、桌面 Agent。
支持接收多模态输入，将视频、图片、对话喂给它，它会记下发生了什么，并弄清楚**谁**说过了什么话、做过了什么事，之后回答问题时将答案和证据一起返回。

## 演示视频

https://github.com/user-attachments/assets/2faea78b-7db6-4a7a-8f6d-22c7cd4a7a38

**谁偷吃了万圣节糖果？** 家用小机器人 Pebble 看着三个孩子进门要糖。开始时它不认识
他们。但是一个晚上下来，它从孩子们互相的对话和称呼中记住了他们的名字，在他们披上一模一样的床单
之后依然分得清谁是谁，并且还抓到了 Mia 趁停电换床单、想把偷吃糖果的锅甩给 Leo 的那一刻。

右边的记事本就是 Pebble 的记忆，它的工作方式和 PersonOS 一样。**情景记忆的剧本**在事情
发生的当下就写下来，用的是人物临时代号（`P1`、`P2`……）。Pebble 将**谁是谁**单独维护在角色表中，
当从观察的人物对象中获取到了更过硬的证据（如更清晰的照片、声音等）就尝试进行身份订正，直到当晚记忆保存时才将人物回填进剧本。
在第二天早上所有人都在怀疑 Leo 的时候，Pebble 凭借着出色的身份识别能力站出来澄清了事实真相。

<table>
  <tr>
    <td width="33%"><img src="docs/assets/demo/beat-1-names.jpg" alt="零样本启动：身份识别从自然对话中获得"></td>
    <td width="33%"><img src="docs/assets/demo/beat-2-ghosts.jpg" alt="多维度身份识别"></td>
    <td width="33%"><img src="docs/assets/demo/beat-3-conflict.jpg" alt="物理矛盾校验"></td>
  </tr>
  <tr>
    <td><b>零样本启动：身份识别从自然对话中获得 </b> 一句"哇……Mia，糖在那边！"让陌生人 <code>P1</code> 变成 <i>Mia?</i></td>
    <td><b>多维度身份识别 </b> 不仅仅依靠声音和人脸来进行身份识别，更通过人物的时空间连贯性、体态等来辨认角色，让蒙上了脸的每个幽灵也能被精准识别。</td>
    <td><b>物理矛盾校验 </b> 一个人物不可能同时出现在两个位置——当番茄酱床单幽灵说自己是 Leo，但是 Leo 的声音却从厨房传来时，总有一个是错的。</td>
  </tr>
  <tr>
    <td><img src="docs/assets/demo/beat-4-fixed.jpg" alt="人物身份可订正"></td>
    <td><img src="docs/assets/demo/beat-5-saved.jpg" alt="最后才确定角色身份和最终剧本"></td>
    <td><img src="docs/assets/demo/beat-6-recall.jpg" alt="回答均带有溯源证据"></td>
  </tr>
  <tr>
    <td><b>……人物身份被改正 </b> 人物身份可被订正，一声偷笑加一双粉色球鞋暴露了这个幽灵其实是 Mia，于是她的临时身份被记录到 Mia 的角色卡上。已经记录的情景记忆无需修改，只需要修改临时身份的指向。</td>
    <td><b>最后才确定角色身份和最终剧本 </b> 当采集结束保存时，将情景记忆剧本里的所有人物临时代号换成真名，得到一份完整记录。</td>
    <td><b>回答均带有溯源证据 </b> 所有的回答均带有情景记忆证据，当 Mum 责问"到底是谁偷吃了糖果？"——Pebble 除了回答 Mia 外，还会附上剧本里的原话和那双球鞋的照片作为证据。</td>
  </tr>
</table>

## 简介

大多数多模态记忆系统只做理解，不做身份识别。PersonOS 在它们的基础上加入身份识别：无需提前录入人物，就能进行零样本身份识别和情景记忆记录。

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/architecture-light.svg">
  <img src="docs/assets/architecture.svg" alt="感知 → 身份解析 → 记忆 → 召回" width="100%">
</picture>

**1. 感知** 多模态模型依次观看每个视频片段，并写出一份情景记忆剧本：谁说了什么、发生了什么、环境信息如何。

**2. 身份解析** 通过人脸 / 声音 / 体态 / 连贯性尝试将视频中人物和已知人物匹配。同时人物链不断收集和更新人物证据（第一次露面、一张
更清楚的脸、一段更清晰的声音、一个被喊出来的名字），会话结束时将临时身份裁决升级成持久人物。

**3. 记忆** 剧本经过这些身份解析后，写进只追加、不覆盖的分层记忆中：

```
evidence  ──►  memcell（情节）   ──►  atom       ──►  atom_chain
原始输入        回答时参考的           原子记忆         同一件事随时间的变化，
从不修改        叙事单元              检索锚点         成组保留
```

**4. 召回** 从分层记忆中定位到答案和证据，整理后进行作答。

## 核心特性

- **零样本身份识别。** 不需要提前录入人物，从自然对话和观察中逐步弄清楚谁是谁。
- **多维度识别，身份可订正。** 综合人脸、声音、体态和时空间连贯性；更过硬的证据出现时，订正临时身份的指向。
- **多模态情景记忆。** 接收视频、图片和对话，将发生的事情写进分层记忆，让情景记忆和人物关联起来。
- **带证据的回答。** 回答时返回对应的情景记忆和原始证据，能查到答案从哪里来。

一个只做视频理解，没有人物理解的多模态记忆系统，记录的记忆如下：

```diff
- [21:15] 一个披着番茄酱床单的幽灵走到糖碗边。
- [21:16] 一个幽灵拿了一颗糖。
- [21:22] 一个幽灵拿了一颗糖。
- [21:29] 一个幽灵拿了一颗糖。
```

由于无法理解谁是谁、谁做了什么，问它“谁吃了糖”，它就无法正确回答。而 PersonOS 记的是：

```diff
+ [21:15] Mia 披着 Leo 那张番茄酱床单，走到糖碗边。
+ [21:16] Mia 拿了一颗糖。
+ [21:22] Mia 拿了一颗糖。
+ [21:29] Mia 拿了一颗糖。
```

所有的情景记忆都指向一个稳定的人物，以 PersonOS 为长期记忆框架的 Agent 才能真正做到看到、看懂、理解。

## 快速开始

### 文本记忆

```bash
pip install personos
export PERSONOS_LLM_API_KEY=sk-...
export PERSONOS_LLM_BASE_URL=https://api.openai.com/v1   # 任何 OpenAI 兼容端点
```

```python
from personos import Memory

m = Memory()   # SQLite 放在 ~/.personos，什么都不用装

m.add("我六月份从杭州搬到了上海", user_id="alice", session_id="s1")
m.end_session(user_id="alice", session_id="s1", sync=True)

print(m.search("我住在哪？", user_id="alice").ans.answer)
# -> 上海
```

### 视频记忆

要进行视频理解，需要额外配置多模态模型：

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

out = m.search("桌边那个女孩叫什么名字？她学什么的？", user_id="robot")
print(out.ans.answer)       # Alice ... 数学
print(out.ans.cited_cells)  # 回答依据的情节（以及片段）
```

运行 `personos doctor` 看看你的当前配置能支持哪些功能。更多示例见
[examples/](examples/README.md)。

### 安装选项与使用说明

按需安装需要的依赖：

| 安装 | 加上什么                        |
|---|-----------------------------|
| `pip install personos` | 文本记忆：分层写入、快速和深度召回、用户画像      |
| `personos[deep]` | 多步深度召回 Agent（推荐）            |
| `personos[image]` | 图片写入与照片召回                   |
| `personos[identity]` | 视频 + 人脸 / 体态 / 声纹身份（约 2 GB） |
| `personos[mysql]` `personos[redis]` | 多进程部署                       |
| `personos[oss]` | 对象存储替代本地媒体文件                |
| `personos[all]` | 全部                          |

<details>
<summary><b>模型权重</b></summary>

<br>

(可选)人物识别所需要的模型：

| 能力 | 权重 | 位置 |
|---|---|---|
| 人脸识别 | InsightFace `buffalo_l`（约 300 MB） | `~/.insightface`（自动下载） |
| 声纹 | SpeechBrain `spkrec-ecapa-voxceleb` | HuggingFace 缓存；可用 `PERSONOS_ECAPA_MODEL` / `PERSONOS_ECAPA_DIR` 覆盖 |
| 多模态理解 | 无，走远程 API | `PERSONOS_MLLM_*` 指向任意 MLLM 端点 |

</details>

<details>
<summary><b>配置</b></summary>

<br>

只有模型端点是必需配置的。其他配置均有降级。

| | 默认 | 没设的话 |
|---|---|---|
| **对话 + 嵌入模型** | — | **必需** |
| 数据库 | `~/.personos` 下的 SQLite | 设 `PERSONOS_DB_URL` 用 MySQL（只有多 worker 时需要） |
| 多模态模型 | 关 | 图片存下但不理解；视频拒绝并给出说明 |
| 媒体存储 | 本地文件 | 设 `PERSONOS_MEDIA_BACKEND=oss` 用对象存储 |
| 重排器 | 关 | 检索保持融合顺序 |
| 深度召回 | 关 | `pip install personos[deep]` |
| Redis | 关 | 单进程；会话状态在内存里 |
| 链路追踪 | 关 | 所有追踪调用都是空操作 |

模型服务通过 URL 拉视频片段，所以用本地媒体存储时，把 `PERSONOS_MEDIA_BASE_URL`
设成一个公网能访问的前缀（或者改用 OSS）。全部选项在 [.env.example](.env.example)。

调用没配置的能力，会得到警告如下：

```
MissingCapability: video understanding is unavailable: no multimodal model is configured
  To enable it: set PERSONOS_MLLM_API_KEY and PERSONOS_MLLM_MODEL
```

</details>

<details>
<summary><b>核心 API</b></summary>

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

写入默认异步。`add()` 和 `end_session()` 入队到按会话有序的队列（会话内 FIFO、
会话间公平、过载时背压），立即返回一个 `AddReceipt`。需要进行同步写入时，
传 `sync=True` 或调用 `m.flush(...)`。当任务队列满了时抛出异常 `QueueBusy`，HTTP API 对应错误为
`503 + Retry-After`。

读取（`search`、`profile`、`trace`）是同步的。

召回结果同时保留答案、人物、证据和裁决过程：

```python
out = m.search("谁吃了糖？", user_id="pebble")
out.ans.answer   # 答案
out.rw.subject   # 问题被解析成在问谁
out.hits         # 检索到的原子，按融合顺序
out.reviews      # 裁决器对草稿的意见
out.to_public()  # 同样的内容，普通 dict
```

</details>

<details>
<summary><b>作为服务运行</b></summary>

<br>

[`server/`](server/README.md) 提供了对外的 PersonOS 长期记忆服务。

</details>

## 评测

**[M3-Bench-robot](https://github.com/bytedance-seed/m3-agent)** 机器人
第一人称视角的各类常见环境下的多模态记忆框架评测集

| | **总体** | 人物理解 | 多跳推理 | 多证据 | 跨模态 | 通用知识 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PersonOS** | **61.5** | **73.5** | **63.5** | **62.8** | **59.0** | **48.0** |
| M3-Agent，同模型 | 37.4 | 50.9 | 42.4 | 37.1 | 37.0 | 28.4 |
| M3-Agent，论文结果 | 30.7 | 43.3 | 29.4 | 32.8 | 31.2 | 19.1 |

**[Video-MME](https://github.com/BradyFU/Video-MME)**，长视频、无字幕的视频理解评测集

| **总体** | 概要 | 物体识别 | 空间推理 | 物体推理 | 时序推理 | 动作识别 | 动作推理 | 时序感知 | 属性感知 | OCR | 计数 | 空间感知 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **87.0** | 93.3 | 92.6 | 90.9 | 87.9 | 87.9 | 84.1 | 85.0 | 83.3 | 85.2 | 78.6 | 70.8 | 33.3 |

PersonOS 在文本评测集上同样表现出色：

| 基准 | 总体 | 分项 |
|---|:---:|---|
| [LoCoMo-10](https://github.com/snap-research/locomo) | **83.3** | 单跳 89.3 · 时间 79.8 · 多跳 77.3 · 开放域 58.7 |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) | **80.6** | 知识更新 91.7 · 单会话-助手 98.2 · 单会话-用户 89.1 · 多会话 76.9 · 时间 76.4 · 偏好 43.3 · 拒答 73.3 |

## 进展

- [x] 分层记忆架构
- [x] 多模态记忆的理解与召回
- [x] 人物的身份识别
- [x] 异步任务处理，HTTP 服务
- [ ] 记忆检查工作台 UI：查看时间线、人物画廊、情景记忆
- [ ] 面向 Claude Code、Cursor 等客户端的 MCP 服务
- [ ] 实时摄像头流的流式写入
- [ ] 论文和完整的视频评测代码

---

**如何贡献**

欢迎 Issue 和 PR，见 [CONTRIBUTING.md](CONTRIBUTING.md)。测试使用 `pytest`， 不需要任何外部服务。

**致谢**

- M3-Bench-robot 评测集来自
  [M3-Agent / M3-Bench](https://github.com/bytedance-seed/m3-agent)
  （ByteDance-Seed，CC BY-NC-SA 4.0）。

**许可证**

Apache-2.0，见 [LICENSE](LICENSE)。
