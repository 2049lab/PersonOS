# 评测

[返回 README](../README.zh-CN.md) · [English](evaluation.md) · 简体中文

本文汇总 PersonOS 在多模态与文本记忆基准上的项目报告结果。所有分数的单位均为百分比（%）。

| 评测集 | 已报告总分（%） | 范围 |
|---|---:|---|
| [M3-Bench-robot](https://github.com/bytedance-seed/m3-agent) | **61.5** | 机器人第一视角多模态记忆 |
| [Video-MME](https://github.com/BradyFU/Video-MME) | **87.0** | 长视频、无字幕 |
| [LoCoMo-10](https://github.com/snap-research/locomo) | **83.3** | 文本记忆 |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) | **80.6** | 文本记忆；总分统计可回答问题，拒答表现单独报告 |

## M3-Bench-robot

[M3-Bench-robot](https://github.com/bytedance-seed/m3-agent) 从机器人的第一人称视角，评测日常环境中的多模态记忆能力。

| 系统 | 总体 | 人物理解 | 多跳推理 | 多证据 | 跨模态 | 通用知识 |
|---|---:|---:|---:|---:|---:|---:|
| **PersonOS** | **61.5** | **73.5** | **63.5** | **62.8** | **59.0** | **48.0** |
| M3-Agent，报告标注“相同模型” | 37.4 | 50.9 | 42.4 | 37.1 | 37.0 | 28.4 |
| M3-Agent，报告标注“论文结果” | 30.7 | 43.3 | 29.4 | 32.8 | 31.2 | 19.1 |

## Video-MME

[Video-MME](https://github.com/BradyFU/Video-MME) 的报告结果采用长视频、无字幕设置。

| 总体 | 概要 | 物体识别 | 空间推理 | 物体推理 | 时序推理 | 动作识别 | 动作推理 | 时序感知 | 属性感知 | OCR | 计数 | 空间感知 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **87.0** | 93.3 | 92.6 | 90.9 | 87.9 | 87.9 | 84.1 | 85.0 | 83.3 | 85.2 | 78.6 | 70.8 | 33.3 |

## LoCoMo-10

[LoCoMo](https://github.com/snap-research/locomo) 评测长篇、多会话对话中的记忆召回能力。

| 分类 | 分数（%） |
|---|---:|
| **总体** | **83.3** |
| 单跳 | 89.3 |
| 时间 | 79.8 |
| 多跳 | 77.3 |
| 开放域 | 58.7 |

[评测指南](../scripts/bench/README.md#scoring-protocol) 定义了两种评分口径：

- **Mem0 口径（`score`）：** mode-A 回答器接收召回摘要，以及重排后前 20 个 cell 中按 cell 分组的 atoms；评判模型对这个回答器的输出评分。
- **产品口径（`score_r5`）：** 评判模型直接对 PersonOS 返回的 R5 答案评分。

已报告分数尚未标明采用哪种口径。脚本采用 CORRECT/WRONG 二元评判，评分前将相对时间转换为绝对日期；第 5 类对抗问题没有标准答案，不计入准确率的分母。

## LongMemEval-S

[LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) 评测跨会话文本记忆，涵盖知识更新、时间推理与个人偏好等任务。

| 分类 | 分数（%） |
|---|---:|
| **总体，可回答问题** | **80.6** |
| 知识更新 | 91.7 |
| 单会话—助手 | 98.2 |
| 单会话—用户 | 89.1 |
| 多会话 | 76.9 |
| 时间推理 | 76.4 |
| 偏好 | 43.3 |
| 拒答 | 73.3 |

脚本使用官方按任务类型定义的评判提示词和 yes/no 解析规则。可回答问题的准确率不包含拒答题；拒答单独统计，衡量模型能否识别现有信息不足以回答的问题。详见 [LongMemEval-S 评分协议](../scripts/bench/README.md#scoring-protocol-1)。

## 复现

[评测指南](../scripts/bench/README.md) 提供文本评测命令、模型与评判模型配置、评分协议及产物格式。运行产物保存在 `data/bench/runs/<run_id>/`，包括完整流程记录、报告与汇总。

仓库尚未包含完整的视频评测代码，也缺少将这些报告结果与精确模型及评判模型版本、数据集版本与划分、配置、评分口径和逐题产物关联起来的完整运行清单。M3-Agent“相同模型”基线对应的模型清单也尚未提供。

[返回 README](../README.zh-CN.md) · [English](evaluation.md)
