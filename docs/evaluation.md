# Evaluation

[Back to README](../README.md) · English · [简体中文](evaluation.zh-CN.md)

This page records project-reported results for PersonOS on multimodal and text-memory benchmarks. All scores are percentages (%).

| Benchmark | Reported overall (%) | Scope |
|---|---:|---|
| [M3-Bench-robot](https://github.com/bytedance-seed/m3-agent) | **61.5** | Robot-view multimodal memory |
| [Video-MME](https://github.com/BradyFU/Video-MME) | **87.0** | Long videos, no subtitles |
| [LoCoMo-10](https://github.com/snap-research/locomo) | **83.3** | Text memory |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) | **80.6** | Text memory; answerable questions, with abstention reported separately |

## M3-Bench-robot

[M3-Bench-robot](https://github.com/bytedance-seed/m3-agent) evaluates multimodal memory from a robot's first-person view across everyday environments.

| System | Overall | Person understanding | Multi-hop | Multi-evidence | Cross-modal | General knowledge |
|---|---:|---:|---:|---:|---:|---:|
| **PersonOS** | **61.5** | **73.5** | **63.5** | **62.8** | **59.0** | **48.0** |
| M3-Agent, reported as “same models” | 37.4 | 50.9 | 42.4 | 37.1 | 37.0 | 28.4 |
| M3-Agent, reported as “as published” | 30.7 | 43.3 | 29.4 | 32.8 | 31.2 | 19.1 |

## Video-MME

[Video-MME](https://github.com/BradyFU/Video-MME) results use long videos without subtitles.

| Overall | Synopsis | Object recognition | Spatial reasoning | Object reasoning | Temporal reasoning | Action recognition | Action reasoning | Temporal perception | Attribute perception | OCR | Counting | Spatial perception |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **87.0** | 93.3 | 92.6 | 90.9 | 87.9 | 87.9 | 84.1 | 85.0 | 83.3 | 85.2 | 78.6 | 70.8 | 33.3 |

## LoCoMo-10

[LoCoMo](https://github.com/snap-research/locomo) evaluates recall across long, multi-session conversations.

| Category | Score (%) |
|---|---:|
| **Overall** | **83.3** |
| Single-hop | 89.3 |
| Temporal | 79.8 |
| Multi-hop | 77.3 |
| Open-domain | 58.7 |

The [benchmark guide](../scripts/bench/README.md#scoring-protocol) defines two scoring conventions:

- **Mem0 convention (`score`):** a mode-A answerer receives the recall brief and atoms from the top-20 reranked cells, grouped by cell. The judge scores that answerer's output.
- **Product convention (`score_r5`):** the judge scores the R5 answer returned by PersonOS.

The reported scores do not identify which convention produced them. The runner uses a binary CORRECT/WRONG judge, normalizes relative times to absolute dates before judging, and excludes category 5 (adversarial questions without gold answers) from the accuracy denominator.

## LongMemEval-S

[LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) evaluates text memory across sessions, including updates, temporal reasoning and personal preferences.

| Category | Score (%) |
|---|---:|
| **Overall, answerable questions** | **80.6** |
| Knowledge update | 91.7 |
| Single-session assistant | 98.2 |
| Single-session user | 89.1 |
| Multi-session | 76.9 |
| Temporal reasoning | 76.4 |
| Preference | 43.3 |
| Abstention | 73.3 |

The runner uses the official task-specific judge prompts and yes/no parser. Answerable accuracy excludes abstention questions. Abstention is scored separately on whether the model recognizes that the question cannot be answered from the available information. See the [LongMemEval-S scoring protocol](../scripts/bench/README.md#scoring-protocol-1).

## Reproduction

The [benchmark guide](../scripts/bench/README.md) provides text benchmark commands, model and judge configuration, scoring protocols, and artifact formats. Run artifacts are stored under `data/bench/runs/<run_id>/` and include pipeline traces, reports and summaries.

The repository does not yet include the full video evaluation code or a complete run manifest linking these reported results to exact model and judge versions, dataset revisions and splits, settings, scoring conventions, and per-question artifacts. The model manifest for the M3-Agent baseline labeled “same models” is also not included.

[Back to README](../README.md) · [简体中文](evaluation.zh-CN.md)
