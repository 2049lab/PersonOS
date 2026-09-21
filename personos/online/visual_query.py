"""R0 之前的一道**视觉理解 query 改写**:用户发图提问时,把图里的人认出来、写进 query。

为什么需要:用户发一张照片问"他和我上周干嘛去了",纯文本链路既看不到图,也无从知道"他"是谁——
`rewrite_query` 只能拿历史对话补指代,补不出照片里这张脸。这里先把视觉信息落成文字:

    "他和我上周干嘛去了?"  ──[图 + 角色表 + 历史]──▶  "李四上周和我干嘛去了?"

改写完的 query 再照常进 `rewrite_query`,后面的召回链路一个字都不用改。

流程(与 ingest 侧认人共用同一套身份资产,不另起一套):
  图 bytes → RGB 帧 → face_detector 检测 → 每张脸取 embedding/质量/crop
  → 角色表召回(小库全取,大库走概率云粗召回)→ 候选卡带上脸/全身/声音/名字/描述
  → 一次 MLLM:看图 + 看候选素材 + 读历史 + 读原 query → 改写后的 query + 认到了谁

**全链路降级**:解码失败/无脸/无候选/MLLM 挂/JSON 坏 —— 一律原样返回 query,只记日志。
召回是读路径,宁可少一层理解,绝不能因为看图失败就答不出来。

查询图**不落 OSS**:它是查询输入,不是记忆内容(与 ingest 的图片不同,那个要存以便深轨回看)。
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from loguru import logger

from personos.identity.harvest import _body_crop_b64, _face_q
from personos.identity.types import CandidateCard, CastEvidence, FacePick
from personos.online.llm import strip_fences

# 大库时每张脸粗召回的候选数;小库(<=small_library_max)直接全取
COARSE_TOP_K = 5
# 一次改写最多看几张脸:图里人多时只取质量最高的几张,防 prompt 爆炸
MAX_FACES = 4

_SYSTEM = """# Role
You rewrite a user's question so that later text-only retrieval can work, using an image the user just sent.

# Input
- The image the user sent (image #1).
- REGISTERED PEOPLE: people already known from this user's memory, each with name/appearance and photos.
- Recent dialogue history (may include `video:` lines — those are narrative summaries of recordings the user watched earlier).
- The user's current question.

# Task
1. Decide, for each detected face in the image, whether it is one of the REGISTERED PEOPLE.
   - Match on facial features first; clothing and body shape are weaker evidence (people change clothes).
   - **It is correct and expected to answer "none of them".** Do NOT force a match. A wrong name is far
     worse than no name: it sends retrieval to the wrong person entirely.
2. Rewrite the question by replacing visual references ("he", "this person", "this place", "that thing")
   with what they actually are, so the question stands alone without the image.
   - A person you matched → use their known name.
   - A person you could NOT match → describe them briefly ("the man in the blue jacket"), do not invent a name.
   - No people in the image → still use what you see (place, object, text, scene) to make the question concrete.
3. Keep the question's original intent and time expressions untouched. Only resolve what the image resolves.
   If the image adds nothing, return the question unchanged.

# Output
Return a single JSON object, nothing else:
{"resolved": "<rewritten question>", "matched": [{"face_index": 0, "person": "<label>", "name": "<name>"}]}
- `matched` lists only faces you are confident about; leave it `[]` when unsure.
- `person` MUST be one of the short labels (p1, p2, …) from ALLOWED LABELS. Never invent one.

# Examples

Question: "他和我上周干嘛去了?"  → FACE 0 matches p2 (known name 李四)
{"resolved": "李四和我上周干嘛去了?", "matched": [{"face_index": 0, "person": "p2", "name": "李四"}]}

Question: "他和我上周干嘛去了?"  → FACE 0 matches nobody in the list
{"resolved": "照片里那个穿蓝色夹克的男生和我上周干嘛去了?", "matched": []}

Question: "这家店我上周去过吗?"  → no face; the image shows a restaurant sign reading 蜀香源
{"resolved": "蜀香源这家川菜馆我上周去过吗?", "matched": []}

Question: "我上周三下午在干嘛?"  → the image is an unrelated screenshot; it resolves nothing
{"resolved": "我上周三下午在干嘛?", "matched": []}"""


@dataclass
class VisualRewrite:
    """视觉改写结果。任何失败路径都给 query=原问题,调用方无需判错。"""

    query: str
    faces: int = 0                              # 图里检测到的人脸数
    matched: list[dict] = field(default_factory=list)   # [{face_index, character_id, name}]
    system: str = ""
    user: str = ""
    raw: str = ""
    skipped: str = ""                           # 非空 = 没做改写的原因(供日志/观测)


@dataclass
class VisualDeps:
    """视觉改写的依赖(runtime.visual_deps 装配)。backends 是进程单例(重模型)。"""

    store: Any            # CharacterStore(本 user 的人物库)
    cloud: Any            # CloudEngine(概率身份云,大库粗召回用)
    backends: dict        # {face_detector, mm_runner, ...} —— 与 ingest 共用同一份单例
    registry: Any         # AnchorRegistry:只用它的 candidate_card(与 ingest 侧同口径组卡)
    small_library_max: int = 20    # 与 AnchorRegistry 同口径:库内人数 <= 此值则全量进候选


def _to_frame(image: bytes) -> Optional[np.ndarray]:
    """图片字节 → RGB numpy 帧(face_detector 吃的就是这个)。坏图返回 None,不抛。"""
    try:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"查询图解码失败,跳过视觉改写: {e}")
        return None


def _detect(deps: VisualDeps, frame: np.ndarray) -> list[FacePick]:
    """检测人脸 → FacePick(带 embedding/质量/脸 crop/全身 crop),按质量降序取前 MAX_FACES。

    复用 harvest 的 _face_q/_body_crop_b64:与 ingest 侧同一套口径,避免两处质量分定义漂移。
    """
    det = deps.backends.get("face_detector")
    if det is None:
        return []
    try:
        dets = det.detect(frame)
    except Exception as e:  # noqa: BLE001  本地推理失败不阻塞召回
        logger.warning(f"查询图人脸检测失败,跳过视觉改写: {e}")
        return []
    picks = [FacePick(t=0.0, embedding=np.asarray(d.embedding, dtype=np.float32),
                      q=_face_q(d), crop_b64=getattr(d, "crop_b64", ""),
                      body_crop_b64=_body_crop_b64(frame, tuple(d.bbox)))
             for d in dets]
    picks.sort(key=lambda p: p.q, reverse=True)
    return picks[:MAX_FACES]


def _candidates(deps: VisualDeps, picks: list[FacePick]) -> list[CandidateCard]:
    """角色表召回:小库全取,大库每张脸走概率云粗召回后按 character_id 去重。"""
    chars = deps.store.list_active_characters(include_wearer=True)
    if not chars:
        return []
    by_id = {c["id"]: c for c in chars}
    if len(chars) <= deps.small_library_max or not picks:
        chosen = list(by_id)
    else:
        chosen, seen = [], set()
        for i, p in enumerate(picks):
            ev = CastEvidence(cast_id=f"F{i}", faces=[p])
            for cid, _score in deps.cloud.coarse_recall(ev, list(by_id), k=COARSE_TOP_K):
                if cid not in seen:
                    seen.add(cid); chosen.append(cid)
    return [deps.registry.candidate_card(by_id[cid]) for cid in chosen]


def build_prompt(query: str, picks: list[FacePick], cands: list[CandidateCard],
                 history: list[tuple[str, str]] | None, now_dt: Any = None,
                 ) -> tuple[str, list[str], dict[str, str]]:
    """拼 prompt。返回 (user_prompt, images, {短标: character_id})。

    - images[0] 恒为用户发的原图;图片编号 image #N 与 build_arbitration_prompt 同风格,
      让模型能把文字引用对上具体图。
    - 人物用**短标 p1/p2**,不给真实 character_id(26 位 ULID)——项目既有规范:凡 LLM 要靠
      复述 id 引用内容,一律走短↔长映射。长 id 容易被抄错/幻觉,短标好抄且能机械校验,
      解析后再回填真实 id。
    """
    images: list[str] = []
    blocks: list[str] = []
    labels = {f"p{i + 1}": c.character_id for i, c in enumerate(cands)}

    blocks.append("THE USER'S IMAGE: image #1"
                  + (f" ({len(picks)} face(s) detected, cropped as the images that follow)"
                     if picks else " (no face detected in it)"))
    for i, p in enumerate(picks):
        attached = []
        if p.crop_b64:
            images.append(p.crop_b64); attached.append(f"image #{len(images) + 1} = face crop")
        if p.body_crop_b64:
            images.append(p.body_crop_b64); attached.append(f"image #{len(images) + 1} = body crop")
        blocks.append(f"  FACE {i}: " + (", ".join(attached) or "(no crop available)"))

    if cands:
        blocks.append("\nREGISTERED PEOPLE (from this user's memory):")
        for label, c in zip(labels, cands):
            rows = [f"  PERSON {label}:"]
            if c.name:
                rows.append(f"    known name: {c.name}")
            if c.desc:
                rows.append(f"    appearance: {c.desc}")
            attached = []
            if c.face_b64:
                images.append(c.face_b64); attached.append(f"image #{len(images) + 1} = face photo")
            if c.body_b64:
                images.append(c.body_b64)
                attached.append(f"image #{len(images) + 1} = full-body photo")
            if attached:
                rows.append(f"    attached: {', '.join(attached)}")
            blocks.append("\n".join(rows))
        blocks.append("\nALLOWED LABELS (use exactly one of these in `person`; never invent): "
                      + ", ".join(labels))
    else:
        blocks.append("\nREGISTERED PEOPLE: (none — this user's memory has no people yet;"
                      " `matched` must be [])")

    # 当前时间锚:与 rewrite_query 同一口径。没有它,模型看到照片里的季节/节日/招牌
    # 就可能把"上周""去年"锚错年份,而这道改写的产物会直接成为 R0 的输入。
    if now_dt is not None:
        blocks.append(f"\nCURRENT TIME: {now_dt.isoformat()}"
                      " (the anchor for any relative time in the question or the image)")
    hist = "\n".join(f"{h}: {t}" for h, t in (history or [])) or "(no history)"
    blocks.append(f"\nRECENT DIALOGUE HISTORY:\n{hist}")
    blocks.append(f"\nTHE USER'S QUESTION:\n{query}")
    return "\n".join(blocks), images, labels


def enrich_query_with_image(
    deps: VisualDeps, *, query: str, image: bytes, content_type: str = "image/jpeg",
    history: list[tuple[str, str]] | None = None, scenario: str = "", now_dt: Any = None,
) -> VisualRewrite:
    """看图改写 query。**任何失败都返回原 query**(skipped 记原因),绝不抛、绝不阻塞召回。"""
    import base64

    frame = _to_frame(image)
    if frame is None:
        return VisualRewrite(query=query, skipped="图片解码失败")

    picks = _detect(deps, frame)
    cands = _candidates(deps, picks) if picks else []
    # 没脸也照做:图里的地点/物体/文字同样能把"那家店""这个东西"落成具体词(已与用户对齐)
    omni = deps.backends.get("mm_runner")
    if omni is None:
        return VisualRewrite(query=query, faces=len(picks), skipped="未装配 mm_runner")

    user, extra, labels = build_prompt(query, picks, cands, history, now_dt)
    sys = _SYSTEM + (f"\n\n# Caller scenario\n{scenario}" if scenario else "")
    images = [base64.b64encode(image).decode()] + extra
    try:
        raw = omni.chat(f"{sys}\n\n{user}", images_b64=images, max_tokens=1000, temperature=0.0)
        obj = json.loads(strip_fences(raw))
        resolved = str(obj.get("resolved") or "").strip() or query
        # 短标校验 + 回填真实 id:标签不在白名单的一律丢(防模型编人),对外仍给 character_id
        matched = [{**m, "character_id": labels[m["person"]]}
                   for m in (obj.get("matched") or [])
                   if isinstance(m, dict) and m.get("person") in labels]
        logger.info(f"视觉改写 faces={len(picks)} 候选={len(cands)} "
                    f"认到={[m.get('name') for m in matched]}\n"
                    f"  原问题={query!r}\n  改写后={resolved!r}")
        return VisualRewrite(query=resolved, faces=len(picks), matched=matched,
                             system=sys, user=user, raw=raw)
    except Exception as e:  # noqa: BLE001  看图失败退回原 query:少一层理解,但答得出来
        logger.warning(f"视觉改写失败,退回原 query: {e}")
        return VisualRewrite(query=query, faces=len(picks), system=sys, user=user,
                             raw=str(e), skipped=f"{type(e).__name__}: {e}")
