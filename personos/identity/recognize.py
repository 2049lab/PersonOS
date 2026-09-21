"""MLLM 身份仲裁协议(⑥c,批量单发 verdict)+ 证据入库/学云(enroll)。

一次调用批量判本 clip 每个未绑 cast:
- query 侧:每 cast 最清晰的脸 + 全身 + 外观描述 + 对白摘录(+ 中途评估 note)。
- 候选侧:每个已注册 character 一张卡 = 名字 + 外观 + 最清脸 + 全身(+ 声纹)。
- 输出空间:BIND|<cast>|<character_id> 或 BIND|<cast>|NEW。
- verdict 即终判(无轮转投票 / 无向量否决);非法输出按 NEW 处理并记 issue(仲裁失败=登记临时新档,
  从不阻塞,可离线合并)。

有意偏离(对齐 mneme arbiter.py 的取舍):候选卡用 personos 的 desc 字段(非 mneme text_profile dict);
不引 documentary profile(personos 场景是眼镜/机器人第一人称);素材 b64 由上游(ChainBook.query_card /
registry.candidate_card)从 OSS 取好再传入,本模块不碰 OSS。

enroll_evidence:把一个 cast 的证据落到某 character(crop 传 OSS、asset 入库、向量学进概率云),
判 NEW 建档后 enroll、判 match 后对既有档 enroll(commit.py 终审复用)。
"""

from __future__ import annotations

import base64
import re
from typing import Any

from loguru import logger

from personos.identity.cloud import CloudEngine
from personos.identity.store import CharacterStore
from personos.identity.types import CandidateCard, CastEvidence

NEW = "NEW"

_BIND_RE = re.compile(r"^BIND\|", re.IGNORECASE)

PROMPT_HEADER = """You are identifying people across separate recordings.
For each QUERY person (seen in the current recording) decide whether they are one of the
REGISTERED characters below, or a new person never registered before.

EVIDENCE — judge like a human, strongest signal first:
1. Face, build, body shape, hairstyle, age impression.
2. Voice samples (audio #N) when attached: a similar voice supports a match but never
   decides alone; listen especially when a person has no face photo.
3. A heard name matching a registered character supports a match but never decides alone
   (different people can share a name).
4. Clothing is the weakest signal — it may have changed between recordings; never rely
   on clothing alone.

DECISION POLICY:
- If you are not reasonably sure a query matches a registered character, answer NEW.
  An unnecessary NEW is recoverable later; merging two different people is not.
- People appearing together at the same time are physically distinct people — they can
  never be the same character.
- Candidate ids starting with "chain:" are OTHER query people from this same recording.
  Bind a query to a chain id ONLY if the two are the same person seen in separate,
  non-overlapping segments. Never bind a query to its own chain id.

Answer with protocol lines ONLY, one per query, then END.

FORMAT EXAMPLE (format only; use the actual QUERY ids and ALLOWED TARGETS you are given):
BIND|<query_id>|<target id copied verbatim from ALLOWED TARGETS>
BIND|<query_id>|NEW
END
"""


def build_arbitration_prompt(queries: list[dict[str, Any]], candidates: list[CandidateCard],
                             ) -> tuple[str, list[str], list[str]]:
    """queries: [{cast_id, desc, name, key_lines, note?, face_b64, body_b64, voice_b64?}]。
    返回 (prompt, images, audios)——image #N / audio #N 各自独立编号。"""
    images: list[str] = []
    audios: list[str] = []
    blocks: list[str] = [PROMPT_HEADER, "REGISTERED CHARACTERS:"]
    for card in candidates:
        rows = [f"CHARACTER {card.character_id}:"]
        if card.name:
            rows.append(f"    known name: {card.name}")
        if card.desc:
            rows.append(f"    appearance: {card.desc}")
        if card.last_seen_session:
            rows.append(f"    last seen: recording {card.last_seen_session}"
                        " (clothing may differ now)")
        attached = []
        if card.face_b64:
            images.append(card.face_b64); attached.append(f"image #{len(images)} = face photo")
        if card.body_b64:
            images.append(card.body_b64); attached.append(f"image #{len(images)} = full-body photo")
        if card.voice_b64:
            audios.append(card.voice_b64); attached.append(f"audio #{len(audios)} = voice sample")
        if attached:
            rows.append(f"    attached: {', '.join(attached)}")
        blocks.append("\n".join(rows))
    blocks.append("QUERIES (people in the current recording):")
    for query in queries:
        rows = [f"QUERY {query['cast_id']}:"]
        if query.get("name"):
            rows.append(f"    heard name: {query['name']}")
        if query.get("desc"):
            rows.append(f"    appearance: {query['desc']}")
        for line in (query.get("key_lines") or [])[:2]:
            rows.append(f"    said: {line}")
        if query.get("note"):
            rows.append(f"    note: {query['note']}")
        attached = []
        if query.get("face_b64"):
            images.append(query["face_b64"]); attached.append(f"image #{len(images)} = face photo")
        if query.get("body_b64"):
            images.append(query["body_b64"]); attached.append(f"image #{len(images)} = full-body photo")
        if query.get("voice_b64"):
            audios.append(query["voice_b64"]); attached.append(f"audio #{len(audios)} = voice sample")
        if attached:
            rows.append(f"    attached: {', '.join(attached)}")
        blocks.append("\n".join(rows))
    # 观测到的失败模式:模型照抄格式示例、编造候选集外的 id,使 verdict 作废。显式白名单 + 声明
    # 无候选时 NEW 是唯一答案,钉死输出空间。
    if candidates:
        ids = ", ".join(card.character_id for card in candidates)
        blocks.append(
            "ALLOWED TARGETS (the target id in each BIND line must be copied"
            f" verbatim from this list, or be NEW): {ids}")
    else:
        blocks.append(
            "ALLOWED TARGETS: none registered — NEW is the only valid answer;"
            " output BIND|<query_id>|NEW for every query.")
    blocks.append("Output BIND lines now.")
    return "\n\n".join(blocks), images, audios


def parse_verdicts(raw: str, *, cast_ids: list[str], candidate_ids: list[str],
                   ) -> tuple[dict[str, str], list[str]]:
    """返回 ({cast_id: character_id|"NEW"}, issues)。缺/越界的 cast 一律回退 NEW
    (单发协议下唯一安全兜底:新档可逆,误认不可)。"""
    verdicts, issues, _defaulted = parse_verdicts_detailed(
        raw, cast_ids=cast_ids, candidate_ids=candidate_ids)
    return verdicts, issues


def parse_verdicts_detailed(raw: str, *, cast_ids: list[str], candidate_ids: list[str],
                            ) -> tuple[dict[str, str], list[str], set[str]]:
    """parse_verdicts 详版:额外返回被默认成 NEW 的 cast 集合。

    defaulted = verdict 非模型显式给出的 cast(缺 BIND 行 / 越白名单回退 NEW)——终审据此回退到链
    假设而非登记重复新档;显式 BIND|x|NEW 是真 verdict,不计入 defaulted。
    """
    verdicts: dict[str, str] = {}
    issues: list[str] = []
    defaulted: set[str] = set()
    valid_candidates = set(candidate_ids)
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or not _BIND_RE.match(line):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            issues.append(f"bad BIND line: {line!r}")
            continue
        cast_id, target = parts[1], parts[2]
        if cast_id not in cast_ids:
            issues.append(f"BIND for unknown query {cast_id!r}")
            continue
        if target.upper() == "NEW":
            verdicts[cast_id] = NEW
            defaulted.discard(cast_id)
        elif target in valid_candidates:
            verdicts[cast_id] = target
            defaulted.discard(cast_id)
        else:
            issues.append(f"BIND to unknown character {target!r} → NEW")
            verdicts[cast_id] = NEW
            defaulted.add(cast_id)
    for cast_id in cast_ids:
        if cast_id not in verdicts:
            issues.append(f"no verdict for {cast_id!r} → NEW")
            verdicts[cast_id] = NEW
            defaulted.add(cast_id)
    return verdicts, issues, defaulted


# ── enroll:证据落库 + 学云(建 NEW / 对既有 match 追加,commit 终审复用)──
def enroll_evidence(store: CharacterStore, cloud: CloudEngine, media_store: Any,
                    character_id: str, evidence: CastEvidence,
                    *, session_id: str = "", clip_index: int = 0) -> None:
    """把一个 cast 的证据落到 character:crop 传 OSS、asset 入库、向量学进概率云。"""
    def _save_img(b64: str, ct: str) -> str:
        if media_store is None or not b64:
            return ""
        try:
            return media_store.save_image(base64.b64decode(b64), owner=store.user_id,
                                          content_type=ct).key
        except Exception as e:  # noqa: BLE001  素材落 OSS 失败不阻塞(向量仍学)
            logger.warning(f"素材存 OSS 失败: {e}")
            return ""

    for f in evidence.faces:
        if f.embedding is None:
            continue
        store.add_asset(character_id, "face", quality=f.q, embedding=f.embedding,
                        payload={"oss_key": _save_img(f.crop_b64, "image/png"), "t": f.t,
                                 "session": session_id, "clip": clip_index,
                                 "descriptor": f.descriptor})
        cloud.learn(character_id, "face", f.embedding, f.q,
                    payload={"session": session_id, "clip": clip_index})
        body_key = _save_img(f.body_crop_b64, "image/jpeg")
        if body_key:
            store.add_asset(character_id, "body", quality=f.q, embedding=None,
                            payload={"oss_key": body_key, "t": f.t, "session": session_id,
                                     "clip": clip_index})
    for v in evidence.voices:
        if v.embedding is None:
            continue
        voice_key = ""
        if media_store is not None and v.wav_bytes:
            try:
                voice_key = media_store.save_audio(v.wav_bytes, owner=store.user_id)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"声纹 wav 存 OSS 失败: {e}")
        store.add_asset(character_id, "voice", quality=v.q, embedding=v.embedding,
                        payload={"oss_key": voice_key, "t0": v.t0, "t1": v.t1,
                                 "session": session_id, "clip": clip_index})
        cloud.learn(character_id, "voice", v.embedding, v.q,
                    payload={"session": session_id, "clip": clip_index})
