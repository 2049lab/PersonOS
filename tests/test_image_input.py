"""图片输入写入侧单测:mock media_store + mllm,验证带图 feed 的行为。

覆盖:
- 带图消息:图存储 + 看图文本进 content_inline + modality 正确 + 看图 purpose 带上下文;
- 图文混排:用户配文与图片理解文本合并,modality=mixed;
- 纯图无配文:modality=image;
- 看图产文本后,W2 照常从文本抽 atom(图片记忆并入统一文本流);
- 降级:mllm 不可用 / 看图返回空 / 存储失败 —— 均不阻塞写入,纯文本链路语义不变;
- 未注入 media_store/mllm 时(纯文本部署)带图也不炸。
"""

from __future__ import annotations

from datetime import datetime

from personos.online.write_path import SessionWriter
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.evidence_store import EvidenceStore

from .fakes import FakeEmbedder

_T0 = datetime(2026, 8, 25, 10, 0)

# —— 复用 test_write_path 的路由式假 LLM 响应 ——
BOUNDARY_KEEP = '{"should_end": false, "confidence": 0.8, "topic_summary": "健身"}'
BOUNDARY_END = '{"should_end": true, "confidence": 0.9, "topic_summary": "健身"}'
EPISODE_OK = ('{"topic": "用户在健身房练腿", '
              '"episode": "用户今天在健身房练腿,照片显示腿举器械。", "domains": ["D05"]}')
ATOMS_OK = ('{"atoms": [{"text": "用户今天在健身房练腿", "object_type": "event", '
            '"holder": "user", "kind": "K06", "domains": ["D05"], "when": "2026-08-25", '
            '"quote": "今天练腿"}]}')


class RoutingLLM:
    """按 system prompt 特征路由到 boundary/episode/atoms 三队列(仿 test_write_path)。"""

    def __init__(self, boundary=(), episode=(), atoms=()):
        self.q = {"boundary": list(boundary), "episode": list(episode), "atoms": list(atoms)}
        self.calls: list[str] = []

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        sysp = messages[0]["content"]
        kind = ("boundary" if "boundary detector" in sysp
                else "episode" if "episode weaver" in sysp
                else "atoms" if "atomic-memory extractor" in sysp else "other")
        assert kind != "other", f"未知 system prompt: {sysp[:40]}"
        self.calls.append(kind)
        assert self.q[kind], f"未预期的 {kind} 调用(队列已空)"
        resp = self.q[kind].pop(0)
        return resp(messages[-1]["content"]) if callable(resp) else resp


class FakeStored:
    def __init__(self, key, sha):
        self.key, self.sha256 = key, sha
        self.content_type, self.byte_size = "image/jpeg", 3


class FakeMediaStore:
    """记录存了什么,返回可预期的 key/sha。"""

    def __init__(self, fail=False):
        self.fail = fail
        self.saved: list[tuple[bytes, str]] = []

    def save_image(self, data, *, owner, content_type="image/jpeg"):
        if self.fail:
            raise RuntimeError("oss down")
        self.saved.append((data, owner))
        return FakeStored(key=f"personos/{owner}/deadbeef.jpg", sha="deadbeef")


class FakeMllm:
    """记录看图的 purpose;按预设返回文本(空串=看不出/失败降级)。"""

    def __init__(self, text="图片显示腿举器械和杠铃片。", available=True):
        self.text = text
        self._available = available
        self.purposes: list[str] = []

    @property
    def available(self):
        return self._available

    def look_image(self, image, purpose, *, content_type="image/jpeg"):
        self.purposes.append(purpose)
        return self.text


class Env:
    def __init__(self, db):
        self.ev = EvidenceStore(db)
        self.cells = CellStore(db)
        self.atoms = AtomStore(db)

    def writer(self, llm, *, media_store=None, mllm=None, session_id="t", max_turns=30):
        return SessionWriter(llm, FakeEmbedder(), self.ev, self.cells, self.atoms,
                             session_id=session_id, user_id="u-img", max_turns=max_turns,
                             media_store=media_store, mllm=mllm)


IMG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


# —— 带图写入:存储 + 看图 + 落库 ——

def test_image_stored_and_understood_into_content_inline(db):
    """带图配文:存 OSS、看图文本并进 content_inline、modality=mixed、content_ref/sha 落库。"""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(text="图片显示腿举器械。")
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP])
    # 段首句:无界可判(不调 boundary),但仍会存图+看图
    w = env.writer(llm, media_store=ms, mllm=ml)
    r = w.feed("user", "今天练腿", now_dt=_T0, image=IMG)

    rec = env.ev.get(r.evidence_id)
    assert rec.modality == "mixed"                    # 有配文 + 有图
    assert rec.content_ref == "personos/u-img/deadbeef.jpg"
    assert rec.sha256 == "deadbeef"
    assert "今天练腿" in rec.content_inline            # 用户配文保留
    assert "腿举器械" in rec.content_inline            # 图片理解文本并入
    assert ms.saved and ms.saved[0][1] == "u-img"     # 存图带 owner


def test_pure_image_no_caption_modality_image(db):
    """纯图无配文:modality=image,content_inline 只含图片理解文本。"""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(text="一张海边日落的照片。")
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    assert rec.modality == "image"
    assert "海边日落" in rec.content_inline


def test_look_image_purpose_carries_segment_context(db):
    """看图 purpose 带上本段已有对话上下文(带目的看图,非漫无目的描述)。"""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm()
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP, BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    w.feed("user", "我最近在减脂", now_dt=_T0)                       # 段首,建立上下文
    w.feed("user", "看我今天的训练", now_dt=_T0, image=IMG)          # 第二句带图
    assert ml.purposes, "应调用过看图"
    # 第二句看图时,purpose 里应带上第一句的上下文
    assert "减脂" in ml.purposes[-1]


def test_image_text_flows_into_w2_atoms(db):
    """图片理解文本进 content_inline 后,段闭合时 W2 照常从文本抽 atom(统一文本流)。"""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(text="照片显示腿举器械。")
    # 段首带图 → 再来一句触发边界闭合 → W2 建 cell 抽 atom
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm, media_store=ms, mllm=ml)
    w.feed("user", "今天练腿", now_dt=_T0, image=IMG)                # 段首句(不调 boundary)
    r = w.feed("user", "换个话题,晚饭吃啥", now_dt=_T0)              # 触发闭合
    assert r.closed_cell is not None
    atoms = env.atoms.list_by_cell(r.closed_cell.cell.id)
    assert len(atoms) >= 1                                          # W2 从(含图片文本的)段抽出 atom
    assert llm.calls.count("episode") == 1 and llm.calls.count("atoms") == 1


# —— 降级:任何一环失败都不阻塞写入 ——

def test_mllm_unavailable_degrades_to_text(db):
    """MLLM 不可用:仍存图(留底),但无理解文本,content_inline 只有配文,写入不炸。"""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(available=False)
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "看这张图", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    assert rec.content_ref == "personos/u-img/deadbeef.jpg"        # 原图仍留底
    assert rec.content_inline == "看这张图"                         # 无理解文本,只配文
    assert rec.modality == "mixed"


def test_mllm_returns_empty_no_image_text_appended(db):
    """看图返回空(看不出与目的相关):不追加图片文本,只保留配文。"""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(text="")
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "随手拍的", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    assert rec.content_inline == "随手拍的"
    assert rec.content_ref == "personos/u-img/deadbeef.jpg"


def test_storage_failure_still_understands(db):
    """存储失败:原图不留底(content_ref 空),但看图理解仍进行,写入不阻塞。"""
    env = Env(db)
    ms, ml = FakeMediaStore(fail=True), FakeMllm(text="图片显示一只猫。")
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "我的猫", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    assert rec.content_ref is None                                  # 没留底
    assert "一只猫" in rec.content_inline                           # 但理解不丢
    assert rec.modality == "mixed"                                 # 有配文+有图(存储失败不改这点)


def test_no_media_deps_image_ignored_gracefully(db):
    """未注入 media_store/mllm(纯文本部署):带图不炸,退化为纯文本证据。"""
    env = Env(db)
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]))            # 不给 media_store/mllm
    r = w.feed("user", "带了图但服务没开图片能力", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    assert rec.modality == "mixed"                                 # 有配文+有图(标记有图,即便没能力处理)
    assert rec.content_ref is None
    assert rec.content_inline == "带了图但服务没开图片能力"


# —— 纯文本零影响:不带图时行为与改动前完全一致 ——

def test_text_only_unchanged(db):
    """不带图:modality=text、content_ref 空,不触碰 media_store/mllm。"""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm()
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "纯文字消息", now_dt=_T0)
    rec = env.ev.get(r.evidence_id)
    assert rec.modality == "text" and rec.content_ref is None
    assert rec.content_inline == "纯文字消息"
    assert ms.saved == [] and ml.purposes == []                    # 无图不调图片依赖
