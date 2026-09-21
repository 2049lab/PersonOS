"""Unit tests for the image-input write path: a mock media_store plus a mock multimodal model,
verifying how a feed carrying an image behaves.

Covers:
- A message with an image: the image is stored, the image-understanding text goes into
  content_inline, modality is correct, and the look-at-image purpose carries context;
- Mixed image and text: the user's caption and the image-understanding text are merged, and
  modality is mixed;
- An image with no caption: modality is image;
- Once the image has produced text, W2 extracts atoms from that text as usual (image memory
  merges into the one unified text stream);
- Degradation: the multimodal model being unavailable / image understanding returning empty /
  storage failing — none of these block the write, and the text-only path keeps its semantics;
- With no media_store or multimodal model injected (a text-only deployment), sending an image
  still must not blow up.
"""

from __future__ import annotations

from datetime import datetime

from personos.online.write_path import SessionWriter
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.evidence_store import EvidenceStore

from .fakes import FakeEmbedder

_T0 = datetime(2026, 8, 25, 10, 0)

# -- Reuses the routing-style fake LLM responses from test_write_path --
BOUNDARY_KEEP = '{"should_end": false, "confidence": 0.8, "topic_summary": "健身"}'
BOUNDARY_END = '{"should_end": true, "confidence": 0.9, "topic_summary": "健身"}'
EPISODE_OK = ('{"topic": "用户在健身房练腿", '
              '"episode": "用户今天在健身房练腿,照片显示腿举器械。", "domains": ["D05"]}')
ATOMS_OK = ('{"atoms": [{"text": "用户今天在健身房练腿", "object_type": "event", '
            '"holder": "user", "kind": "K06", "domains": ["D05"], "when": "2026-08-25", '
            '"quote": "今天练腿"}]}')


class RoutingLLM:
    """Routes on features of the system prompt into the boundary / episode / atoms queues
    (mirrors test_write_path)."""

    def __init__(self, boundary=(), episode=(), atoms=()):
        self.q = {"boundary": list(boundary), "episode": list(episode), "atoms": list(atoms)}
        self.calls: list[str] = []

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        sysp = messages[0]["content"]
        kind = ("boundary" if "boundary detector" in sysp
                else "episode" if "episode weaver" in sysp
                else "atoms" if "atomic-memory extractor" in sysp else "other")
        assert kind != "other", f"unknown system prompt: {sysp[:40]}"
        self.calls.append(kind)
        assert self.q[kind], f"unexpected {kind} call (the queue is empty)"
        resp = self.q[kind].pop(0)
        return resp(messages[-1]["content"]) if callable(resp) else resp


class FakeStored:
    def __init__(self, key, sha):
        self.key, self.sha256 = key, sha
        self.content_type, self.byte_size = "image/jpeg", 3


class FakeMediaStore:
    """Records what was stored and returns a predictable key and sha."""

    def __init__(self, fail=False):
        self.fail = fail
        self.saved: list[tuple[bytes, str]] = []

    def save_image(self, data, *, owner, content_type="image/jpeg"):
        if self.fail:
            raise RuntimeError("oss down")
        self.saved.append((data, owner))
        return FakeStored(key=f"personos/{owner}/deadbeef.jpg", sha="deadbeef")


class FakeMllm:
    """Records the purpose it was asked to look at the image for, and returns preset text (an
    empty string means it could not tell, i.e. the degraded case)."""

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


# -- Writing with an image: store, understand, persist --

def test_image_stored_and_understood_into_content_inline(db):
    """An image with a caption: stored in object storage, the understanding text also goes into
    content_inline, modality is mixed, and content_ref plus sha are persisted."""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(text="图片显示腿举器械。")
    llm = RoutingLLM(boundary=[BOUNDARY_KEEP])
    # First sentence of a segment: there is no boundary to judge yet (boundary is not called),
    # but the image is still stored and understood.
    w = env.writer(llm, media_store=ms, mllm=ml)
    r = w.feed("user", "今天练腿", now_dt=_T0, image=IMG)

    rec = env.ev.get(r.evidence_id)
    assert rec.modality == "mixed"                    # both a caption and an image
    assert rec.content_ref == "personos/u-img/deadbeef.jpg"
    assert rec.sha256 == "deadbeef"
    assert "今天练腿" in rec.content_inline            # the user's caption is preserved
    assert "腿举器械" in rec.content_inline            # the image-understanding text is merged in
    assert ms.saved and ms.saved[0][1] == "u-img"     # the image is stored with its owner


def test_pure_image_no_caption_modality_image(db):
    """An image with no caption: modality is image and content_inline holds only the
    image-understanding text."""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(text="一张海边日落的照片。")
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    assert rec.modality == "image"
    assert "海边日落" in rec.content_inline


def test_look_image_purpose_carries_segment_context(db):
    """The look-at-image purpose carries the conversation so far in this segment, so the model
    looks with a goal rather than describing aimlessly."""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm()
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP, BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    w.feed("user", "我最近在减脂", now_dt=_T0)                       # segment start, establishes context
    w.feed("user", "看我今天的训练", now_dt=_T0, image=IMG)          # second message carries the image
    assert ml.purposes, "image understanding should have been called"
    # When looking at the image for the second message, the purpose should carry the context
    # from the first one.
    assert "减脂" in ml.purposes[-1]


def test_image_text_flows_into_w2_atoms(db):
    """Once the image-understanding text is in content_inline, W2 extracts atoms from that text
    as usual when the segment closes (the one unified text stream)."""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(text="照片显示腿举器械。")
    # An image at the start of a segment, then another message triggers the boundary close, so
    # W2 builds a cell and extracts atoms.
    llm = RoutingLLM(boundary=[BOUNDARY_END], episode=[EPISODE_OK], atoms=[ATOMS_OK])
    w = env.writer(llm, media_store=ms, mllm=ml)
    w.feed("user", "今天练腿", now_dt=_T0, image=IMG)                # segment start (boundary not called)
    r = w.feed("user", "换个话题,晚饭吃啥", now_dt=_T0)              # triggers the close
    assert r.closed_cell is not None
    atoms = env.atoms.list_by_cell(r.closed_cell.cell.id)
    # W2 extracted atoms from the segment, image-derived text included.
    assert len(atoms) >= 1
    assert llm.calls.count("episode") == 1 and llm.calls.count("atoms") == 1


# -- Degradation: a failure at any link must not block the write --

def test_mllm_unavailable_degrades_to_text(db):
    """The multimodal model is unavailable: the image is still stored (kept on file) but there
    is no understanding text, so content_inline holds only the caption and the write survives."""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(available=False)
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "看这张图", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    assert rec.content_ref == "personos/u-img/deadbeef.jpg"        # the original image is still on file
    assert rec.content_inline == "看这张图"                         # no understanding text, just the caption
    assert rec.modality == "mixed"


def test_mllm_returns_empty_no_image_text_appended(db):
    """Image understanding returns empty (nothing relevant to the purpose): append no image
    text, keep only the caption."""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm(text="")
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "随手拍的", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    assert rec.content_inline == "随手拍的"
    assert rec.content_ref == "personos/u-img/deadbeef.jpg"


def test_storage_failure_still_understands(db):
    """Storage fails: the original image is not kept on file (content_ref is empty), but image
    understanding still runs and the write is not blocked."""
    env = Env(db)
    ms, ml = FakeMediaStore(fail=True), FakeMllm(text="图片显示一只猫。")
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "我的猫", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    assert rec.content_ref is None                                  # nothing kept on file
    assert "一只猫" in rec.content_inline                           # but the understanding is not lost
    # Caption plus image; a storage failure does not change that.
    assert rec.modality == "mixed"


def test_no_media_deps_image_ignored_gracefully(db):
    """No media_store or multimodal model injected (a text-only deployment): an image does not
    blow up, it degrades to plain text evidence."""
    env = Env(db)
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]))            # no media_store, no mllm
    r = w.feed("user", "带了图但服务没开图片能力", now_dt=_T0, image=IMG)
    rec = env.ev.get(r.evidence_id)
    # Caption plus image: the presence of an image is recorded even with no ability to process it.
    assert rec.modality == "mixed"
    assert rec.content_ref is None
    assert rec.content_inline == "带了图但服务没开图片能力"


# -- Zero impact on text-only: without an image, behaviour is exactly what it was before --

def test_text_only_unchanged(db):
    """No image: modality is text, content_ref is empty, and neither media_store nor the
    multimodal model is touched."""
    env = Env(db)
    ms, ml = FakeMediaStore(), FakeMllm()
    w = env.writer(RoutingLLM(boundary=[BOUNDARY_KEEP]), media_store=ms, mllm=ml)
    r = w.feed("user", "纯文字消息", now_dt=_T0)
    rec = env.ev.get(r.evidence_id)
    assert rec.modality == "text" and rec.content_ref is None
    assert rec.content_inline == "纯文字消息"
    # With no image, the image dependencies are not called at all.
    assert ms.saved == [] and ml.purposes == []
