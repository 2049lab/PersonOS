"""Deep track image re-inspection: with a mocked media_store and mllm, check that rendering evidence re-examines
image evidence using task_query.

Covers:
- when rendering image evidence, mllm.look_image is called with d.task_query and the result is appended to the
  original-text line;
- the look-up purpose is the user's original question (task_query), not keywords or the topic;
- plain text evidence never triggers a look;
- no content_ref, no injected media_store or mllm, or an empty task_query means no look, degrading to
  content_inline alone;
- an empty look result or a raised exception shows only the original text with no appended line, and never blocks.
"""

from __future__ import annotations

from datetime import datetime

from personos.models import EvidenceRecord
from personos.online.deep_recall import DeepDeps, _look_image_note, evidence_page
from personos.online.rerank import NoopReranker

_T = datetime(2026, 8, 25, 10, 0)


class FakeMediaStore:
    def __init__(self, data=b"\xff\xd8\xffimg", fail=False):
        self.data, self.fail = data, fail
        self.read_keys: list[str] = []

    def read_bytes(self, key):
        self.read_keys.append(key)
        if self.fail:
            raise RuntimeError("oss read fail")
        return self.data


class FakeMllm:
    def __init__(self, text="图中横幅显示 ArtScience Museum。", available=True):
        self.text, self._available = text, available
        self.purposes: list[str] = []

    @property
    def available(self):
        return self._available

    def look_image(self, image, purpose, *, content_type="image/jpeg"):
        self.purposes.append(purpose)
        return self.text


def _deps(*, task_query="", media_store=None, mllm=None):
    return DeepDeps(embedder=None, reranker=NoopReranker(),
                    atoms=None, cells=None, evidence=None,
                    task_query=task_query, media_store=media_store, mllm=mllm)


def _img_rec(content_inline="[图片] 一张展馆照片", ref="personos/u/2026/09/ab/abc.jpg", modality="image"):
    return EvidenceRecord(holder="user", content_inline=content_inline, modality=modality,
                          content_ref=ref, captured_at=_T)


# -- Core: looking at the image with task_query --

def test_image_evidence_looked_with_task_query():
    """Image evidence is re-examined with task_query and the extra line is folded into the rendering; the purpose
    passed to the model is the user's original question.
    """
    ms, ml = FakeMediaStore(), FakeMllm(text="横幅显示 ArtScience Museum。")
    d = _deps(task_query="Caroline 那次画展办在哪", media_store=ms, mllm=ml)
    note = _look_image_note(d, _img_rec())
    assert "ArtScience Museum" in note
    assert ml.purposes == ["Caroline 那次画展办在哪"]     # the look purpose is the original question, not keywords or the topic
    assert ms.read_keys == ["personos/u/2026/09/ab/abc.jpg"]


def test_evidence_page_appends_look_note_for_image():
    """When evidence_page is given d, the image line gets an extra, question-specific look-up note beneath it."""
    ms, ml = FakeMediaStore(), FakeMllm(text="横幅:ArtScience Museum。")
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    body, _ = evidence_page([_img_rec()], page=1, d=d)
    assert "一张展馆照片" in body            # the original text, i.e. what was understood at write time, is kept
    assert "ArtScience Museum" in body       # the deep track's question-specific look-up note
    assert "↳[看图" in body


def test_text_evidence_not_looked():
    """Plain text evidence never triggers an image look."""
    ms, ml = FakeMediaStore(), FakeMllm()
    d = _deps(task_query="随便问", media_store=ms, mllm=ml)
    rec = EvidenceRecord(holder="user", content_inline="纯文本", modality="text", captured_at=_T)
    assert _look_image_note(d, rec) == ""
    assert ml.purposes == []


# -- Degrading: any missing piece or failure falls back to reading content_inline only --

def test_no_task_query_no_look():
    ms, ml = FakeMediaStore(), FakeMllm()
    d = _deps(task_query="", media_store=ms, mllm=ml)
    assert _look_image_note(d, _img_rec()) == ""
    assert ml.purposes == []


def test_no_deps_no_look():
    """With no media_store or mllm injected (a text-only deployment), image evidence only reads content_inline."""
    d = _deps(task_query="画展在哪")           # neither media_store nor mllm is supplied
    assert _look_image_note(d, _img_rec()) == ""


def test_mllm_unavailable_no_look():
    ms, ml = FakeMediaStore(), FakeMllm(available=False)
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    assert _look_image_note(d, _img_rec()) == ""


def test_no_content_ref_no_look():
    """modality is image but there is no content_ref, meaning the original image was never retained, so there is
    nothing to look back at and the render degrades.
    """
    ms, ml = FakeMediaStore(), FakeMllm()
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    rec = EvidenceRecord(holder="user", content_inline="[图片] x", modality="image",
                         content_ref=None, captured_at=_T)
    assert _look_image_note(d, rec) == ""


def test_look_empty_returns_no_note():
    ms, ml = FakeMediaStore(), FakeMllm(text="")      # the model sees nothing relevant to the purpose
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    assert _look_image_note(d, _img_rec()) == ""


def test_look_exception_degrades():
    """If reading the original image raises, nothing blows up: the note comes back empty and the original text is
    still shown as usual.
    """
    ms, ml = FakeMediaStore(fail=True), FakeMllm()
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    assert _look_image_note(d, _img_rec()) == ""


def test_evidence_page_without_d_no_look():
    """Without d (older callers, or the fast path), evidence_page degrades to rendering the original text only and
    does not look at images.
    """
    body, _ = evidence_page([_img_rec()], page=1)
    assert "一张展馆照片" in body
    assert "↳[看图" not in body
