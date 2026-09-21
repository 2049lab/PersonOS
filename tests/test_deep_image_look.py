"""深轨看图单测:mock media_store + mllm,验证证据渲染时带 task_query 重看图片证据。

覆盖:
- 图片证据渲染时,带 d.task_query 调 mllm.look_image 并把结果拼进原话行;
- 看图目的(purpose)= 用户原始问题(task_query),非 keywords/topic;
- 纯文本证据不触发看图;
- 无 content_ref / 未注入 media_store|mllm / task_query 为空 → 不看图(降级读 content_inline);
- 看图返回空 / 抛异常 → 只展示原话,不拼补充行(不阻塞)。
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


# —— 核心:带 task_query 看图 ——

def test_image_evidence_looked_with_task_query():
    """图片证据:带 task_query 重看,补充行拼进渲染;purpose 用的是用户原始问题。"""
    ms, ml = FakeMediaStore(), FakeMllm(text="横幅显示 ArtScience Museum。")
    d = _deps(task_query="Caroline 那次画展办在哪", media_store=ms, mllm=ml)
    note = _look_image_note(d, _img_rec())
    assert "ArtScience Museum" in note
    assert ml.purposes == ["Caroline 那次画展办在哪"]     # 看图目的=原始问题,非 keywords/topic
    assert ms.read_keys == ["personos/u/2026/09/ab/abc.jpg"]


def test_evidence_page_appends_look_note_for_image():
    """evidence_page 传 d 时,图片行下方多一条针对性看图补充。"""
    ms, ml = FakeMediaStore(), FakeMllm(text="横幅:ArtScience Museum。")
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    body, _ = evidence_page([_img_rec()], page=1, d=d)
    assert "一张展馆照片" in body            # 原话(写入时的理解)保留
    assert "ArtScience Museum" in body       # 深轨针对性看图补充
    assert "↳[看图" in body


def test_text_evidence_not_looked():
    """纯文本证据不触发看图。"""
    ms, ml = FakeMediaStore(), FakeMllm()
    d = _deps(task_query="随便问", media_store=ms, mllm=ml)
    rec = EvidenceRecord(holder="user", content_inline="纯文本", modality="text", captured_at=_T)
    assert _look_image_note(d, rec) == ""
    assert ml.purposes == []


# —— 降级:任何缺失/失败都只读 content_inline ——

def test_no_task_query_no_look():
    ms, ml = FakeMediaStore(), FakeMllm()
    d = _deps(task_query="", media_store=ms, mllm=ml)
    assert _look_image_note(d, _img_rec()) == ""
    assert ml.purposes == []


def test_no_deps_no_look():
    """未注入 media_store/mllm(纯文本部署):图片证据只读 content_inline。"""
    d = _deps(task_query="画展在哪")           # 不给 media_store/mllm
    assert _look_image_note(d, _img_rec()) == ""


def test_mllm_unavailable_no_look():
    ms, ml = FakeMediaStore(), FakeMllm(available=False)
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    assert _look_image_note(d, _img_rec()) == ""


def test_no_content_ref_no_look():
    """有 modality=image 但无 content_ref(原图没留底):无法回看,降级。"""
    ms, ml = FakeMediaStore(), FakeMllm()
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    rec = EvidenceRecord(holder="user", content_inline="[图片] x", modality="image",
                         content_ref=None, captured_at=_T)
    assert _look_image_note(d, rec) == ""


def test_look_empty_returns_no_note():
    ms, ml = FakeMediaStore(), FakeMllm(text="")      # 看不出与目的相关
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    assert _look_image_note(d, _img_rec()) == ""


def test_look_exception_degrades():
    """读原图抛异常:不炸,返回空补充(原话仍照常展示)。"""
    ms, ml = FakeMediaStore(fail=True), FakeMllm()
    d = _deps(task_query="画展在哪", media_store=ms, mllm=ml)
    assert _look_image_note(d, _img_rec()) == ""


def test_evidence_page_without_d_no_look():
    """不传 d(旧调用方 / 快链):evidence_page 退化为纯原话渲染,不看图。"""
    body, _ = evidence_page([_img_rec()], page=1)
    assert "一张展馆照片" in body
    assert "↳[看图" not in body
