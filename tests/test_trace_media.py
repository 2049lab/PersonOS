"""溯源返图单测:图片证据在 recall memories / trace 里带 modality + media_url(签名 URL)。

覆盖:
- evidence_entries:图片证据带 modality 与 media_url;纯文本不带;
- media_store 未给 → 只标 modality 不带 media_url;
- sign_url 抛异常 → 不带 media_url(不炸);
- _ev_dict(trace 的 pair 用)同规则;
- memory_view 端到端把 media_url 透进 evidence 条目。
"""

from __future__ import annotations

from datetime import datetime

from personos.online.views import memory_view
from personos.models import EvidenceRecord, EvidenceRef, MemoryAtom
from personos.online.trust import _ev_dict, evidence_entries

_T = datetime(2026, 8, 25, 10, 0)


class FakeEvStore:
    def __init__(self, recs):
        self._by_id = {r.id: r for r in recs}

    def get(self, eid):
        return self._by_id.get(eid)

    def reply_for(self, eid):
        return None


class FakeSigner:
    def __init__(self, fail=False):
        self.fail = fail
        self.signed: list[str] = []

    def sign_url(self, key):
        if self.fail:
            raise RuntimeError("sign fail")
        self.signed.append(key)
        return f"https://oss.example/{key}?sig=abc"


def _img_ev():
    return EvidenceRecord(id="ev_img", holder="user", content_inline="[图片] 展馆照片",
                          modality="image", content_ref="personos/u/2026/09/ab/abc.jpg",
                          captured_at=_T)


def _text_ev():
    return EvidenceRecord(id="ev_txt", holder="user", content_inline="纯文本",
                          modality="text", captured_at=_T)


def _atom(refs):
    return MemoryAtom(text="t", holder="user",
                      evidence_refs=[EvidenceRef(evidence_id=r) for r in refs])


def test_image_evidence_carries_media_url():
    store = FakeEvStore([_img_ev()])
    signer = FakeSigner()
    evs = evidence_entries(_atom(["ev_img"]), store, signer)
    assert evs[0]["modality"] == "image"
    assert evs[0]["media_url"] == "https://oss.example/personos/u/2026/09/ab/abc.jpg?sig=abc"
    assert signer.signed == ["personos/u/2026/09/ab/abc.jpg"]


def test_text_evidence_no_media_fields():
    store = FakeEvStore([_text_ev()])
    evs = evidence_entries(_atom(["ev_txt"]), store, FakeSigner())
    assert "modality" not in evs[0] and "media_url" not in evs[0]


def test_no_signer_marks_modality_only():
    store = FakeEvStore([_img_ev()])
    evs = evidence_entries(_atom(["ev_img"]), store, None)   # 无签名器
    assert evs[0]["modality"] == "image"
    assert "media_url" not in evs[0]


def test_sign_failure_degrades():
    store = FakeEvStore([_img_ev()])
    evs = evidence_entries(_atom(["ev_img"]), store, FakeSigner(fail=True))
    assert evs[0]["modality"] == "image"
    assert "media_url" not in evs[0]        # 签名失败:只标 modality,不炸


def test_ev_dict_image_media_url():
    """trace 的 pair 用 _ev_dict,同规则带 media_url。"""
    d = _ev_dict(_img_ev(), FakeSigner())
    assert d["modality"] == "image"
    assert d["media_url"].startswith("https://oss.example/")
    d2 = _ev_dict(_text_ev(), FakeSigner())
    assert "modality" not in d2


def test_memory_view_threads_media_url():
    """memory_view 端到端:图片证据 evidence 条目带 media_url。"""
    store = FakeEvStore([_img_ev()])
    view = memory_view(_atom(["ev_img"]), store, FakeSigner())
    assert view["evidence"][0]["media_url"].startswith("https://oss.example/")
