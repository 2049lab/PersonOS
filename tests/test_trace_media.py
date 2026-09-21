"""Returning images along the provenance path: image evidence carries modality plus a signed media_url in recalled
memories and in trace output.

Covers:
- evidence_entries gives image evidence both modality and media_url, while plain text gets neither;
- with no media_store supplied, only modality is set and media_url is omitted;
- if sign_url raises, media_url is omitted and nothing blows up;
- _ev_dict, used by the trace pairs, follows the same rules;
- memory_view threads media_url all the way through to the evidence entries.
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
    evs = evidence_entries(_atom(["ev_img"]), store, None)   # no signer
    assert evs[0]["modality"] == "image"
    assert "media_url" not in evs[0]


def test_sign_failure_degrades():
    store = FakeEvStore([_img_ev()])
    evs = evidence_entries(_atom(["ev_img"]), store, FakeSigner(fail=True))
    assert evs[0]["modality"] == "image"
    assert "media_url" not in evs[0]        # signing failed, so only modality is set and no error escapes


def test_ev_dict_image_media_url():
    """The trace pairs go through _ev_dict, which attaches media_url under the same rules."""
    d = _ev_dict(_img_ev(), FakeSigner())
    assert d["modality"] == "image"
    assert d["media_url"].startswith("https://oss.example/")
    d2 = _ev_dict(_text_ev(), FakeSigner())
    assert "modality" not in d2


def test_memory_view_threads_media_url():
    """End to end through memory_view: the evidence entry for image evidence carries media_url."""
    store = FakeEvStore([_img_ev()])
    view = memory_view(_atom(["ev_img"]), store, FakeSigner())
    assert view["evidence"][0]["media_url"].startswith("https://oss.example/")
