"""Visual understanding rewrite on the recall side: correct person recognition plus
DEGRADATION ACROSS THE WHOLE PATH.

Degradation is this module's weak point: recall is a read path, and looking at the image is
only a bonus. A decode failure, no face, no candidates, the multimodal model being down, bad
JSON — every one of these must return the query unchanged and raise nothing, or the user ends
up with no answer at all just because an image could not be understood.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from personos.identity.types import CandidateCard
from personos.online.visual_query import VisualDeps, enrich_query_with_image

Q = "他和我上周干嘛去了?"


def _png(w=64, h=64) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


class _Det:
    """One detected face; the fields match the handful of backends.base.FaceDet attributes this
    module actually uses."""

    def __init__(self, box=(10, 10, 40, 40), q=0.9):
        self.bbox = box
        self.embedding = np.ones(8, dtype=np.float32) / np.sqrt(8)
        self.quality = q
        self.blur_score = q
        self.crop_b64 = "ZmFjZQ=="


class _Detector:
    def __init__(self, dets=None, boom=False):
        self._d = dets if dets is not None else []
        self._boom = boom

    def detect(self, frame):
        if self._boom:
            raise RuntimeError("推理挂了")
        return list(self._d)


class _Omni:
    def __init__(self, out="", boom=False):
        self.out, self.boom, self.calls = out, boom, []

    def chat(self, prompt, **kw):
        self.calls.append((prompt, kw))
        if self.boom:
            raise RuntimeError("MLLM 挂了")
        return self.out


class _Store:
    def __init__(self, chars):
        self._c = chars

    def list_active_characters(self, include_wearer=False):
        return list(self._c)

    def names_for(self, cid):
        return [c["primary_name"] for c in self._c if c["id"] == cid and c.get("primary_name")]


class _Registry:
    """Only candidate_card is needed, and it behaves exactly as in production (which passes a
    real AnchorRegistry)."""

    def __init__(self, store):
        self._s = store

    def candidate_card(self, character):
        return CandidateCard(character_id=character["id"],
                             name=character.get("primary_name") or "",
                             desc=(character.get("payload") or {}).get("text_profile", {})
                             .get("appearance", ""))


def _deps(chars=(), dets=None, omni_out="", omni_boom=False, det_boom=False):
    store = _Store(list(chars))
    return VisualDeps(store=store, cloud=None,
                      backends={"face_detector": _Detector(dets, det_boom),
                                "mm_runner": _Omni(omni_out, omni_boom)},
                      registry=_Registry(store))


CHARS = [{"id": "char_A", "primary_name": "李四",
          "payload": {"text_profile": {"appearance": "戴眼镜的男生"}}},
         {"id": "char_B", "primary_name": "王五", "payload": {}}]


def test_matched_face_rewrites_query_with_name():
    """The person is recognised, so the pronoun in the query is replaced with a concrete name
    and matched carries the character_id back."""
    out = '{"resolved": "李四上周和我干嘛去了?", "matched": [{"face_index": 0, "person": "p1", "name": "李四"}]}'
    d = _deps(CHARS, dets=[_Det()], omni_out=out)
    r = enrich_query_with_image(d, query=Q, image=_png(), history=[("user", "聊过李四")])
    assert r.query == "李四上周和我干嘛去了?"
    # The short label p1 is resolved back into a real character_id for the caller.
    assert r.matched == [{"face_index": 0, "person": "p1", "character_id": "char_A", "name": "李四"}]
    assert r.faces == 1 and not r.skipped


def test_unknown_face_must_not_be_given_a_name():
    """If it cannot be recognised, do not force a name onto it — misidentifying someone is
    worse than not identifying them, because retrieval then veers off onto the wrong person
    entirely."""
    d = _deps(CHARS, dets=[_Det()],
              omni_out='{"resolved": "照片里那个穿蓝夹克的男生和我上周干嘛去了?", "matched": []}')
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.matched == [] and "蓝夹克" in r.query


def test_hallucinated_label_is_dropped():
    """The model invented a short label outside the allow-list, so it is dropped and must not
    count as a successful recognition."""
    d = _deps(CHARS, dets=[_Det()],
              omni_out='{"resolved": "张三上周和我干嘛去了?", "matched": [{"face_index": 0, "person": "p9", "name": "张三"}]}')
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.matched == [], "a short label outside the allow-list must be dropped"


def test_real_character_id_never_reaches_the_model():
    """A real character_id (a 26-character ULID) must never appear in the prompt — project
    convention is that the LLM only ever sees short labels.

    Long ids make the model miscopy or hallucinate; short labels are easy to copy and can be
    checked mechanically, and the code fills the real id back in after parsing.
    """
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    enrich_query_with_image(d, query=Q, image=_png())
    prompt, _kw = d.backends["mm_runner"].calls[0]
    assert "char_A" not in prompt and "char_B" not in prompt, "a real id leaked to the model"
    assert "PERSON p1:" in prompt and "ALLOWED LABELS" in prompt


def test_no_face_still_rewrites_from_scene():
    """Do it even when there is no face in the image: a place, an object or some text can
    resolve the reference into concrete words just as well (agreed with the user)."""
    d = _deps(CHARS, dets=[], omni_out='{"resolved": "那家川菜馆我上周去过吗?", "matched": []}')
    r = enrich_query_with_image(d, query="这家店我上周去过吗?", image=_png())
    assert r.faces == 0 and r.query == "那家川菜馆我上周去过吗?"
    assert d.backends["mm_runner"].calls, "the multimodal model should be called even with no face"


def test_empty_library_still_rewrites_but_matches_nothing():
    """An empty character library (a new user): the image is still looked at and the query
    rewritten, but recognising anyone is impossible."""
    d = _deps([], dets=[_Det()], omni_out='{"resolved": "照片里的男生和我上周干嘛去了?", "matched": []}')
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.matched == [] and r.query.startswith("照片里")


# -- Degradation: every case below must return the query unchanged and raise nothing --

def test_broken_image_falls_back():
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "不该用到"}')
    r = enrich_query_with_image(d, query=Q, image=b"not-an-image")
    assert r.query == Q and "decode failed" in r.skipped
    assert not d.backends["mm_runner"].calls, \
        "the image never even decoded, so do not spend money calling the multimodal model"


def test_face_detector_crash_falls_back_to_no_face_path():
    """Local inference crashed, so carry on as if no face was detected (the scene information
    can still drive the rewrite) rather than taking recall down."""
    d = _deps(CHARS, det_boom=True, omni_out='{"resolved": "改写了", "matched": []}')
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.faces == 0 and r.query == "改写了"


def test_mllm_crash_falls_back():
    d = _deps(CHARS, dets=[_Det()], omni_boom=True)
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.query == Q and r.skipped


def test_bad_json_falls_back():
    d = _deps(CHARS, dets=[_Det()], omni_out="这不是 JSON")
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.query == Q and r.skipped


def test_missing_mm_runner_falls_back():
    d = _deps(CHARS, dets=[_Det()])
    d.backends.pop("mm_runner")
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.query == Q and "mm_runner" in r.skipped


def test_user_image_is_always_image_one():
    """Numbering convention: the user's own image is always image #1 and attachments start at
    #2, which is how the textual references in the prompt line up with the right picture."""
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    img = _png()
    enrich_query_with_image(d, query=Q, image=img)
    _prompt, kw = d.backends["mm_runner"].calls[0]
    assert kw["images_b64"][0] == base64.b64encode(img).decode()
    assert "THE USER'S IMAGE: image #1" in _prompt
    assert "image #2" in _prompt, "attachment numbering should start at 2"


def test_history_and_allowed_ids_reach_the_prompt():
    """The history context (video episodes included) and the candidate allow-list must actually
    make it into the prompt, or recognition and rewriting have nothing to go on."""
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    enrich_query_with_image(d, query=Q, image=_png(),
                            history=[("video", "李四和我上周去爬山了")])
    prompt, _kw = d.backends["mm_runner"].calls[0]
    assert "李四和我上周去爬山了" in prompt
    assert "ALLOWED LABELS" in prompt and "p1" in prompt
    assert Q in prompt


def test_current_time_anchor_reaches_the_prompt():
    """The current-time anchor must reach the prompt, on the same terms as rewrite_query.

    Without it, a season, a holiday or a shop sign visible in the photo can make the model
    anchor "last week" or "last year" to the wrong year. The output of this rewrite feeds
    straight into R0, so an error here propagates all the way down.
    """
    from datetime import datetime, timezone

    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    t = datetime(2026, 9, 18, 14, 30, tzinfo=timezone.utc)
    enrich_query_with_image(d, query=Q, image=_png(), now_dt=t)
    prompt, _kw = d.backends["mm_runner"].calls[0]
    assert "CURRENT TIME: 2026-09-18T14:30" in prompt, prompt[-400:]


def test_no_now_dt_is_still_fine():
    """Omitting the time anchor does not blow up (older callers, or scripts calling directly);
    that one line is simply absent."""
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    r = enrich_query_with_image(d, query=Q, image=_png())
    prompt, _kw = d.backends["mm_runner"].calls[0]
    assert "CURRENT TIME" not in prompt and r.query == "x"
