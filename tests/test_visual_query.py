"""召回侧视觉理解改写:认人正确性 + **全链路降级**。

降级是这个模块的命门:召回是读路径,看图只是锦上添花。解码失败/无脸/无候选/MLLM 挂/
JSON 坏 —— 任何一条都必须原样返回 query 且不抛,否则用户会因为"图没看懂"而彻底答不出来。
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
    """一张检出的脸(字段对齐 backends.base.FaceDet 里被本模块用到的那几个)。"""

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
    """只需 candidate_card —— 与生产同口径(生产传的是真 AnchorRegistry)。"""

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
    """认出人 → query 里的"他"换成具体名字,matched 带回 character_id。"""
    out = '{"resolved": "李四上周和我干嘛去了?", "matched": [{"face_index": 0, "person": "p1", "name": "李四"}]}'
    d = _deps(CHARS, dets=[_Det()], omni_out=out)
    r = enrich_query_with_image(d, query=Q, image=_png(), history=[("user", "聊过李四")])
    assert r.query == "李四上周和我干嘛去了?"
    # 短标 p1 → 回填成真实 character_id 给调用方
    assert r.matched == [{"face_index": 0, "person": "p1", "character_id": "char_A", "name": "李四"}]
    assert r.faces == 1 and not r.skipped


def test_unknown_face_must_not_be_given_a_name():
    """认不出就别硬套名字 —— 认错人比不认人更糟(检索会整个跑偏到别人身上)。"""
    d = _deps(CHARS, dets=[_Det()],
              omni_out='{"resolved": "照片里那个穿蓝夹克的男生和我上周干嘛去了?", "matched": []}')
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.matched == [] and "蓝夹克" in r.query


def test_hallucinated_label_is_dropped():
    """模型编了个白名单外的短标 → 丢掉,不得当成认人成功。"""
    d = _deps(CHARS, dets=[_Det()],
              omni_out='{"resolved": "张三上周和我干嘛去了?", "matched": [{"face_index": 0, "person": "p9", "name": "张三"}]}')
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.matched == [], "白名单外的短标必须被丢弃"


def test_real_character_id_never_reaches_the_model():
    """真实 character_id(26 位 ULID)不得出现在 prompt 里 —— 项目规范:LLM 只看短标。

    长 id 让模型抄错/幻觉;短标好抄又能机械校验,解析后由代码回填真实 id。
    """
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    enrich_query_with_image(d, query=Q, image=_png())
    prompt, _kw = d.backends["mm_runner"].calls[0]
    assert "char_A" not in prompt and "char_B" not in prompt, "真实 id 泄漏给了模型"
    assert "PERSON p1:" in prompt and "ALLOWED LABELS" in prompt


def test_no_face_still_rewrites_from_scene():
    """图里没人脸也照做:地点/物体/文字同样能把指代落成具体词(已与用户对齐)。"""
    d = _deps(CHARS, dets=[], omni_out='{"resolved": "那家川菜馆我上周去过吗?", "matched": []}')
    r = enrich_query_with_image(d, query="这家店我上周去过吗?", image=_png())
    assert r.faces == 0 and r.query == "那家川菜馆我上周去过吗?"
    assert d.backends["mm_runner"].calls, "无脸也应该调 MLLM"


def test_empty_library_still_rewrites_but_matches_nothing():
    """人物库为空(新用户):仍看图改写,但绝不可能认出人。"""
    d = _deps([], dets=[_Det()], omni_out='{"resolved": "照片里的男生和我上周干嘛去了?", "matched": []}')
    r = enrich_query_with_image(d, query=Q, image=_png())
    assert r.matched == [] and r.query.startswith("照片里")


# —— 降级:以下每一条都必须原样返回 query 且不抛 ——

def test_broken_image_falls_back():
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "不该用到"}')
    r = enrich_query_with_image(d, query=Q, image=b"not-an-image")
    assert r.query == Q and "解码" in r.skipped
    assert not d.backends["mm_runner"].calls, "图都没解开,不该再花钱调 MLLM"


def test_face_detector_crash_falls_back_to_no_face_path():
    """本地推理挂了 → 当作没检测到脸继续(仍可靠场景信息改写),不炸召回。"""
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
    """编号约定:用户原图恒为 image #1,附图从 #2 起 —— prompt 里的文字引用靠它对上图。"""
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    img = _png()
    enrich_query_with_image(d, query=Q, image=img)
    _prompt, kw = d.backends["mm_runner"].calls[0]
    assert kw["images_b64"][0] == base64.b64encode(img).decode()
    assert "THE USER'S IMAGE: image #1" in _prompt
    assert "image #2" in _prompt, "附图编号应从 2 起"


def test_history_and_allowed_ids_reach_the_prompt():
    """历史上下文(含视频 episode)与候选白名单必须真的进 prompt —— 否则认人/改写没依据。"""
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    enrich_query_with_image(d, query=Q, image=_png(),
                            history=[("video", "李四和我上周去爬山了")])
    prompt, _kw = d.backends["mm_runner"].calls[0]
    assert "李四和我上周去爬山了" in prompt
    assert "ALLOWED LABELS" in prompt and "p1" in prompt
    assert Q in prompt


def test_current_time_anchor_reaches_the_prompt():
    """当前时间锚必须进 prompt —— 与 rewrite_query 同口径。

    没有它,模型看到照片里的季节/节日/招牌就可能把"上周""去年"锚错年份,
    而这道改写的产物会直接成为 R0 的输入,错了会一路传下去。
    """
    from datetime import datetime, timezone

    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    t = datetime(2026, 9, 18, 14, 30, tzinfo=timezone.utc)
    enrich_query_with_image(d, query=Q, image=_png(), now_dt=t)
    prompt, _kw = d.backends["mm_runner"].calls[0]
    assert "CURRENT TIME: 2026-09-18T14:30" in prompt, prompt[-400:]


def test_no_now_dt_is_still_fine():
    """不传时间锚也不炸(老调用方/脚本直调),只是少这一行。"""
    d = _deps(CHARS, dets=[_Det()], omni_out='{"resolved": "x", "matched": []}')
    r = enrich_query_with_image(d, query=Q, image=_png())
    prompt, _kw = d.backends["mm_runner"].calls[0]
    assert "CURRENT TIME" not in prompt and r.query == "x"
