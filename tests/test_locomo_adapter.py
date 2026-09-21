"""bench 适配器单测:图片消息 caption 内联(评测口径与 EverMemOS 对标)。

关键不变量:内联发生在 turn 级(同人合并之前)、单 turn 单行——run_locomo 的
trace 对齐按 text.split("\n") zip dia,行数不能变。
"""

from __future__ import annotations

from scripts.bench.locomo_adapter import _group_turns


def test_image_turn_inlines_caption_as_own_line():
    """图片消息(text 空)→ caption 成独立一行;同人下一 turn 追加后行数=dia 数。"""
    turns = [
        {"speaker": "Janet", "text": "", "dia_id": "D1:3",
         "img_url": "http://x/1.jpg", "blip_caption": "a man riding a horse on a beach"},
        {"speaker": "Janet", "text": "look at this photo", "dia_id": "D1:4"},
    ]
    out = _group_turns(turns, speaker_a="Janet", speaker_b="Pam")
    assert len(out) == 1 and out[0].holder == "user"        # 同人合并,speaker_a→user
    assert out[0].text == "[shared an image: a man riding a horse on a beach]\nlook at this photo"
    assert out[0].dia == "D1:3,D1:4"
    # trace 对齐不变量:行数 == dia 数
    assert len(out[0].text.split("\n")) == len(out[0].dia.split(","))


def test_image_with_text_and_missing_caption_falls_back():
    """图片自带文字 → 前缀拼接;缺 blip_caption → "an image" 兜底。"""
    turns = [{"speaker": "Pam", "text": "so cute", "dia_id": "D1:5", "img_url": "http://x/2.jpg"}]
    out = _group_turns(turns, speaker_a="Janet", speaker_b="Pam")
    assert out[0].text == "[shared an image: an image] so cute"
    assert out[0].holder == "Pam"                            # speaker_b 真名直传


def test_plain_text_turn_untouched():
    """无 img_url 的消息原样透传(不注入任何标记)。"""
    turns = [{"speaker": "Janet", "text": "hello there", "dia_id": "D1:1"}]
    out = _group_turns(turns, speaker_a="Janet", speaker_b="Pam")
    assert out[0].text == "hello there"
