"""Benchmark adapter unit tests: inlining the caption of an image message, matching the EverMemOS
evaluation convention.

The key invariant is that inlining happens at turn level, before same-speaker turns are merged, and
produces exactly one line per turn -- run_locomo aligns traces by zipping text.split("\n") with dia,
so the line count must not change.
"""

from __future__ import annotations

from scripts.bench.locomo import _group_turns


def test_image_turn_inlines_caption_as_own_line():
    """An image message with empty text turns its caption into its own line; after the same speaker's
    next turn is appended, the line count still equals the dia count."""
    turns = [
        {"speaker": "Janet", "text": "", "dia_id": "D1:3",
         "img_url": "http://x/1.jpg", "blip_caption": "a man riding a horse on a beach"},
        {"speaker": "Janet", "text": "look at this photo", "dia_id": "D1:4"},
    ]
    out = _group_turns(turns, speaker_a="Janet", speaker_b="Pam")
    assert len(out) == 1 and out[0].holder == "user"        # same speaker merged, speaker_a maps to user
    assert out[0].text == "[shared an image: a man riding a horse on a beach]\nlook at this photo"
    assert out[0].dia == "D1:3,D1:4"
    # The trace alignment invariant: line count == dia count
    assert len(out[0].text.split("\n")) == len(out[0].dia.split(","))


def test_image_with_text_and_missing_caption_falls_back():
    """An image that comes with its own text gets the marker as a prefix, and a missing blip_caption
    falls back to "an image"."""
    turns = [{"speaker": "Pam", "text": "so cute", "dia_id": "D1:5", "img_url": "http://x/2.jpg"}]
    out = _group_turns(turns, speaker_a="Janet", speaker_b="Pam")
    assert out[0].text == "[shared an image: an image] so cute"
    assert out[0].holder == "Pam"                            # speaker_b keeps their real name


def test_plain_text_turn_untouched():
    """A message without img_url passes through unchanged, with no marker injected."""
    turns = [{"speaker": "Janet", "text": "hello there", "dia_id": "D1:1"}]
    out = _group_turns(turns, speaker_a="Janet", speaker_b="Pam")
    assert out[0].text == "hello there"
