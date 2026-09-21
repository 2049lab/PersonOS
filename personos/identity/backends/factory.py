"""Assemble the video-identity backends: {mm_runner, face_detector, voiceprint}.

Three profiles, selected by ``PERSONOS_VIDEO_BACKEND``:

- ``none`` (**default**): video identity is switched off. Asking for it raises,
  with a message saying what to install and what to set.
- ``real``: InsightFace for faces, ECAPA for voiceprints, a multimodal LLM for
  the screenplay pass. Needs ``pip install personos[identity]`` and model
  weights. Missing dependencies fail loudly, on purpose.
- ``mock``: fake backends for tests. **Never use outside tests.**

Why ``mock`` is not the default, and why ``real`` is not either:

``mock`` *invents* screenplays and *generates* fake face vectors. Pointed at a
real database it writes fabricated memories and identities while everything
looks healthy from the outside. That happened once: a deployment without this
variable set defaulted to mock, and the only symptom was a video pipeline
quietly producing nonsense.

``real`` was the previous default, chosen precisely to avoid that. But for a
library installed from PyPI it is the wrong default too: the identity extra is
roughly 2 GB, so a text-only user who happens to pass a video would get an
import explosion from deep inside a worker thread instead of an explanation.

``none`` is the honest default: the feature is off until asked for, and asking
without the dependencies gets you a sentence telling you what to do.
"""

from __future__ import annotations

from typing import Any

_DISABLED_HINT = (
    "video identity is disabled (PERSONOS_VIDEO_BACKEND=none). To enable it: "
    "pip install 'personos[identity]' and set PERSONOS_VIDEO_BACKEND=real"
)


def make_backends(profile: str | None = None) -> dict[str, Any]:
    from personos.config import get_config

    profile = (profile or get_config().video_backend or "none").strip().lower()
    if profile in ("none", "off", "disabled"):
        raise RuntimeError(_DISABLED_HINT)
    if profile == "mock":
        from personos.identity.backends.mock import (
            MockFaceDetector, MockOmni, MockVoiceprint,
        )
        return {"mm_runner": MockOmni(), "face_detector": MockFaceDetector(),
                "voiceprint": MockVoiceprint()}
    if profile in ("real", "wearable"):
        from personos.identity.backends.real import make_real_backends
        return make_real_backends()
    raise ValueError(f"PERSONOS_VIDEO_BACKEND must be one of none/real/mock, got {profile!r}")


__all__ = ["make_backends"]
