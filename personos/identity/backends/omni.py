"""Multimodal runner for the screenplay pass over a video clip.

OpenAI-compatible chat completions, with ``video_url`` in the content list. The
clip is passed **by URL, not by bytes**: inlining a clip runs into request size
limits long before a two-minute video does, and the model service fetches it
itself. That is also why local media storage needs a publicly reachable base
URL before video works — see ``storage/media/local.sign_url``.

Content layout: video_url (the clip) + optional image_url (face crops used as
labels during coarse attribution) + text (the prompt).
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from loguru import logger

from personos import obs
from personos.config import Config, get_config

# The screenplay call over video must **not** share the text LLM's timeout
# (120s by default). We hand the model an entire clip, and the model service then
# has to download tens to hundreds of MB itself and walk it frame by frame, so
# minutes is normal: measured, a 60s/120-frame clip already takes 109s, which
# means a 2-minute clip is certain to blow through 120s. The failure mode is
# read timeout -> retry -> the message gets judged poisonous -> **the memory for
# this clip is lost entirely**. So the video path gets its own generous ceiling,
# and the real guard against runaway clips is MAX_CLIP_DURATION_S.
VIDEO_TIMEOUT_S = float(os.environ.get("PERSONOS_VIDEO_MLLM_TIMEOUT", "600"))


class ContentRejectedError(RuntimeError):
    """The content filter rejected the request (data_inspection_failed).

    Callers use this to skip the clip.
    """


class MediaUnfetchableError(RuntimeError):
    """The model service **could not pull the media URL we gave it** — its own
    download timed out or failed.

    Measured: a 2-minute clip at 7.3Mbps (106MB) reliably produces
    `Download multimodal file timed out`, while the same duration re-encoded to
    1.6Mbps (23MB) goes through. This is a deterministic "the file itself is too
    big" failure, so retrying the same URL only burns another download window.
    It gets its own class precisely so callers turn it into a recorded
    ClipRejected and move on, rather than retrying it as a transient fault.
    """


class OmniRunner:
    """The real mm_runner backend, talking to a hosted omni-style model.

    chat(prompt, video_url=...) returns text.
    """

    def __init__(self, cfg: Config | None = None, max_retries: int = 2) -> None:
        self.cfg = cfg or get_config()
        self.max_retries = max_retries
        # trust_env=False: a browsing proxy inherited from the shell would
        # silently intercept these calls; providers are addressed explicitly.
        self._client = httpx.Client(trust_env=False)
        self.model = self.cfg.mllm_model

    @property
    def available(self) -> bool:
        return bool(self.cfg.mllm_api_key)

    def _endpoint(self) -> str:
        """Full URL override if configured, otherwise the standard path."""
        cfg = self.cfg
        return cfg.mllm_endpoint or f"{cfg.effective_mllm_base_url}/chat/completions"

    def chat(self, prompt: str, *, video_url: str | None = None,
             images_b64: list[str] | None = None, audio_b64_list: list[str] | None = None,
             max_tokens: int = 8192, temperature: float = 0.0,
             video_fps: float | None = None) -> str:
        content: list[dict[str, Any]] = []
        if video_url:
            item: dict[str, Any] = {"type": "video_url", "video_url": {"url": video_url}}
            if video_fps is not None:
                item["fps"] = float(video_fps)          # the service only reads fps as a sibling of video_url
            content.append(item)
        for b64 in audio_b64_list or []:
            content.append({"type": "input_audio",
                            "input_audio": {"data": f"data:audio/wav;base64,{b64}", "format": "wav"}})
        for b64 in images_b64 or []:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        payload = {"model": self.model, "stream": False, "temperature": temperature,
                   "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
        with obs.observation("omni.chat", as_type="generation", model=self.model,
                             input=prompt, metadata={"max_tokens": max_tokens,
                                                     "has_video": bool(video_url)}) as gen:
            # Calls carrying video take the long timeout. Image-only and text-only
            # calls (arbitration, final adjudication review) keep the text budget
            # so they fail fast and retry fast.
            data = self._post(payload, timeout_s=VIDEO_TIMEOUT_S if video_url
                              else self.cfg.mllm_timeout)
            out = data["choices"][0]["message"]["content"] or ""
            obs.update(gen, output=out)
            return out

    def _post(self, payload: dict, *, timeout_s: float | None = None) -> dict:
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.cfg.mllm_api_key}"}
        attempt = 0
        while True:
            try:
                resp = self._client.post(
                    self._endpoint(), headers=headers, json=payload,
                    timeout=httpx.Timeout(timeout_s or self.cfg.mllm_timeout, connect=10.0))
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError("5xx", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                body = getattr(getattr(e, "response", None), "text", "") or ""
                if "data_inspection_failed" in body:
                    raise ContentRejectedError(f"rejected by the content filter: {body[:200]}") from e
                if "Download multimodal file" in body:
                    raise MediaUnfetchableError(f"the model service failed to fetch the media: {body[:300]}") from e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status and 400 <= status < 500:      # client errors are not retried
                    # Truncating to 200 chars cuts off the upstream's actual reason
                    # (measured: the service's detail prefix alone runs past a
                    # hundred characters). Terminal, non-retried errors have to be
                    # logged in full, or diagnosing them means reproducing them.
                    logger.error(f"4xx, not retrying: {status} {body[:1500]}")
                    raise
                attempt += 1
                if attempt >= self.max_retries:
                    raise
                logger.warning(f"attempt {attempt} failed ({e}), backing off and retrying")
                time.sleep(0.5 * attempt)
