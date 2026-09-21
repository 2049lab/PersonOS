"""Omni 多模态 runner:走 xhs MAAS(OpenAI 兼容,api-key 头),支持 video_url 直取。

调用格式对齐 mneme backends/mm_qwen_omni.py 的 HTTP 路径:content 放 video_url(签名 URL,
**不下载字节**——传原始字节反而有大小上限)+ 可选 image_url(人脸 crop 标签,粗格子归属用)
+ text(prompt)。transport 复用 personos mllm 的 httpx + api-key(clients/mllm.py 同款)。
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from loguru import logger

from personos import obs
from personos.config import Settings, settings

# 视频剧本调用**不能**共用文本 LLM 的超时(MAAS_MLLM_TIMEOUT 默认 120s):送进去的是整段 clip,
# 模型侧还要自己去拉几十~上百 MB 视频再逐帧过,分钟级是常态——实测 60s/120帧 就要 109s,
# 2min clip 必然顶穿 120s,表现为读超时→重试→判毒消息→**这条 clip 的记忆彻底丢**。
# 这里给视频路径单独一个宽裕的上限;真正防呆靠 MAX_CLIP_DURATION_S 卡住过长 clip。
VIDEO_TIMEOUT_S = float(os.environ.get("PERSONOS_VIDEO_MLLM_TIMEOUT", "600"))


class ContentRejectedError(RuntimeError):
    """内容审查拒绝(data_inspection_failed);上层可据此跳过该 clip。"""


class MediaUnfetchableError(RuntimeError):
    """模型服务侧**拉不动我们给的媒体 URL**(它自己下载超时/失败)。

    实测:2min @7.3Mbps(106MB)的 clip 必现 `Download multimodal file timed out`,同样时长
    压到 1.6Mbps(23MB)则正常。属于"这个文件本身过大"的确定性失败——重试同一个 URL 只会
    再烧一个下载窗口,故单独成类,由上层转成 ClipRejected 留痕跳过,而不是当瞬时故障重试。
    """


class OmniRunner:
    """mm_runner 真后端:qwen3.5-omni-plus。chat(prompt, video_url=...) → 文本。"""

    def __init__(self, cfg: Settings = settings, max_retries: int = 2) -> None:
        self.cfg = cfg
        self.max_retries = max_retries
        self._client = httpx.Client(trust_env=False)   # 内网网关,禁读系统代理
        self.model = cfg.mllm_model

    @property
    def available(self) -> bool:
        return bool(self.cfg.mllm_key)

    def chat(self, prompt: str, *, video_url: str | None = None,
             images_b64: list[str] | None = None, audio_b64_list: list[str] | None = None,
             max_tokens: int = 8192, temperature: float = 0.0,
             video_fps: float | None = None) -> str:
        content: list[dict[str, Any]] = []
        if video_url:
            item: dict[str, Any] = {"type": "video_url", "video_url": {"url": video_url}}
            if video_fps is not None:
                item["fps"] = float(video_fps)          # MAAS 只认 video_url 同级的 fps
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
            # 带视频的调用走长超时;纯图/纯文(仲裁、终审复核)仍用文本口径,快失败快重试
            data = self._post(payload, timeout_s=VIDEO_TIMEOUT_S if video_url
                              else self.cfg.mllm_timeout)
            out = data["choices"][0]["message"]["content"] or ""
            obs.update(gen, output=out)
            return out

    def _post(self, payload: dict, *, timeout_s: float | None = None) -> dict:
        headers = {"Content-Type": "application/json", "api-key": self.cfg.mllm_key}
        attempt = 0
        while True:
            try:
                resp = self._client.post(
                    self.cfg.mllm_endpoint, headers=headers, json=payload,
                    timeout=httpx.Timeout(timeout_s or self.cfg.mllm_timeout, connect=10.0))
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError("5xx", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                body = getattr(getattr(e, "response", None), "text", "") or ""
                if "data_inspection_failed" in body:
                    raise ContentRejectedError(f"Omni 内容审查拒绝: {body[:200]}") from e
                if "Download multimodal file" in body:
                    raise MediaUnfetchableError(f"模型侧拉取媒体失败: {body[:300]}") from e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status and 400 <= status < 500:      # 客户端错误不重试
                    # 截 200 字会把上游的真实原因切掉(实测 MAAS 的 detail 前缀就占一百多字),
                    # 这类"不重试"的终态错误必须留全,否则只能靠再复现一次才能定位。
                    logger.error(f"Omni 4xx 不重试: {status} {body[:1500]}")
                    raise
                attempt += 1
                if attempt >= self.max_retries:
                    raise
                logger.warning(f"Omni 第{attempt}次失败({e}),退避重试")
                time.sleep(0.5 * attempt)
