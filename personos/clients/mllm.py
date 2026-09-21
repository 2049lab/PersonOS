"""MLLM 客户端:带目的看图,产出事实文本(图片输入的唯一看图点)。

核心原则:MLLM **永远带着目的看图**,不漫无目的描述整张图。目的(purpose)由调用方给:
- 写入侧:目的 = 图片周围的对话上下文;产出的文本写进 evidence.content_inline,
  之后 W2 的文本 LLM 照常从文本里抽 atom——图片记忆由此并入统一文本流,判链/召回一视同仁;
- 检索侧(深轨):目的 = 用户的原始问题;产出的文本直接返回给 agent(agent 感知不到底层是图)。

两处都只产**文本**,不直接产 atom:写入侧的 atom 仍由 W2 统一抽,保证图文同构、不分叉管线。
看图失败一律降级(返回空串),绝不阻塞写入或检索。

鉴权与 MaaS chat 一致(`api-key` 头);endpoint/model/key 皆可配(见 config)。
"""

from __future__ import annotations

import base64
import time

import httpx
from loguru import logger

from .. import obs
from ..config import Settings, settings

# OpenAI 兼容多模态消息:content 是 [ {type:text,...}, {type:image_url,...} ]
_LOOK_SYSTEM = """You describe ONE image as factual text, guided by a PURPOSE.

# Iron rule: look with purpose, do not narrate the whole image
Report ONLY what serves the PURPOSE. Ignore everything unrelated. If the image shows nothing
relevant to the purpose, reply with exactly "(无相关信息)". Never invent details you cannot see;
state only what is visually verifiable — text/signs on the image, countable objects, the scene,
who/what is shown.

# Output
Plain factual sentences (no markdown, no preamble, in the same language as the purpose). Prefer
concrete, verifiable statements over vague description. Quote on-image text literally when present."""


class MllmClient:
    """看图客户端。endpoint 是完整 URL(非 base_url+path),鉴权用 api-key 头。"""

    def __init__(self, cfg: Settings = settings, max_retries: int = 2):
        self.cfg = cfg
        self.max_retries = max_retries
        # trust_env=False:内网网关,禁读系统代理(同 MaasClient 的理由)
        self._client = httpx.Client(trust_env=False)

    @property
    def available(self) -> bool:
        """未配 key 时不可用;调用方应据此降级(纯文本链路不依赖本客户端)。"""
        return bool(self.cfg.mllm_key)

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", "api-key": self.cfg.mllm_key}
        attempt = 0
        while True:
            try:
                resp = self._client.post(
                    self.cfg.mllm_endpoint, headers=headers, json=payload,
                    timeout=httpx.Timeout(self.cfg.mllm_timeout, connect=10.0))
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError("5xx", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status and 400 <= status < 500:   # 客户端错误不重试
                    logger.error(f"MLLM 4xx 不重试: {status} {e.response.text[:200]}")
                    raise
                attempt += 1
                if attempt >= self.max_retries:
                    raise
                logger.warning(f"MLLM 第{attempt}次失败({e}),退避重试")
                time.sleep(0.5 * attempt)

    _EMPTY = "(无相关信息)"

    def _chat(self, image_data_url: str, purpose: str) -> str:
        payload = {
            "model": self.cfg.mllm_model,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 1000,
            "messages": [
                {"role": "system", "content": _LOOK_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"PURPOSE: {purpose}\n\nDescribe what in the image serves this purpose."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ]},
            ],
        }
        with obs.observation("mllm.look_image", as_type="generation", model=self.cfg.mllm_model,
                             input=purpose, metadata={"max_tokens": 1000}) as gen:
            data = self._post(payload)
            content = data["choices"][0]["message"]["content"]
            obs.update(gen, output=content)
            return content

    def look_image(
        self, image: bytes, purpose: str, *,
        content_type: str = "image/jpeg",
    ) -> str:
        """带目的看一张图,产出事实文本。

        image: 原图字节(调用方从 OSS read_bytes 或上传时的 bytes 拿到)。
        purpose: 看图目的(对话上下文 / 用户问题)——决定 MLLM 关注什么。
        返回 "" 表示看不出与目的相关的信息,或调用失败(降级,不抛)。
        """
        if not self.available:
            logger.warning("MLLM key 未配置,跳过看图(图片理解降级)")
            return ""
        if not image or not purpose.strip():
            return ""
        b64 = base64.b64encode(image).decode()
        data_url = f"data:{content_type};base64,{b64}"
        try:
            raw = self._chat(data_url, purpose.strip())
        except Exception as e:   # noqa: BLE001  看图失败绝不阻塞主链路
            logger.warning(f"MLLM 看图失败,返回空: {e}")
            return ""
        text = (raw or "").strip()
        # 去 markdown 围栏残留;判空(模型按约定对无关图回「(无相关信息)」)
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        if not text or text == self._EMPTY:
            return ""
        return text


# 进程级单例(与 settings 同生命周期);测试可自行构造 MllmClient(cfg=...)
mllm = MllmClient()
