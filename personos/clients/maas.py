"""MAAS 客户端:chat 补全 + 文本向量 + 精排打分。带超时/重试/结构化日志(dev 原则 §4/§9)。

注意 chat 与 embedding 的鉴权头不同:
- chat:      header `api-key` + `x-maas-app-id`
- embedding: header `Authorization: Bearer <key>`
- rerank:    同 chat(api-key 头;rerank_key 未配时复用 chat_key)
"""

from __future__ import annotations

import random
import time

import httpx
import numpy as np
from loguru import logger

from .. import obs
from ..config import Settings, settings


def _usage(data: dict) -> dict | None:
    """从 OpenAI 兼容响应取 token 用量 → langfuse usage_details(值必须数字,非数字/缺失过滤掉)。"""
    u = (data or {}).get("usage") or {}
    out = {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens"),
           "total": u.get("total_tokens")}
    out = {k: v for k, v in out.items() if isinstance(v, int)}
    return out or None


def _jitter(base: float) -> float:
    """退避加随机抖动(×0.5~1.5):并发下固定退避 = 同时睡同时醒,一齐重发再撞限流。"""
    return base * random.uniform(0.5, 1.5)


class MaasClient:
    """超时按接口分档:chat 宽(非流式长生成)/ embed+rerank 紧(快接口);connect 一律 10s。

    显式传 timeout 则全程覆盖该档(历史脚本在用);默认读 Settings(env MAAS_CHAT_TIMEOUT/MAAS_IO_TIMEOUT)。
    """

    def __init__(self, cfg: Settings = settings, timeout: float | None = None, max_retries: int = 3,
                 rate_limit_attempts: int = 2):
        self.cfg = cfg
        self.timeout = timeout
        self.max_retries = max_retries
        # 429 预算收紧为 2 次尝试:旧深预算(6 次/62s 睡眠)在并发下会让所有请求
        # 同步挂等再一齐重发,worker 被睡满 = 重试地狱;快速抛出让 API 层透传 429
        # 才是正解。评测要深预算在构造处显式传参(bench 外层另有 per-question 重试兜底)
        self.rate_limit_attempts = rate_limit_attempts
        # trust_env=False:MAAS 是纯内网网关,禁读 http_proxy 等环境变量——本地代理
        # (Clash 默认 7897)开过系统代理后,继承到变量的会话会把内网请求转给代理,
        # 代理不认内网 → Connection refused;长评测被这种环境波动打断不可接受
        self._client = httpx.Client(trust_env=False)   # 客户端级不设超时,逐请求按接口给

    def _chat_timeout(self) -> float:
        return self.timeout if self.timeout is not None else self.cfg.llm_timeout

    def _io_timeout(self) -> float:
        return self.timeout if self.timeout is not None else self.cfg.io_timeout

    def _post(self, path: str, headers: dict, payload: dict, timeout: float) -> dict:
        """带重试的 POST;网络异常与 5xx 按 max_retries 退避,429 另给深预算(指数退避),其余 4xx 直接抛。"""
        url = f"{self.cfg.llm_base_url}{path}"
        attempt = 0   # 网络异常 / 5xx 计数
        rl = 0        # 429 计数(独立预算)
        while True:
            try:
                resp = self._client.post(url, headers=headers, json=payload,
                                         timeout=httpx.Timeout(timeout, connect=10.0))
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError("5xx", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 429:
                    rl += 1
                    if rl >= self.rate_limit_attempts:
                        logger.error(f"MAAS {path} 429 重试预算 {self.rate_limit_attempts} 次耗尽,"
                                     f"原样抛出(API 层透传 429 / rerank 退 Noop)")
                        raise
                    # 短退避 + 抖动:并发下固定间隔会让被限流的请求同时醒来重发,再撞限流
                    wait = _jitter(min(1.0 * (2 ** (rl - 1)), 4.0))
                    logger.warning(f"MAAS {path} 429 限流,退避 {wait:.1f}s 重试"
                                   f"({rl}/{self.rate_limit_attempts})")
                    time.sleep(wait)
                    continue
                if status and 400 <= status < 500:  # 客户端错误不重试
                    logger.error(f"MAAS {path} 4xx 不重试: {status} {e.response.text[:200]}")
                    raise
                attempt += 1
                if attempt >= self.max_retries:
                    raise
                logger.warning(f"MAAS {path} 第{attempt}次失败({e};上限{timeout:g}s),退避重试")
                time.sleep(_jitter(0.5 * attempt))

    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 10240) -> str:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.cfg.llm_api_key,
            "x-maas-app-id": self.cfg.llm_app_id,
            "x-maas-user-email": "",
        }
        payload = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with obs.observation(obs.current_stage() or "maas.chat", as_type="generation",
                             model=self.cfg.llm_model, input=messages,
                             metadata={"stage": obs.current_stage() or "chat",
                                       "temperature": temperature, "max_tokens": max_tokens}) as gen:
            data = self._post("/chat/completions", headers, payload, timeout=self._chat_timeout())
            content = data["choices"][0]["message"]["content"]
            obs.update(gen, output=content, usage=_usage(data))   # token 必须数字,_usage 已过滤
            logger.debug(f"chat 返回 {len(content)} 字符")
            return content

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量取向量,返回 shape=(n, dim) 的 float32 矩阵。"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.effective_embedding_api_key}",
        }
        payload = {"model": self.cfg.embedding_model, "input": texts, "encoding_format": "float"}
        with obs.observation("maas.embed", as_type="embedding", model=self.cfg.embedding_model,
                             metadata={"n_texts": len(texts)}) as gen:
            data = self._post("/embeddings", headers, payload, timeout=self._io_timeout())
            vecs = [np.asarray(item["embedding"], dtype=np.float32) for item in data["data"]]
            obs.update(gen, output={"n_vectors": len(vecs)}, usage=_usage(data))
            logger.debug(f"embed {len(texts)} 条 -> dim={vecs[0].shape[0] if vecs else 0}")
            return np.vstack(vecs)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """精排打分:text_1=query × text_2=候选列表 → 与 documents 等长同序的相关性分。

        鉴权同 chat(api-key 头;rerank_key 未配时复用 chat_key——网关同一套)。
        """
        if not documents:
            return []
        headers = {
            "Content-Type": "application/json",
            "api-key": self.cfg.rerank_api_key or self.cfg.llm_api_key,
            "x-maas-app-id": self.cfg.llm_app_id,
            "x-maas-user-email": "",
        }
        payload = {"model": self.cfg.rerank_model, "text_1": query, "text_2": documents}
        with obs.observation("maas.rerank", as_type="span", model=self.cfg.rerank_model,
                             input=query, metadata={"n_docs": len(documents)}) as gen:
            data = self._post("/score", headers, payload, timeout=self._io_timeout())
            by_index = {item["index"]: float(item["score"]) for item in data["data"]}
            scores = [by_index.get(i, 0.0) for i in range(len(documents))]
            obs.update(gen, output={"top_score": max(scores) if scores else 0.0})
            logger.debug(f"rerank {len(documents)} 条 -> top={max(scores):.3f}")
            return scores

    def close(self) -> None:
        self._client.close()
