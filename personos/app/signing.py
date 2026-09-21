"""AK/SK 服务间验签(对齐《内部系统开发安全规范》3.3.3 自建 AK/SK 方案)。

- 算法 HMAC-SHA256(签名 hex);规范不强制统一算法,固定为此。
- 请求头:X-Access-Key(AK,明文)/ X-Timestamp(unix 秒)/ X-Nonce / X-Signature / X-Signed-Headers。
- 规范串覆盖:METHOD + path + 排序 query + 应签头 + body 的 SHA-256 + timestamp + nonce。
- 应签头「最小集」:content-type、x-user-token —— 请求里出现(非空)就必须纳入签名(防被剥离/篡改)。
- 服务端流程:字段齐 → 时间窗 → 查 AK → nonce 防重放 → 重建规范串 → 常量时间比对;失败对外统一「认证失败」,内部日志记真因。
- **无条件强制**:不留任何 env 旁路(避免"一个环境变量关掉鉴权"的安全气味);测试用 dependency_overrides 显式绕过。
- AK→SK 从环境变量 PERSONOS_AKSK_MAP(JSON)读并缓存;SK 不出网、不落库。

服务端验签与调用方签名共用 canonical_string/sign,构造上保证两侧一致(见 sign_request)。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time

from fastapi import HTTPException, Request
from loguru import logger

from ..config import get_secret, settings

_WINDOW = int(os.environ.get("PERSONOS_SIGN_WINDOW_S", "300"))   # 时间戳窗口(秒)
_MIN_SIGNED = ("content-type", "x-user-token")                  # 应签头最小集(出现即必签)

_aksk_cache: dict | None = None
_mem_nonce: dict[str, float] = {}                               # 无 Redis 时的内存兜底(单副本/测试)


class _SigError(Exception):
    """验签失败(内部原因;对外一律折叠成 401 认证失败)。"""


# —— 共享:规范串 + 签名(服务端与调用方共用,防漂移)——

def canonical_string(method: str, path: str, query_items, signed_headers: dict,
                     body: bytes, timestamp: str, nonce: str) -> str:
    """把请求规范化为待签字符串。query_items 为 (k,v) 列表;signed_headers 为 名→值。"""
    q = "&".join(f"{k}={v}" for k, v in sorted(query_items or []))
    hb = "\n".join(f"{k}:{(v or '').strip()}" for k, v in sorted(signed_headers.items()))
    body_hash = hashlib.sha256(body or b"").hexdigest()
    return "\n".join([method.upper(), path, q, hb, body_hash, str(timestamp), nonce])


def sign(sk: str, canonical: str) -> str:
    return hmac.new(sk.encode(), canonical.encode(), hashlib.sha256).hexdigest()


def sign_request(ak: str, sk: str, method: str, path: str, *,
                 query_items=None, headers: dict | None = None, body: bytes = b"") -> dict:
    """调用方用:给一次请求算出应带的签名头。headers 传本次实际发送的头(至少含 content-type/x-user-token)。"""
    ts = str(int(time.time()))
    nonce = secrets.token_hex(16)
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    signed = sorted(h for h in _MIN_SIGNED if hdrs.get(h))
    canon = canonical_string(method, path, list(query_items or []),
                             {h: hdrs.get(h, "") for h in signed}, body, ts, nonce)
    return {"X-Access-Key": ak, "X-Timestamp": ts, "X-Nonce": nonce,
            "X-Signature": sign(sk, canon), "X-Signed-Headers": ",".join(signed)}


# —— 服务端:AK→SK、nonce、验签 ——

def _load_aksk() -> dict:
    """AK→SK 映射(JSON),从 KMS/env 读。SK 只在内存,不外泄。

    只缓存「非空」结果:若首次加载时凭证尚未 provision(空),不缓存、下次重试 →
    补好凭证后无需重启即自愈(避免「先起服务后补 map」被永久缓存空 map 卡死)。
    """
    global _aksk_cache
    if _aksk_cache:                                # 非空才用缓存
        return _aksk_cache
    raw = get_secret("PERSONOS_AKSK_MAP", required=False)
    try:
        m = json.loads(raw) if raw else {}
    except Exception:                              # noqa: BLE001  格式坏 → 视为无凭证(拒绝一切)
        logger.error("PERSONOS_AKSK_MAP 解析失败,验签将全部拒绝")
        m = {}
    if m:
        _aksk_cache = m                            # 拿到非空才缓存
    return m


def _reset_cache() -> None:
    """测试用:清 AK/SK 与 nonce 缓存。"""
    global _aksk_cache
    _aksk_cache = None
    _mem_nonce.clear()


def _seen_nonce(ak: str, nonce: str, ttl: int) -> bool:
    """原子记录 AK+Nonce;已存在=重放返回 True。有 Redis 用 Redis(多副本),否则内存兜底。"""
    if settings.redis_cluster:
        try:
            from ..storage.redis_client import get_redis, key
            ok = get_redis().set(key("nonce", ak, nonce), "1", nx=True, ex=ttl)
            return not ok
        except Exception:                          # noqa: BLE001  Redis 异常 → 退内存(尽力而为)
            pass
    now = time.time()
    for k, exp in list(_mem_nonce.items()):
        if exp < now:
            _mem_nonce.pop(k, None)
    mk = f"{ak}:{nonce}"
    if mk in _mem_nonce:
        return True
    _mem_nonce[mk] = now + ttl
    return False


def _verify(method: str, path: str, query_items, headers, body: bytes) -> str:
    """核验一次请求,返回调用方 AK;任一步失败抛 _SigError。"""
    ak = headers.get("x-access-key", "")
    ts = headers.get("x-timestamp", "")
    nonce = headers.get("x-nonce", "")
    sig = headers.get("x-signature", "")
    signed_raw = headers.get("x-signed-headers", "")   # 可为空(无最小集头时);故不列入必填
    if not (ak and ts and nonce and sig):
        raise _SigError("缺少签名头")
    try:
        tsi = int(ts)
    except ValueError:
        raise _SigError("时间戳非法")
    if abs(time.time() - tsi) > _WINDOW:
        raise _SigError("时间戳超窗")
    signed = sorted({h.strip().lower() for h in signed_raw.split(",") if h.strip()})
    # 安全相关头若出现在请求里,必须纳入签名(防攻击者剥离后重放)
    for h in _MIN_SIGNED:
        if headers.get(h) and h not in signed:
            raise _SigError(f"应签头 {h} 未纳入签名")
    sk = _load_aksk().get(ak)
    if not sk:
        raise _SigError(f"未知或未启用 AK: {ak}")
    if _seen_nonce(ak, nonce, _WINDOW):
        raise _SigError("重放 nonce")
    canon = canonical_string(method, path, query_items,
                             {h: headers.get(h, "") for h in signed}, body, ts, nonce)
    if not hmac.compare_digest(sign(sk, canon), sig):
        raise _SigError("签名不匹配")
    return ak


async def verify_signature(request: Request) -> None:
    """FastAPI 依赖:挂在 /api/v1 router 上,**无条件**逐请求验签(不留任何 env 旁路)。

    测试需绕过时用 FastAPI 官方机制显式覆写:app.dependency_overrides[verify_signature]=lambda:None。
    """
    body = await request.body()                    # Starlette 缓存,不影响后续 Pydantic 解析
    try:
        ak = _verify(request.method, request.url.path,
                     list(request.query_params.multi_items()), request.headers, body)
    except _SigError as e:
        logger.warning(f"AK/SK 验签失败 path={request.url.path}: {e}")   # 内部记真因
        raise HTTPException(status_code=401, detail="authentication failed")          # 对外统一,不暴露 AK 是否存在
    request.state.caller_ak = ak
