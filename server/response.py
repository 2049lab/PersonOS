"""对外响应统一信封 {code, data, msg} + 中心化装配(不逐个改端点)。

约定(与调用方对齐):
- **成功**:HTTP 走 REST 语义(200/201/202),body `code=0`、`data=<业务载荷>`、`msg="ok"`。
- **失败**:HTTP 走语义状态码(4xx/5xx),body `code=<该 HTTP 状态号>`(HTTP 同族粗码)、
  `data=null`、`msg=<人可读原因>`。

中心化两件套(零改动各 handler):
1. `EnvelopeRoute`:自定义 APIRoute,把每个 handler **返回**的响应(成功 dict 或旧式
   `JSONResponse({"error": ...})`)统一包成信封;保留原状态码与响应头(如 Retry-After)。
2. `install(app)`:注册全局异常处理器,接管框架/依赖 **抛出** 的 HTTPException(401/404…)、
   请求校验错误(422)、以及未预期异常(500)——这些不经过 handler 的返回值,route 层兜不到。

两者只对 `/api/v1` 生效;平台探针(/healthz、/readyz)保持原样(k8s 不解析信封)。
公共构造器 `ResponseUtils.ok/error` 供内部与处理器共用(Java ResponseUtils 风格)。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

_API_PREFIX = "/api/v1"
CODE_OK = 0
_ENVELOPE_KEYS = {"code", "data", "msg"}


def _envelope(code: int, data: Any, msg: str) -> dict:
    return {"code": code, "data": data, "msg": msg}


class ResponseUtils:
    """公共响应构造器(Java ResponseUtils 风格):内部与异常处理器共用一份信封逻辑。"""

    @staticmethod
    def ok(data: Any = None, msg: str = "ok", *, status_code: int = 200,
           headers: dict | None = None) -> JSONResponse:
        return JSONResponse(status_code=status_code, headers=headers,
                            content=_envelope(CODE_OK, data, msg))

    @staticmethod
    def error(http_status: int, msg: str, *, code: int | None = None,
              data: Any = None, headers: dict | None = None) -> JSONResponse:
        """失败信封。code 缺省 = HTTP 状态号(HTTP 同族粗码);msg 为人可读原因。"""
        return JSONResponse(status_code=http_status, headers=headers,
                            content=_envelope(code if code is not None else http_status, data, msg))


def _is_enveloped(body: Any) -> bool:
    return isinstance(body, dict) and set(body.keys()) == _ENVELOPE_KEYS


def _err_msg(body: Any, fallback: str) -> str:
    """从旧式错误体里取人可读原因:优先 error / detail / msg,取不到用 fallback。"""
    if isinstance(body, dict):
        for k in ("error", "detail", "msg", "message"):
            v = body.get(k)
            if isinstance(v, str) and v.strip():
                return v
    if isinstance(body, str) and body.strip():
        return body
    return fallback


def _wrap_response(resp) -> Any:
    """把 handler 返回的响应包成信封。非 JSON / 已是信封 → 原样返回。

    保留原状态码与响应头(Retry-After 等限流头必须透传);成功 code=0,失败 code=HTTP 状态号。
    """
    body_bytes = getattr(resp, "body", None)
    media = (getattr(resp, "media_type", "") or "").lower()
    if body_bytes is None or ("json" not in media and media != ""):
        return resp                                  # 流式/非 JSON:不动
    try:
        body = json.loads(body_bytes) if body_bytes else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return resp                                  # 无法解析:原样
    if _is_enveloped(body):
        return resp                                  # 已是信封:不二次包
    status = resp.status_code
    # 剥 content-length/type,由新 JSONResponse 重算;其余头(Retry-After 等)透传
    headers = {k: v for k, v in resp.headers.items()
               if k.lower() not in ("content-length", "content-type")}
    if status < 400:
        return ResponseUtils.ok(data=body, status_code=status, headers=headers or None)
    return ResponseUtils.error(status, _err_msg(body, "请求失败"), headers=headers or None)


class EnvelopeRoute(APIRoute):
    """自定义 APIRoute:handler 正常返回的响应统一包信封(抛出的异常由全局处理器接管)。"""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def custom(request: Request):
            return _wrap_response(await original(request))

        return custom


# —— 全局异常处理器:接管框架/依赖抛出的异常(handler 返回值兜不到的路径)——

def _http_exc_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
    if not request.url.path.startswith(_API_PREFIX):
        return JSONResponse(status_code=exc.status_code, content={"detail": detail},
                            headers=getattr(exc, "headers", None))   # 非业务接口:保持原样
    return ResponseUtils.error(exc.status_code, detail or "请求失败",
                               headers=getattr(exc, "headers", None))


def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errs = exc.errors()
    first = errs[0] if errs else {}
    loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
    msg = f"请求参数校验失败: {loc or '?'} {first.get('msg', '')}".strip()
    if not request.url.path.startswith(_API_PREFIX):
        return JSONResponse(status_code=422, content={"detail": errs})
    return ResponseUtils.error(422, msg)


def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"未处理异常 path={request.url.path}: {exc!r}")
    if not request.url.path.startswith(_API_PREFIX):
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    return ResponseUtils.error(500, "服务内部错误")   # 不外泄堆栈/内部细节(已落日志)


def install(app: FastAPI) -> None:
    """在 app 上注册三类全局异常处理器(仅 /api/v1 返信封,其余保持默认)。"""
    app.add_exception_handler(StarletteHTTPException, _http_exc_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
