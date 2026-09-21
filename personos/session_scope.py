"""入口层的 id 校验与调用方 session 改写(纯函数,不依赖 runtime,可离线单测)。

两件事:
- **id 卫生**:user_id / session_id / caller 都是「拼进 Redis key 与 MySQL 行」的
  调用方可控字符串。Redis key 用 ":" 做段分隔(seg/lock 键,redis_client.key),
  段内若允许 ":" 则 (u="a:b", s="c") 与 (u="a", s="b:c") 拼出同一个键 = 跨用户
  串读写未闭合段(上线前安全审查 P1)。这里在入口处用白名单字符集挡住;
  redis_client.key 另做段转义兜底(纵深防御,见其 docstring)。
- **调用方隔离**:不同调用方(接入记忆服务的 agent)可能各自用 "chat-001" 这类
  重复的 session_id。入口处把 session_id 改写为 f"{caller}:{session_id}",
  下游(write/recall/session_context/trace)只见改写后的 id,无感知差异;
  caller 缺省 = 不改写(旧调用方向前兼容)。
"""

from __future__ import annotations

import re

# caller:调用方标识,机器可读短标识(拼进 session_id 前缀,也进 MySQL 列)
RE_CALLER = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# 原始 session_id:客户端自选;禁 ":"(Redis key 分隔符)、"/"(OSS/路径歧义)
RE_SESSION = re.compile(r"^[A-Za-z0-9_.-]{1,95}$")
# user_id:注册时可选自选;与 MySQL VARCHAR(128) 对齐
RE_USER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# 改写后的 session_id 总长上限:caller(≤32) + ":" + 原始(≤95) = 128,对齐列宽
_SCOPED_MAX = 128


def valid_user_id(user_id: str | None) -> str | None:
    """校验自选 user_id;不合法抛 ValueError。None/空白 = 自动生成,合法。"""
    uid = (user_id or "").strip()
    if not uid:
        return None
    if not RE_USER.match(uid):
        raise ValueError(
            f"user_id 只允许字母/数字/下划线/连字符,长度 1-128(禁冒号、斜杠等): {uid[:40]!r}")
    return uid


def scoped_session(caller: str, session_id: str) -> str:
    """调用方改写:caller 非空 → f"{caller}:{session_id}";空 → 原样(向前兼容)。

    先校验 caller 与原始 session_id 的字符集(不合法抛 ValueError,入口转 400),
    再拼前缀——下游拿到的 id 保证无歧义且对齐 Redis key / MySQL 列宽。
    """
    caller = (caller or "").strip()
    if caller and not RE_CALLER.match(caller):
        raise ValueError(
            f"caller 只允许字母/数字/下划线/连字符,长度 1-32: {caller[:40]!r}")
    if not RE_SESSION.match(session_id or ""):
        raise ValueError(
            f"session_id 只允许字母/数字/下划线/连字符/点,长度 1-95(禁冒号): {(session_id or '')[:40]!r}")
    sid = f"{caller}:{session_id}" if caller else session_id
    if len(sid) > _SCOPED_MAX:
        raise ValueError(f"caller+session_id 总长超 {_SCOPED_MAX}: {len(sid)}")
    return sid
