"""公司 Redis 接入:redinfra 服务发现连接池。

- 懒加载:import 本模块不触网;首次 get_redis() 才建池(EDS 发现 corvus-<cluster>)。
  服务/测试只要不真正用 Redis,就不需要 EDS 环境变量。
- key 一律经 key() 构造:"{环境}:{应用}:{其余段}"。环境取平台注入的 XHS_ENV
  (本地不设 → local,天然与线上键隔离),前缀使共享集群里的键可区分、可按段清理。
- 约束:所有 key 必须带 TTL(共享集群不留常驻键)。
- 依赖:redis-py 必须锁 6.x——redinfra 0.1.19 与 8.x 的握手协议不兼容(requirements 已锁)。
"""

from __future__ import annotations

import os
import threading

from ..config import settings

_lock = threading.Lock()
_client = None   # 惰性单例;进程内共享一个连接池(后台线程自动刷新实例列表)


def get_redis():
    """取 Redis 客户端单例(首次调用建池:EDS 服务发现 + corvus 直连)。失败如实抛。"""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                import redis as _redis
                from redinfra.redis.pool import DiscoveryBlockingConnectionPool
                # redinfra import 期的 init_logger() 会 logger.remove() 掀翻我们的全部
                # sink(只留它的 stdout xray sink)——建池后立即补挂文件 sink,保住落盘
                from ..logging_setup import reinstall_file_sink
                reinstall_file_sink(settings.log_dir)
                _client = _redis.Redis(
                    connection_pool=DiscoveryBlockingConnectionPool(
                        cluster_name=settings.redis_cluster))
    return _client


def _esc(part: str) -> str:
    """段内转义:":"→"%3A"、"%"→"%25"(顺序:先 % 后 :,防二次替换)。"""
    return part.replace("%", "%25").replace(":", "%3A")


def key(*parts: str) -> str:
    """规范化 key:f"{env}:personos:{parts...}",如 sit:personos:seg:u1:s1。

    各段先做分隔符转义再拼接:段内容是调用方可控字符串(user_id/session_id),
    不转义时 (u="a:b", s="c") 与 (u="a", s="b:c") 会拼出同一个键 = 跨用户串
    读写(seg 键存对话原文,最高敏)。入口层另有字符白名单(session_scope),
    这里是纵深兜底——就算某条路径漏校验,也拼不出碰撞键。
    """
    env = os.environ.get("XHS_ENV") or "local"
    return f"{env}:personos:" + ":".join(_esc(p) for p in parts)
