"""后台在途任务的上限闸:满则拒(503),任务终态归还。

上游故障、任务普遍变慢时,202 无限接单会让线程池队列无界膨胀(每项都攥着
请求 payload 和闭包)——这是"用着用着内存吃穿"的最后防线。独立小模块、
零 import 副作用,单测可直接引入(不触发 runtime 单例/连库)。
"""
from __future__ import annotations

import threading

_MAX_PENDING = 200   # 后台在途任务上限(排队+执行中)


class TaskOverloaded(RuntimeError):
    """后台在途已满:调用方应快速失败/稍后重试,而不是继续堆积。"""


class AdmissionGate:
    """在途名额闸(threading.BoundedSemaphore 语义封装):满则拒,任务终态归还。

    跨线程释放(提交线程占名额、worker 归还);归还次数超占用会炸 ValueError,
    恰好是"每条路径都归还"的自检。"""

    def __init__(self, cap: int = _MAX_PENDING):
        self._sem = threading.BoundedSemaphore(cap)

    def try_enter(self) -> bool:
        return self._sem.acquire(blocking=False)

    def leave(self) -> None:
        self._sem.release()
