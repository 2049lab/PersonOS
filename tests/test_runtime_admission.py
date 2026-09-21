"""AdmissionGate(后台在途上限闸)语义:满则拒、终态归还、跨线程归还、多还即炸。

不 import runtime 单例(那会连真实 MySQL);闸是独立小类,直接测真实现。
"""
import threading

import pytest

from personos.app.admission import AdmissionGate, TaskOverloaded


def test_gate_fills_then_rejects():
    g = AdmissionGate(cap=3)
    assert all(g.try_enter() for _ in range(3))     # 3 个名额全占
    assert not g.try_enter()                        # 满:快速拒绝(非阻塞)


def test_gate_leave_makes_room():
    g = AdmissionGate(cap=2)
    g.try_enter(); g.try_enter()
    assert not g.try_enter()
    g.leave()
    assert g.try_enter()                            # 归还一个,名额回一个


def test_gate_cross_thread_release():
    """提交线程占名额、worker 线程归还——信号量无线程归属,跨线程归还必须成立。"""
    g = AdmissionGate(cap=1)
    assert g.try_enter()
    t = threading.Thread(target=g.leave)
    t.start(); t.join()
    assert g.try_enter()


def test_gate_double_release_explodes():
    """BoundedSemaphore 多还即 ValueError:恰好自检"每条路径只归还一次"。"""
    g = AdmissionGate(cap=1)
    g.try_enter(); g.leave()
    with pytest.raises(ValueError):
        g.leave()


def test_overloaded_is_runtime_error():
    """API 层按 TaskOverloaded 捕获转 503;它必须是 RuntimeError 子类。"""
    assert issubclass(TaskOverloaded, RuntimeError)
