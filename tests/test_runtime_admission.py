"""AdmissionGate semantics (the cap on in-flight background tasks): reject when full, release
on terminal state, release across threads, and blow up on double release.

This does not import the runtime singleton, since that would open a real database connection;
the gate is a small standalone class, so the real implementation is tested directly.
"""
import threading

import pytest

from personos.admission import AdmissionGate, TaskOverloaded


def test_gate_fills_then_rejects():
    g = AdmissionGate(cap=3)
    assert all(g.try_enter() for _ in range(3))     # all three slots taken
    assert not g.try_enter()                        # full: reject immediately, do not block


def test_gate_leave_makes_room():
    g = AdmissionGate(cap=2)
    g.try_enter(); g.try_enter()
    assert not g.try_enter()
    g.leave()
    assert g.try_enter()                            # release one, get one slot back


def test_gate_cross_thread_release():
    """The submitting thread takes the slot and the worker thread releases it. A semaphore has
    no thread ownership, so releasing across threads has to work."""
    g = AdmissionGate(cap=1)
    assert g.try_enter()
    t = threading.Thread(target=g.leave)
    t.start(); t.join()
    assert g.try_enter()


def test_gate_double_release_explodes():
    """BoundedSemaphore raises ValueError on an extra release, which is exactly the self-check
    for "every path releases exactly once"."""
    g = AdmissionGate(cap=1)
    g.try_enter(); g.leave()
    with pytest.raises(ValueError):
        g.leave()


def test_overloaded_is_runtime_error():
    """The API layer catches TaskOverloaded and turns it into a 503, so it has to be a subclass
    of RuntimeError."""
    assert issubclass(TaskOverloaded, RuntimeError)
