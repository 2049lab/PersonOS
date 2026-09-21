"""视频消费侧机制测(mock/spy,无真模型):验证队列分派/顺序/cursor/混合交错/并发不串。

不跑真身份管线——把 video_ingest.process_clip/finalize_video 换成 spy,只验消费侧 plumbing:
- kind 分派:video→process_clip(逐 clip)、session_end→writer.end_session + finalize_video、ingest→feed_batch;
- 混合会话顺序 文→视频→文→session_end + 交错多批 文→clips→文→clips→end;
- cursor 恰好==消息数(不丢不重);多 session 并发独立(各自 cursor/调用不串)。
真身份/记忆正确性见 scripts/video/verify_pipeline(real backends)。
"""

from __future__ import annotations

import os
import tempfile

import pytest

from personos.app.ingest_worker import SessionConsumer
from personos.online import video_ingest
from personos.storage.msg_queue import MemoryMsgQueue
from personos.storage.session_lock import MemorySessionLock


class _Writer:
    """假 SessionWriter:只记 feed_batch / end_session 调用。"""
    def __init__(self, log, sid):
        self.log, self.sid = log, sid

    def feed_batch(self, msgs, **k):
        self.log.append(("feed", self.sid, len(msgs)))

    def end_session(self, **k):
        self.log.append(("end", self.sid))


@pytest.fixture
def spy(monkeypatch):
    """spy 掉真身份管线(process_clip/finalize_video),只验分派。"""
    log = []
    monkeypatch.setattr(video_ingest, "process_clip",
                        lambda deps, *, session_id, clip_key="", clip_url="", **k:
                        log.append(("clip", session_id, clip_key or clip_url)))
    monkeypatch.setattr(video_ingest, "finalize_video",
                        lambda deps, *, session_id: log.append(("final", session_id)))
    return log


@pytest.fixture
def spy_deps(monkeypatch):
    """同上,但额外记录 deps(用于验跨 user 隔离:deps 必须属于该消息的 user)。"""
    log = []
    monkeypatch.setattr(video_ingest, "process_clip",
                        lambda deps, *, session_id, clip_key="", clip_url="", **k:
                        log.append(("clip", deps, session_id, clip_key or clip_url)))
    monkeypatch.setattr(video_ingest, "finalize_video",
                        lambda deps, *, session_id: log.append(("final", deps, session_id)))
    return log


def _consumer(log):
    mq = MemoryMsgQueue()
    sc = SessionConsumer(mq, MemorySessionLock(), lambda u, s: _Writer(log, s),
                         video_deps=lambda u: f"deps:{u}")
    return mq, sc


def _txt(speaker="user", text="hi"):
    return {"messages": [{"speaker": speaker, "text": text}]}


def _vid(*keys):
    """我方 OSS key 形态(endpoint 真实会把未填的 video_url 置 None,这里如实模拟)。"""
    return {"messages": [{"speaker": "user", "video_oss_key": k, "video_url": None,
                          "clip_index": i} for i, k in enumerate(keys)]}


def _vid_url(*urls):
    """外链形态:video_oss_key 为 None、video_url 有值——与 endpoint 产出的 payload 一致。
    (回归:曾因 dict.get('video_oss_key','') 返回 None 而 TypeError,整条消息被判毒消息跳过。)"""
    return {"messages": [{"speaker": "user", "video_oss_key": None, "video_url": u,
                          "clip_index": i} for i, u in enumerate(urls)]}


def test_mixed_session_dispatch_and_cursor(spy):
    """文本→视频clips→文本→session_end:分派正确 + cursor==消息数。"""
    mq, sc = _consumer(spy)
    U, S = "u1", "s1"
    mq.enqueue(U, S, _txt(text="开场白"), kind="ingest")
    mq.enqueue(U, S, _vid("oss/c0.mp4", "oss/c1.mp4"), kind="video")
    mq.enqueue(U, S, _txt(text="拍完聊两句"), kind="ingest")
    mq.enqueue(U, S, {"task_type": None}, kind="session_end")
    rep = sc.drain_session(U, S)
    assert rep.applied == 4
    assert spy == [("feed", S, 1), ("clip", S, "oss/c0.mp4"), ("clip", S, "oss/c1.mp4"),
                   ("feed", S, 1), ("end", S), ("final", S)]   # 视频逐 clip;end 后 finalize
    assert mq.cursor_get(U, S) == 4                            # 恰好消费 4 条


def test_interleaved_video_bursts(spy):
    """变态调用方 文→clips批1→文→clips批2→session_end:两批 clips 都进 process_clip,end 后一次 finalize。"""
    mq, sc = _consumer(spy)
    U, S = "u2", "s2"
    mq.enqueue(U, S, _txt(text="t1"), kind="ingest")
    mq.enqueue(U, S, _vid("k0.mp4"), kind="video")
    mq.enqueue(U, S, _txt(text="t2"), kind="ingest")
    mq.enqueue(U, S, _vid("k1.mp4"), kind="video")
    mq.enqueue(U, S, {}, kind="session_end")
    sc.drain_session(U, S)
    clips = [c for c in spy if c[0] == "clip"]
    assert clips == [("clip", S, "k0.mp4"), ("clip", S, "k1.mp4")]   # 两批都消费
    assert spy.count(("final", S)) == 1                              # 只终审一次
    assert spy[-1] == ("final", S) and spy[-2] == ("end", S)
    assert mq.cursor_get(U, S) == 5


def test_concurrent_sessions_independent(spy):
    """多 session:各自 cursor 独立==自己消息数,调用不串。"""
    mq, sc = _consumer(spy)
    U = "u3"
    for s in ("sa", "sb"):
        mq.enqueue(U, s, _vid(f"{s}-c0.mp4"), kind="video")
        mq.enqueue(U, s, {}, kind="session_end")
    sc.drain_session(U, "sa")
    sc.drain_session(U, "sb")
    assert mq.cursor_get(U, "sa") == 2 and mq.cursor_get(U, "sb") == 2
    assert ("clip", "sa", "sa-c0.mp4") in spy and ("clip", "sb", "sb-c0.mp4") in spy
    # sa 的 clip 不会带 sb 的 session_id(不串)
    assert not any(c == ("clip", "sa", "sb-c0.mp4") for c in spy)


def test_video_idempotent_no_double_consume(spy):
    """重投幂等:同一视频消息 drain 两次,cursor 挡住不重复 process_clip。"""
    mq, sc = _consumer(spy)
    U, S = "u4", "s4"
    mq.enqueue(U, S, _vid("once.mp4"), kind="video")
    sc.drain_session(U, S)
    sc.drain_session(U, S)   # 再 drain:队列已空 + cursor 已推进
    assert [c for c in spy if c[0] == "clip"] == [("clip", S, "once.mp4")]   # 只处理一次


def test_bulk_mixed_stress_no_loss_no_dup(spy):
    """批量压测(生产>消费模型:先猛灌再消费):多 session × 多混合消息 → 各 cursor 恰好==灌入数、
    队列排空、clip 全处理一次(不丢不重)。"""
    mq, sc = _consumer(spy)
    U = "ustress"
    sessions = [f"s{i}" for i in range(6)]
    per_session_msgs = {}
    for s in sessions:                                   # 先全部猛灌(不消费)= 堆积
        n = 0
        mq.enqueue(U, s, _txt(text="开"), kind="ingest"); n += 1
        mq.enqueue(U, s, _vid(f"{s}-a.mp4", f"{s}-b.mp4"), kind="video"); n += 1
        mq.enqueue(U, s, _txt(text="中"), kind="ingest"); n += 1
        mq.enqueue(U, s, _vid(f"{s}-c.mp4"), kind="video"); n += 1
        mq.enqueue(U, s, {}, kind="session_end"); n += 1
        per_session_msgs[s] = n
    # 消费(每 session drain 到空;drain 有 max_drain 上限,循环到 more=False)
    for s in sessions:
        while sc.drain_session(U, s).more:
            pass
    for s in sessions:
        assert mq.cursor_get(U, s) == per_session_msgs[s]        # 恰好消费自己那些,不多不少
        assert mq.depth(U, s) == 0                               # 队列排空
    clips = [c for c in spy if c[0] == "clip"]
    assert len(clips) == len(sessions) * 3                       # 每 session 3 个 clip,全处理一次
    assert len(clips) == len({(c[1], c[2]) for c in clips})      # 无重复(不重)
    assert sum(1 for c in spy if c[0] == "final") == len(sessions)  # 每 session 终审一次


def test_parallel_drain_many_sessions(spy):
    """真并发:线程池并行 drain 多 session(锁竞争下)→ 各 cursor 精确、clip 不丢不重不串。"""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    mq, sc = _consumer(spy)
    U = "upar"
    sessions = [f"p{i}" for i in range(8)]
    for s in sessions:
        mq.enqueue(U, s, _txt(text="a"), kind="ingest")
        mq.enqueue(U, s, _vid(f"{s}-x.mp4", f"{s}-y.mp4"), kind="video")
        mq.enqueue(U, s, {}, kind="session_end")
    lock = threading.Lock()                      # 只保护 spy list 本身(list.append 已原子,双保险)

    def work(s):
        with lock:
            pass
        while sc.drain_session(U, s).more:
            pass

    with ThreadPoolExecutor(max_workers=8) as ex:   # 8 线程真并行
        list(ex.map(work, sessions))

    for s in sessions:
        assert mq.cursor_get(U, s) == 3 and mq.depth(U, s) == 0      # 各自精确消费、排空
    clips = [c for c in spy if c[0] == "clip"]
    assert len(clips) == len(sessions) * 2                            # 不丢
    assert len(clips) == len({(c[1], c[2]) for c in clips})           # 不重
    for s in sessions:                                                # 不串:key 前缀与 session 对应
        assert all(c[2].startswith(s) for c in clips if c[1] == s)
    assert sum(1 for c in spy if c[0] == "final") == len(sessions)


def test_same_session_lock_contention_no_double_consume(spy):
    """同会话锁竞争:多线程同时 drain 同一 session → 只有拿到锁的处理,消息不被重复消费。"""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    mq, sc = _consumer(spy)
    U, S = "ulock", "s1"
    for i in range(6):
        mq.enqueue(U, S, _vid(f"c{i}.mp4"), kind="video")
    barrier = threading.Barrier(4)

    def work(_):
        barrier.wait()                       # 4 线程同时冲同一 session
        reps = []
        while True:
            r = sc.drain_session(U, S)
            reps.append(r)
            if not r.more:
                break
        return reps

    with ThreadPoolExecutor(max_workers=4) as ex:
        all_reps = list(ex.map(work, range(4)))

    clips = [c for c in spy if c[0] == "clip"]
    assert len(clips) == 6                                   # 6 条全处理
    assert len(clips) == len({c[2] for c in clips})           # 每条只处理一次(无双重消费)
    assert mq.cursor_get(U, S) == 6 and mq.depth(U, S) == 0
    assert any(not r.locked for reps in all_reps for r in reps)   # 确有线程抢锁失败(竞争真发生)


def test_multi_user_parallel_no_cross_contamination(spy_deps):
    """**多用户并发**(生产刚需):N 个不同 user 各自多 session 并行消费 →
    ①每条 clip 拿到的 deps 必须属于自己那个 user(跨 user 零串);②各 user 各 session cursor 精确。"""
    from concurrent.futures import ThreadPoolExecutor

    mq = MemoryMsgQueue()
    sc = SessionConsumer(mq, MemorySessionLock(), lambda u, s: _Writer([], s),
                         video_deps=lambda u: f"deps:{u}")   # deps 带 user 身份,供隔离断言
    users = [f"u{i}" for i in range(6)]
    sessions = ["a", "b"]
    for u in users:
        for s in sessions:
            mq.enqueue(u, s, _vid(f"{u}-{s}-1.mp4", f"{u}-{s}-2.mp4"), kind="video")
            mq.enqueue(u, s, {}, kind="session_end")

    jobs = [(u, s) for u in users for s in sessions]
    with ThreadPoolExecutor(max_workers=12) as ex:          # 12 线程跨 user 并行
        list(ex.map(lambda j: [None for _ in iter(lambda: sc.drain_session(*j).more, False)], jobs))

    clips = [c for c in spy_deps if c[0] == "clip"]
    assert len(clips) == len(users) * len(sessions) * 2                  # 不丢
    assert len(clips) == len({c[3] for c in clips})                      # 不重
    for _tag, deps, _sid, keyname in clips:                             # **跨 user 隔离**
        owner = keyname.split("-")[0]                                    # key 前缀即所属 user
        assert deps == f"deps:{owner}", f"user 串了: {deps} 处理了 {keyname}"
    for _tag, deps, sid in [c for c in spy_deps if c[0] == "final"]:
        assert deps.startswith("deps:u")
    for u in users:
        for s in sessions:
            assert mq.cursor_get(u, s) == 2 and mq.depth(u, s) == 0      # 各自精确、排空


def test_video_url_payload_consumed(spy):
    """外链形态(video_oss_key=None + video_url)能被正常消费——回归 None[-16:] 崩溃打成毒消息。"""
    mq, sc = _consumer(spy)
    U, S = "uurl", "s1"
    mq.enqueue(U, S, _vid_url("https://caller-bucket/a.mp4", "https://caller-bucket/b.mp4"),
               kind="video")
    mq.enqueue(U, S, {}, kind="session_end")
    rep = sc.drain_session(U, S)
    assert rep.applied == 2 and rep.poisoned == 0          # 不再被判毒消息
    assert [c for c in spy if c[0] == "clip"] == [
        ("clip", S, "https://caller-bucket/a.mp4"), ("clip", S, "https://caller-bucket/b.mp4")]
    assert mq.cursor_get(U, S) == 2


def test_mixed_url_and_key_bursts(spy):
    """混用两种来源(批1 外链 / 批2 我方 key)都能消费,cursor 精确。"""
    mq, sc = _consumer(spy)
    U, S = "umix", "s1"
    mq.enqueue(U, S, _vid_url("https://caller/x.mp4"), kind="video")
    mq.enqueue(U, S, _vid("ourkey/y.mp4"), kind="video")
    mq.enqueue(U, S, {}, kind="session_end")
    sc.drain_session(U, S)
    assert [c[2] for c in spy if c[0] == "clip"] == ["https://caller/x.mp4", "ourkey/y.mp4"]
    assert mq.cursor_get(U, S) == 3


def test_clip_rejected_is_recorded_not_silently_dropped(monkeypatch):
    """clip 永久性失败(外链失效/超大/超长)→ **留痕**后跳过,不重试不静默;整批其余 clip 继续。"""
    from personos.online import video_ingest

    recorded = []

    class _TS:   # 假 TaskStore:记留痕调用
        def create(self, tid, kind, u, s):
            recorded.append(("create", kind, u, s))

        def mark_error(self, tid, err):
            recorded.append(("error", err))

    def _proc(deps, *, session_id, clip_key="", clip_url="", **k):
        src = clip_key or clip_url
        if "bad" in src:
            raise video_ingest.ClipRejected(f"视频地址不可访问(HTTP 403):{src}")
        recorded.append(("ok", src))

    monkeypatch.setattr(video_ingest, "process_clip", _proc)
    monkeypatch.setattr(video_ingest, "finalize_video", lambda deps, *, session_id: None)

    mq = MemoryMsgQueue()
    sc = SessionConsumer(mq, MemorySessionLock(), lambda u, s: _Writer([], s),
                         video_deps=lambda u: "deps", task_store=_TS())
    U, S = "urej", "s1"
    mq.enqueue(U, S, _vid_url("https://ok/a.mp4", "https://bad/b.mp4", "https://ok/c.mp4"),
               kind="video")
    rep = sc.drain_session(U, S)

    assert rep.applied == 1 and rep.poisoned == 0          # 整条消息算成功(不重试不毒消息)
    assert ("ok", "https://ok/a.mp4") in recorded and ("ok", "https://ok/c.mp4") in recorded  # 其余继续
    assert ("create", "video_clip_rejected", U, S) in recorded                                # 留痕了
    assert any(t == "error" and "403" in e for t, e in
               [(r[0], r[1]) for r in recorded if r[0] == "error"])                           # 原因可查
    assert mq.cursor_get(U, S) == 1


def test_lock_renewed_during_long_clip_processing(monkeypatch):
    """**锁续期**:单条 clip 处理远长于续期间隔(视频常态 2-3min)时,心跳必须持续续锁,
    且期间别的消费者抢不到锁(不会并发消费同一会话)。"""
    import threading
    import time

    from personos.online import video_ingest

    renews: list[float] = []

    class _SpyLock(MemorySessionLock):
        def renew(self, u, s, handle):
            renews.append(time.monotonic())
            return super().renew(u, s, handle)

    PROC_S, RENEW_S = 0.9, 0.15          # 处理时长 ≫ 续期间隔 → 应续多次
    monkeypatch.setattr(video_ingest, "process_clip",
                        lambda deps, **k: time.sleep(PROC_S))
    monkeypatch.setattr(video_ingest, "finalize_video", lambda deps, *, session_id: None)

    mq, lock = MemoryMsgQueue(), _SpyLock()
    sc = SessionConsumer(mq, lock, lambda u, s: _Writer([], s),
                         video_deps=lambda u: "deps", renew_interval_s=RENEW_S)
    U, S = "ulongclip", "s1"
    mq.enqueue(U, S, _vid("slow.mp4"), kind="video")

    stolen = []

    def intruder():                       # 处理进行中,另一消费者尝试抢同一会话
        time.sleep(PROC_S / 2)
        stolen.append(lock.try_acquire(U, S))

    t = threading.Thread(target=intruder)
    t.start()
    rep = sc.drain_session(U, S)
    t.join()

    assert rep.applied == 1                                   # 长耗时下正常完成
    assert len(renews) >= 2, f"续锁次数不足:{len(renews)}(处理 {PROC_S}s / 间隔 {RENEW_S}s)"
    assert stolen == [None], "处理期间锁被别人抢走了——会并发消费同一会话"


class _FakeResp:
    def __init__(self, status=200, headers=None, chunks=()):
        self.status_code, self.headers, self._chunks = status, headers or {}, chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self):
        yield from self._chunks


def test_fetch_rejects_http_error(monkeypatch):
    """外链 4xx/5xx → ClipRejected(永久失败,不重试)。"""
    import httpx

    from personos.online import video_ingest

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeResp(status=403))
    with pytest.raises(video_ingest.ClipRejected, match="不可访问"):
        video_ingest._materialize_clip(object(), clip_key="", clip_url="https://expired/a.mp4",
                                       owner="u")


def test_fetch_rejects_oversize_declared(monkeypatch):
    """Content-Length 声明超上限 → 立刻拒,不下载。"""
    import httpx

    from personos.online import video_ingest

    big = str(video_ingest.MAX_CLIP_BYTES + 1)
    monkeypatch.setattr(httpx, "stream",
                        lambda *a, **k: _FakeResp(headers={"content-length": big}))
    with pytest.raises(video_ingest.ClipRejected, match="上限"):
        video_ingest._materialize_clip(object(), clip_key="", clip_url="https://x/big.mp4",
                                       owner="u")


def test_fetch_rejects_oversize_while_streaming(monkeypatch):
    """未声明 Content-Length 但下载中超限 → 边下边卡,不把超大文件读满内存。"""
    import httpx

    from personos.online import video_ingest

    monkeypatch.setattr(video_ingest, "MAX_CLIP_BYTES", 10)
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeResp(chunks=[b"x" * 6, b"x" * 6]))
    with pytest.raises(video_ingest.ClipRejected, match="下载中超限"):
        video_ingest._materialize_clip(object(), clip_key="", clip_url="https://x/nolen.mp4",
                                       owner="u")


def test_temp_file_cleaned_on_reject(monkeypatch, tmp_path):
    """被拒(超大/超长/下载失败)时不留垃圾临时文件——失败路径也清盘。"""
    import httpx

    from personos.online import video_ingest

    before = set(os.listdir(tempfile.gettempdir()))
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _FakeResp(status=404))
    with pytest.raises(video_ingest.ClipRejected):
        video_ingest._materialize_clip(object(), clip_key="", clip_url="https://x/404.mp4",
                                       owner="u")
    leaked = [f for f in set(os.listdir(tempfile.gettempdir())) - before if f.endswith(".mp4")]
    assert not leaked, f"失败路径泄漏临时文件:{leaked}"


def test_video_concurrency_is_bounded_only_by_the_pool(monkeypatch):
    """视频并发只由线程池大小决定——process_clip 内不得再有第二道闸。

    回归:曾经池是 50、函数内还有个 BoundedSemaphore(4),生效的永远是更严的那道,
    46 个线程卡在闸上干等且各自握着会话锁。两道闸管同一件事就只该留一道。
    这里用 8 宽的池跑 8 个 clip,若函数内还有闸,峰值并行会被压到闸的大小。
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from personos.online import video_ingest

    live, peak, lk = 0, 0, threading.Lock()

    def _slow(deps, **k):
        nonlocal live, peak
        with lk:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lk:
            live -= 1
        return 0

    monkeypatch.setattr(video_ingest, "_process_clip_locked", _slow)
    deps = object()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda i: video_ingest.process_clip(deps, session_id="s", clip_key=f"{i}.mp4"),
                    range(8)))
    assert peak == 8, f"池给了 8 个线程,却只并行了 {peak} —— 函数内还残留着闸"


def _fake_deps(monkeypatch, omni_calls):
    """构造最小 VideoDeps(记录 omni 调用次数),供时长/顺序类断言。"""
    from personos.identity.draft import MemoryDraftStore
    from personos.online import video_ingest

    class _Omni:
        def chat(self, *a, **k):
            omni_calls.append(1)
            return "{}"

    class _MS:
        def read_bytes(self, key):
            return b"fakebytes"

        def sign_url(self, key):
            return f"https://our-oss/{key}"

    return video_ingest.VideoDeps(
        store=type("S", (), {"user_id": "u"})(), cloud=None, draft=MemoryDraftStore("u"),
        backends={"mm_runner": _Omni()}, media_store=_MS(), maas=None,
        evidence=None, cells=None, atoms=None, chains=None)


def test_url_path_rejects_too_long_before_spending_mllm(monkeypatch):
    """外链路径:文件已在手 → **先卡时长再跑剧本**,过长立刻拒,省掉一次 2-3min MLLM。"""
    from personos.online import video_ingest

    calls = []
    monkeypatch.setattr(video_ingest, "MAX_CLIP_DURATION_S", 5.0)
    monkeypatch.setattr(video_ingest, "_duration", lambda p: 99.0)
    monkeypatch.setattr(video_ingest, "_materialize_clip",
                        lambda ms, *, clip_key, clip_url, owner: ("/tmp/fake.mp4", "our/k.mp4"))
    deps = _fake_deps(monkeypatch, calls)
    with pytest.raises(video_ingest.ClipRejected, match="时长"):
        video_ingest.process_clip(deps, session_id="s", clip_url="https://caller/a.mp4")
    assert calls == [], "外链路径应在跑剧本前就拒掉过长 clip"


def test_key_path_downloads_only_after_screenplay(monkeypatch):
    """我方 key 路径:剧本期间盘上无文件,用到 harvest 时才下载(临时文件存活最短)。"""
    from personos.online import video_ingest

    calls, order = [], []
    monkeypatch.setattr(video_ingest, "MAX_CLIP_DURATION_S", 5.0)
    monkeypatch.setattr(video_ingest, "_duration", lambda p: 99.0)

    def _mat(ms, *, clip_key, clip_url, owner):
        order.append(f"download(omni_calls={len(calls)})")
        return "/tmp/fake.mp4", clip_key or "our/k.mp4"

    monkeypatch.setattr(video_ingest, "_materialize_clip", _mat)
    deps = _fake_deps(monkeypatch, calls)
    with pytest.raises(video_ingest.ClipRejected, match="时长"):
        video_ingest.process_clip(deps, session_id="s", clip_key="k.mp4")
    assert order == ["download(omni_calls=1)"], f"下载应发生在剧本之后:{order}"


def test_session_end_without_video_still_finalizes_noop(spy):
    """纯文本会话 session_end:也调 finalize_video(内部判 pending 为空→no-op),不报错。"""
    mq, sc = _consumer(spy)
    U, S = "u5", "s5"
    mq.enqueue(U, S, _txt(), kind="ingest")
    mq.enqueue(U, S, {}, kind="session_end")
    sc.drain_session(U, S)
    assert ("end", S) in spy and ("final", S) in spy   # finalize 总被调(spy 里 no-op)


def test_dispatcher_routes_video_to_dedicated_pool():
    """**视频独立池**:队头是 video 的会话派到视频池,文本会话派到文本池——
    视频忙不占文本 worker(不会把文本消费饿死)。"""
    from personos.app.ingest_worker import Dispatcher

    submitted = {"text": [], "video": []}

    class _Pool:
        def __init__(self, tag):
            self.tag = tag

        def submit(self, fn, *a):
            submitted[self.tag].append(a[:2])

            class _F:
                pass
            return _F()

    mq = MemoryMsgQueue()
    mq.enqueue("u", "s_txt", _txt(), kind="ingest")
    mq.enqueue("u", "s_vid", _vid("a.mp4"), kind="video")
    d = Dispatcher(mq, object(), _Pool("text"), 10,
                   video_pool=_Pool("video"), video_cap=10)
    d.run_once()
    assert submitted["video"] == [("u", "s_vid")], f"视频会话没进视频池:{submitted}"
    assert submitted["text"] == [("u", "s_txt")], f"文本会话没进文本池:{submitted}"


def test_video_pool_full_does_not_block_text():
    """视频池占满时,文本会话仍能被派发(隔离的核心价值)。"""
    from personos.app.ingest_worker import Dispatcher

    got = []

    class _Pool:
        def __init__(self, tag):
            self.tag = tag

        def submit(self, fn, *a):
            got.append((self.tag, a[1]))

            class _F:
                pass
            return _F()

    mq = MemoryMsgQueue()
    for i in range(3):
        mq.enqueue("u", f"v{i}", _vid(f"{i}.mp4"), kind="video")
    mq.enqueue("u", "t1", _txt(), kind="ingest")
    d = Dispatcher(mq, object(), _Pool("text"), 10,
                   video_pool=_Pool("video"), video_cap=1)   # 视频池只有 1 个名额
    d.run_once()
    assert ("text", "t1") in got, f"视频池满把文本也挡住了:{got}"
    assert sum(1 for t, _ in got if t == "video") == 1        # 视频只派出 1 个(受限)


def test_poisoned_message_is_recorded(monkeypatch):
    """瞬时故障重试耗尽被判毒消息时也要**留痕**——否则那条 clip 的记忆丢了只剩会被冲掉的日志。"""
    from personos.online import video_ingest

    recorded = []

    class _TS:
        def create(self, tid, kind, u, s):
            recorded.append(("create", kind))

        def mark_error(self, tid, err):
            recorded.append(("error", err[:40]))

    def _boom(deps, **k):
        raise RuntimeError("Omni read timeout")      # 瞬时故障(非 ClipRejected)

    monkeypatch.setattr(video_ingest, "process_clip", _boom)
    monkeypatch.setattr(video_ingest, "finalize_video", lambda deps, *, session_id: None)

    mq = MemoryMsgQueue()
    sc = SessionConsumer(mq, MemorySessionLock(), lambda u, s: _Writer([], s),
                         video_deps=lambda u: "deps", task_store=_TS(), max_retries=2)
    U, S = "upoison", "s1"
    mq.enqueue(U, S, _vid("boom.mp4"), kind="video")
    for _ in range(3):                                # 重试到判毒
        try:
            sc.drain_session(U, S)
        except Exception:
            pass
    assert ("create", "poisoned_video") in recorded, f"判毒未留痕:{recorded}"
    assert any(t == "error" for t, _ in recorded)


def test_upstream_media_failures_become_clip_rejected(monkeypatch):
    """上游"拉不动媒体"/"内容审查拒绝" → ClipRejected(留痕跳过),不当瞬时故障重试 5 次。

    回归两起实测事故:
    - 2min@7.3Mbps(106MB)的 clip 让模型服务侧下载超时返回 400;
    - SIT 镜像没装 PyAV,`import av` 抛 ModuleNotFoundError。
    两者都被当瞬时故障重试 5 轮,而我方 key 路径的失败点在剧本 MLLM **之后**——
    每轮先白跑一次 2 分钟的剧本调用,5 轮就是十分钟上游配额,最后照样判毒。
    """
    from personos.identity.backends.omni import ContentRejectedError, MediaUnfetchableError
    from personos.online import video_ingest

    for exc in (MediaUnfetchableError("Download multimodal file timed out"),
                ContentRejectedError("data_inspection_failed"),
                ModuleNotFoundError("No module named 'av'")):
        def _boom(*a, **k):
            raise exc

        monkeypatch.setattr(video_ingest, "_process_clip_locked", _boom)
        with pytest.raises(video_ingest.ClipRejected):
            video_ingest.process_clip(object(), session_id="s", clip_key="k.mp4")


def test_omni_maps_media_download_failure_to_its_own_error():
    """MAAS 的 `Download multimodal file timed out` 必须落到 MediaUnfetchableError,
    而不是被泛化的 4xx 分支吞成普通 HTTPStatusError(那样上层分不出该不该重试)。"""
    import httpx

    from personos.identity.backends import omni as omni_mod

    body = ('{"detail":"{\\"error\\":{\\"message\\":\\"<400> InternalError.Algo.'
            'InvalidParameter: Download multimodal file timed out\\"}}"}')

    class _Resp:
        status_code = 400
        text = body
        request = None

        def raise_for_status(self):
            raise httpx.HTTPStatusError("400", request=None, response=self)

    class _Client:
        def post(self, url, **kw):
            return _Resp()

    r = omni_mod.OmniRunner()
    r._client = _Client()
    r.cfg = type("C", (), {"mllm_api_key": "k", "mllm_endpoint": "http://x",
                           "mllm_timeout": 120.0})()
    with pytest.raises(omni_mod.MediaUnfetchableError):
        r.chat("p", video_url="https://oss/a.mp4")


def _idem_deps(monkeypatch, calls, fail_on=None):
    """构造能跑通 _process_clip_locked 全程的最小 deps(记录每个 clip 被真正处理了几次)。"""
    from personos.identity.draft import MemoryDraftStore
    from personos.identity.screenplay import ClipScript
    from personos.online import video_ingest

    class _Omni:
        def chat(self, *a, **k):
            return "{}"

    class _MS:
        def read_bytes(self, key):
            return b"x"

        def sign_url(self, key):
            return f"https://our/{key}"

    def _parse(raw, duration_sec=None):
        return ClipScript(cast_map={}, parsed_ok=True)

    monkeypatch.setattr(video_ingest, "parse_clip_output", _parse)
    monkeypatch.setattr(video_ingest, "build_clip_prompt", lambda **k: ("p", []))
    monkeypatch.setattr(video_ingest, "_duration", lambda p: 10.0)
    monkeypatch.setattr(video_ingest, "_materialize_clip",
                        lambda ms, *, clip_key, clip_url, owner: ("", clip_key or "our/k.mp4"))

    def _harvest(path, script, backends):
        calls.append(path)
        if fail_on is not None and len(calls) == fail_on:
            raise RuntimeError("瞬时故障")       # 非 ClipRejected → 应触发整条消息重投
        return {}

    monkeypatch.setattr(video_ingest, "harvest_clip", _harvest)
    return video_ingest.VideoDeps(
        store=type("S", (), {"user_id": "u"})(), cloud=None, draft=MemoryDraftStore("u"),
        backends={"mm_runner": _Omni()}, media_store=_MS(), maas=None,
        evidence=None, cells=None, atoms=None, chains=None)


def test_replayed_clip_is_skipped_not_double_counted(monkeypatch):
    """同一 clip 被重放 → 只真正处理一次,且返回原 clip_index。

    回归真缺陷:队列重投重放的是**整条消息**(一条最多 20 个 clip),批内靠后的 clip 一次
    瞬时故障就会让前面的全部重跑 —— presence 记两遍、台词进两次 memcell、素材传两份。
    """
    from personos.online import video_ingest

    calls = []
    deps = _idem_deps(monkeypatch, calls)
    i1 = video_ingest.process_clip(deps, session_id="s", clip_key="a.mp4")
    i2 = video_ingest.process_clip(deps, session_id="s", clip_key="a.mp4")   # 重放
    assert (i1, i2) == (0, 0), f"重放应返回原序号:{i1},{i2}"
    assert len(calls) == 1, f"重放不应再跑一次 harvest:{calls}"
    assert deps.draft.clip_keys("s") == {0: "a.mp4"}


def test_failed_clip_is_retried_not_marked_done(monkeypatch):
    """中途失败的 clip **必须**能重跑——去重挂在"已完成"而非"序号已分配"上。

    若挂在 next_clip_seq(在剧本 MLLM 之前分配),失败的 clip 重投时会被误判成已处理而跳过,
    把"重复记账"换成"静默丢记忆",比原缺陷更糟。这条锁住那个更糟的实现。
    """
    from personos.online import video_ingest

    calls = []
    deps = _idem_deps(monkeypatch, calls, fail_on=1)      # 第 1 次处理抛瞬时故障
    with pytest.raises(RuntimeError, match="瞬时故障"):
        video_ingest.process_clip(deps, session_id="s", clip_key="a.mp4")
    assert deps.draft.clip_done_index("s", "a.mp4") is None, "失败的 clip 不得标成已完成"

    idx = video_ingest.process_clip(deps, session_id="s", clip_key="a.mp4")   # 重投重试
    assert len(calls) == 2, f"失败的 clip 必须真正重跑:{calls}"
    assert idx == 1, "重跑拿新序号(上次失败的序号作废,无 lines 不影响 flush)"
    assert deps.draft.clip_done_index("s", "a.mp4") == 1
