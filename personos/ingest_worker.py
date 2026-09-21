"""有序消费的两个角色:SessionConsumer(消费一个会话的队列)+ Dispatcher(调度谁来消费)。

架构(见 docs/design 对齐记录):
- **一会话一队列**(msg_queue),消息按 seq FIFO 排队,持久跨副本。
- **Dispatcher**:每 pod 一个后台循环,扫看板(有待处理消息的会话)→ 对每个会话向 ingest 池
  提交一个「drain 该会话」的作业;池满则本轮跳过(背压),下一轮再来——绝不自旋占线程。
- **SessionConsumer.drain_session**:非阻塞抢会话锁(单飞:同会话同时只有一个消费者,跨副本
  互斥)→ 先回放上次崩溃遗留的在途消息(recover)→ 再 FIFO drain 至多 max_drain 条(公平:
  话痨会话不霸占线程)→ 松锁 → 排空则移出看板。
- **不重复**:每条消息按 seq 与会话游标比对,seq<=cursor 判重投直接 ack 跳过;正常消费成功后
  推进游标再 ack。可靠队列 + 游标 → 正常恰好一次、崩溃至少一次且幂等去重。

依赖注入(不耦合 runtime/FastAPI),便于单测:传 mq / lock / writer_for 回调即可。
"""

from __future__ import annotations

import base64
import threading
import uuid
import time
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from . import obs
from .online.write_path import FeedMsg, SessionWriter
from .storage.msg_queue import Envelope, MsgQueue
from .storage.session_lock import SessionLock


@dataclass
class DrainReport:
    """一次 drain_session 的结果(供调度/测试/日志)。"""
    locked: bool               # 是否抢到锁(False=别处正在消费该会话,本次空转)
    applied: int = 0           # 真正应用的消息数
    skipped: int = 0           # 判重投跳过的消息数
    poisoned: int = 0          # 判毒消息跳过的数(连续失败达上限)
    more: bool = False         # 松锁时队列仍有积压(达 max_drain 主动歇手 / 期间又来新消息)


class SessionConsumer:
    """消费单个会话队列:单飞 + 有序 + 幂等去重。无自身状态,可被多线程复用。"""

    def __init__(self, mq: MsgQueue, lock: SessionLock,
                 writer_for: Callable[[str, str], SessionWriter],
                 *, max_drain: int = 20, max_retries: int = 5, renew_interval_s: float = 200.0,
                 after_drain: Callable[[str, str, str], None] | None = None,
                 video_deps: Callable[[str], Any] | None = None,
                 task_store: Any | None = None):
        self._mq = mq
        self._lock = lock
        self._writer_for = writer_for
        # 视频依赖取数器(runtime.video_deps);None=该进程不处理视频(纯文本 pod)。
        self._video_deps = video_deps
        self._task_store = task_store   # clip 永久失败留痕(可为 None:不留痕但不阻塞)
        self._max_drain = max_drain
        self._max_retries = max_retries   # 一条消息连续失败达此数 → 判毒消息,跳过
        self._renew_interval = renew_interval_s   # 心跳续锁间隔(应 < 锁 TTL,取 TTL/3)
        # 关段后钩子(可选):有消息被应用后调用,如触发画像整理。松锁后调,失败不影响消费。
        # 第三参 = 触发它的最近一批 ingest 的 scenario(per-request 场景透传给画像整理)。
        self._after_drain = after_drain

    def _record_failure(self, user_id: str, session_id: str, kind: str, detail: str) -> None:
        """失败留痕(tasks 表):让"这部分内容没进记忆"可查、可告警,而不是只躺在会被冲掉的日志里。
        kind 形如 video_clip_rejected / poisoned_video / poisoned_ingest。留痕失败不阻塞消费。"""
        if self._task_store is None:
            return
        try:
            tid = f"fail_{uuid.uuid4().hex[:16]}"
            self._task_store.create(tid, kind, user_id, session_id)
            self._task_store.mark_error(tid, detail)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"失败留痕未写成(不阻塞消费): {e}")

    def _do_write(self, user_id: str, session_id: str, env: Envelope,
                  last_scenario: dict | None = None) -> None:
        """真正写入(可能抛):session_end 强制闭合 / ingest 批喂入。

        last_scenario:可变载体(drain 内局部);记下本批 scenario,供 after_drain 透传给画像整理
        (per-request 的 scenario 没有落库,靠"触发它的那批 ingest"带出来)。
        """
        writer = self._writer_for(user_id, session_id)
        # langfuse 根 trace:一条消息消费 = 一个 trace(其下 W1 边界/W2 episode·atom/W2.5 判链
        # 的 maas.chat 自动 nest 到本 span)。input 设为消息文本 → langfuse 主界面可读,不空白。
        # 注:段首条消息只缓冲不触发 LLM → 该 trace 无子阶段(正常,LLM 在关段那条 trace)。
        if env.kind == "session_end":
            _inp = "(session end: 闭合文本尾段 + 视频终审)"
        elif env.kind == "video":
            # 注意:video_oss_key 键存在但值可能是 None(走 video_url 时),dict.get 的默认值
            # 不会生效 → 必须 `or ''` 兜底,否则 None[-16:] 直接 TypeError 把整条消息打成毒消息。
            _inp = " | ".join(
                f"[clip {(m.get('video_oss_key') or m.get('video_url') or '')[-16:]}]"
                for m in env.payload.get("messages", []))[:500]
        else:
            _inp = " | ".join((str(m.get("text")) if m.get("text") else "[image]")
                              for m in env.payload.get("messages", []))[:500]
        with obs.root_span(f"ingest.{env.kind}", user_id=user_id, session_id=session_id,
                           input=_inp, metadata={"msg_id": env.msg_id, "seq": env.seq},
                           trace_id=env.payload.get("trace_id")) as _sp:
            if _sp is not None:
                logger.info(f"ingest langfuse trace_id={obs.current_trace_id()} "
                            f"kind={env.kind} u={user_id} s={session_id}")
            scenario = env.payload.get("scenario") or ""
            if last_scenario is not None:
                last_scenario["scenario"] = scenario   # 记最近一批的场景(供 after_drain → 画像)
            if env.kind == "session_end":
                writer.end_session(task_type=env.payload.get("task_type"),
                                   scenario=scenario)   # 收尾:闭合文本尾段
                # 视频终审:若该 session 有 pending 视频草稿 → commit_session + 行归属落记忆
                # (finalize_video 内部判 pending,无视频则 no-op)。与文本尾段同 drain 锁内串行。
                if self._video_deps is not None:
                    from personos.online import video_ingest
                    video_ingest.finalize_video(self._video_deps(user_id), session_id=session_id)
            elif env.kind == "video":
                if self._video_deps is None:
                    raise RuntimeError("video backends are not configured in this process (text-only worker), cannot consume a video message")
                from personos.online import video_ingest
                deps = self._video_deps(user_id)
                for m in env.payload.get("messages", []):     # 一批可含多 clip,逐 clip 顺序处理
                    src = m.get("video_oss_key") or m.get("video_url") or ""
                    try:
                        video_ingest.process_clip(
                            deps, session_id=session_id,
                            clip_key=m.get("video_oss_key") or "",
                            clip_url=m.get("video_url") or "",
                            scene=scenario or video_ingest.DEFAULT_SCENE,
                            clip_meta={"clip_index": m.get("clip_index"),
                                       "duration_sec": m.get("duration_sec")})
                    except video_ingest.ClipRejected as e:
                        # 永久性失败(外链失效/超大/超长/格式错):重试无意义 → **留痕**后跳过这条 clip,
                        # 绝不静默丢(调用方可凭 task 查到哪条 clip 没进记忆)。整批其余 clip 继续。
                        self._record_failure(user_id, session_id, "video_clip_rejected",
                                             f"src={src}\n{e}")
                        logger.warning(f"clip 被拒(已留痕,跳过)u={user_id} s={session_id} "
                                       f"src={src[-40:]} 原因={e}")
            else:
                p = env.payload
                msgs = [FeedMsg(speaker=m.get("speaker", "user"), text=m.get("text", ""),
                                image=base64.b64decode(m["image_b64"]) if m.get("image_b64") else None,
                                image_content_type=m.get("image_content_type", "image/jpeg"))
                        for m in p.get("messages", [])]       # 一条队列消息 = 一个原子批
                writer.feed_batch(msgs, source_extra=p.get("source_extra"),
                                  task_type=p.get("task_type"), scenario=scenario)

    def _apply(self, user_id: str, session_id: str, env: Envelope,
               last_scenario: dict | None = None) -> str:
        """应用一条消息(已持锁)。返回 'applied' | 'skipped'(重投) | 'poisoned'(毒消息跳过)。

        - 幂等:seq<=游标 → 判重投,直接 ack 跳过(崩溃恢复重取同一条不会记两遍)。
        - 写入失败:累计失败 < 上限 → 抛出(留在途,下轮重试,瞬时故障自愈);
          达上限 → 判毒消息,记 ERROR + 推进游标跳过(解除队头阻塞,后续消息继续)。
        """
        cur = self._mq.cursor_get(user_id, session_id)
        if env.seq <= cur:
            logger.debug(f"消费去重跳过 u={user_id} s={session_id} seq={env.seq}<=cursor={cur}")
            self._mq.ack(user_id, session_id, env)
            return "skipped"
        try:
            self._do_write(user_id, session_id, env, last_scenario)
        except Exception:   # noqa: BLE001
            n = self._mq.mark_failed(user_id, session_id, env.msg_id)
            if n < self._max_retries:
                raise                                   # 未达上限:交给 drain_session 记录并留在途重试
            text = ""
            if env.payload.get("messages"):
                text = str(env.payload["messages"][0].get("text") or "")[:80]   # 防非串 text
            extra = "(尾段未闭合,该会话尾部对话不会进 cell,靠 24h seg TTL 作废)" \
                if env.kind == "session_end" else ""
            logger.error(f"毒消息跳过(连续失败 {n} 次)u={user_id} s={session_id} seq={env.seq} "
                         f"msg_id={env.msg_id} kind={env.kind} text={text!r}{extra}")
            # 判毒:推进游标 + 出队,当作已处理(内容进不了记忆)。除 ERROR 日志外**再留一条 task**,
            # 让"这条消息的记忆丢了"可查可告警——视频尤其要紧:一条 clip 没进记忆,日志一冲就无从追溯。
            self._record_failure(user_id, session_id, f"poisoned_{env.kind}",
                                 f"seq={env.seq} msg_id={env.msg_id} kind={env.kind} "
                                 f"text={text!r}{extra}\n连续失败 {n} 次判毒消息跳过")
            self._mq.cursor_set(user_id, session_id, env.seq)
            self._mq.ack(user_id, session_id, env)
            self._mq.clear_failed(user_id, session_id, env.msg_id)   # 计数键用完即清,不留垃圾
            return "poisoned"
        # 先推进游标再 ack:崩溃在两步之间 → 重取时 seq<=cursor 已成立 → 跳过(不重复应用)
        self._mq.cursor_set(user_id, session_id, env.seq)
        self._mq.ack(user_id, session_id, env)
        return "applied"

    def drain_session(self, user_id: str, session_id: str) -> DrainReport:
        """消费一个会话:抢锁→回放在途→FIFO drain(≤max_drain)→松锁→排空则下看板。"""
        token = self._lock.try_acquire(user_id, session_id)
        if not token:
            return DrainReport(locked=False)            # 别处正在消费,交给它
        applied = skipped = poisoned = 0
        last_scenario = {"scenario": ""}   # 本次 drain 最近一批 ingest 的 scenario(透传画像整理)

        def _tally(r):
            nonlocal applied, skipped, poisoned
            applied += r == "applied"
            skipped += r == "skipped"
            poisoned += r == "poisoned"

        # 心跳续锁:后台线程按 TTL/3 续期,覆盖"整段 drain"与"单条超长 apply"(一批最多 20 句、
        # 每句 W1/W2/看图可能各几十秒,单次 apply 可能逼近 TTL)。本 pod 活着就一直续 → 锁不中途过期;
        # pod 死则心跳随进程终止 → 锁到期由别副本接管。比"每条处理完才续"严密(那对单条超长无效)。
        stop_hb = threading.Event()

        def _heartbeat():
            while not stop_hb.wait(self._renew_interval):
                self._lock.renew(user_id, session_id, token)

        hb = threading.Thread(target=_heartbeat, name="drain-hb", daemon=True)
        hb.start()
        try:
            for env in self._mq.recover(user_id, session_id):   # 崩溃遗留:先按序回放
                _tally(self._apply(user_id, session_id, env, last_scenario))
            n = 0
            while n < self._max_drain:
                env = self._mq.reserve(user_id, session_id)
                if env is None:
                    break
                _tally(self._apply(user_id, session_id, env, last_scenario))
                n += 1
        except Exception:   # noqa: BLE001  单条消费失败不吞后续:该消息留在 proc,下轮 recover 重试
            logger.exception(f"会话消费异常 u={user_id} s={session_id}(消息留在途,下轮重试)")
        finally:
            stop_hb.set()
            hb.join(timeout=1)
            self._lock.release(user_id, session_id, token)
        more = self._mq.depth(user_id, session_id) > 0
        if not more:
            self._mq.deactivate_if_empty(user_id, session_id)   # 排空下看板(内含孤儿竞态复查)
        if applied or skipped or poisoned:
            logger.info(f"会话消费 u={user_id} s={session_id} applied={applied} "
                        f"skipped={skipped} poisoned={poisoned} more={more}")
        if applied and self._after_drain is not None:           # 有消息落库 → 触发画像整理(松锁后,不阻塞)
            try:
                self._after_drain(user_id, session_id, last_scenario["scenario"])
            except Exception:   # noqa: BLE001  画像触发失败绝不影响 ingest 消费结果
                logger.exception(f"after_drain 钩子异常 u={user_id} s={session_id}")
        return DrainReport(locked=True, applied=applied, skipped=skipped,
                           poisoned=poisoned, more=more)


class _Admission:
    """在途名额闸(非阻塞):满则拒。与 app.admission.AdmissionGate 同语义,这里内联避免循环依赖。"""

    def __init__(self, cap: int):
        self._sem = threading.BoundedSemaphore(cap)

    def try_enter(self) -> bool:
        return self._sem.acquire(blocking=False)

    def leave(self) -> None:
        self._sem.release()


class Dispatcher:
    """每 pod 一个:后台循环扫看板,把有活的会话派给 ingest 池消费。池满则背压跳过。"""

    def __init__(self, mq: MsgQueue, consumer: SessionConsumer, pool, cap: int,
                 *, tick_s: float = 0.05, idle_tick_s: float = 0.5, batch: int = 256,
                 video_pool=None, video_cap: int = 0):
        self._mq = mq
        self._consumer = consumer
        self._pool = pool                 # ThreadPoolExecutor(文本 ingest 专用)
        self._gate = _Admission(cap)      # 与池同容量:控制在途 drain 作业数(背压)
        # 视频独立池:clip 处理是重活(落盘+本地推理+2-3min MLLM),与文本共用池会把 worker
        # 占满、把文本消费饿死。按会话**队头消息的 kind** 分派:队头是 video 的会话走视频池,
        # 其余走文本池 —— 文本会话永远不排在视频作业后面。未配则退回共用(向前兼容)。
        self._video_pool = video_pool
        # 与视频池同容量的在途名额(背压:池满就别再往里塞);与 video_ingest 无关——
        # 那边曾有个同名的信号量,已删,视频并发只由池大小决定
        self._video_admission = _Admission(video_cap) if video_pool is not None else None
        self._tick = tick_s
        self._idle_tick = idle_tick_s
        self._batch = batch
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # 本 pod 内"已派发未完成"的会话:同会话不重复派(否则先到会话的重复派发会占满槽位,
        # 把后到会话饿死——锁只保证跨副本单飞,进程内饥饿要这层去重来防)
        self._inflight: set[tuple[str, str]] = set()
        self._inflight_guard = threading.Lock()
        # 派发次序:**最久未被派**的排最前(LRU)。曾用"每轮自增的轮转起点",但那个指针按
        # tick(毫秒级)走、名额却按 drain(秒~分钟级)释放,名额空出那一刻指针指向谁纯属偶然——
        # 刚跑完的会话常常立刻把名额抢回去,后到的会话只有"机会"没有"保证"。实测(cap=1、
        # 长会话 30 条+后到短会话)约 10% 的运行里短会话一直等到长会话全部排空。
        # 改 LRU 后"刚派过的排队尾"是确定的,与时序相位无关。
        self._dispatch_seq = 0
        self._last_dispatch: dict[tuple[str, str], int] = {}

    def run_once(self) -> int:
        """一轮调度(可单测):扫看板 → 有名额就派 drain 作业。返回本轮派出的作业数。

        同一会话即便被派多次,消费者的非阻塞锁保证只有一个真正在 drain,其余空转即返回——
        所以不需要在调度器维护「进行中」状态,靠锁天然去重。
        """
        sessions = self._mq.active_sessions(self._batch)
        if sessions:
            # 从未派过的记 -1 → 排在所有派过的之前(新会话优先,不被老话痨挡住)
            sessions = sorted(sessions, key=lambda k: self._last_dispatch.get(k, -1))
            live = set(sessions)            # 下看板的会话及时清出,字典不随时间无界增长
            if len(self._last_dispatch) > len(live):
                self._last_dispatch = {k: v for k, v in self._last_dispatch.items()
                                       if k in live}
        dispatched = 0
        video_full = text_full = False
        for user_id, session_id in sessions:
            if video_full and text_full:
                break                     # 两类池都满:本轮收工,下一轮再来(背压,不自旋)
            key = (user_id, session_id)
            with self._inflight_guard:
                if key in self._inflight:
                    continue              # 该会话已在派中:跳过,把槽位让给别的会话(防饥饿)
            pool, gate, is_video = self._route(user_id, session_id)
            if not gate.try_enter():
                # 该类池满:只跳过这一类会话,另一类仍可派(视频忙不拖累文本)。两类都满才收工。
                if is_video:
                    video_full = True
                else:
                    text_full = True
                continue
            # check-and-add 一次持锁完成(固化不变量:不依赖"仅单线程调 run_once"的脆弱前提)
            with self._inflight_guard:
                if key in self._inflight:
                    gate.leave()
                    continue
                self._inflight.add(key)
            try:
                pool.submit(self._job, user_id, session_id, gate)
                self._dispatch_seq += 1          # 刚派过 → 排到队尾(下轮让别人先)
                self._last_dispatch[key] = self._dispatch_seq
            except Exception:   # noqa: BLE001  提交失败(池关闭等):回滚 inflight + 归还名额,不泄漏
                with self._inflight_guard:
                    self._inflight.discard(key)
                gate.leave()
                raise
            dispatched += 1
        return dispatched

    def _route(self, user_id: str, session_id: str):
        """按会话**队头消息的 kind** 选池:video→视频池,其余→文本池。返回 (pool, gate, is_video)。
        未配视频池则一律走文本池(向前兼容)。"""
        if self._video_pool is not None and self._mq.head_kind(user_id, session_id) == "video":
            return self._video_pool, self._video_admission, True
        return self._pool, self._gate, False

    def _job(self, user_id: str, session_id: str, gate=None) -> None:
        try:
            self._consumer.drain_session(user_id, session_id)
        finally:
            with self._inflight_guard:
                self._inflight.discard((user_id, session_id))
            (gate or self._gate).leave()   # 名额务必归还给**它来自的那个**池(每条路径)

    def _loop(self) -> None:
        logger.info("ingest dispatcher 启动")
        while not self._stop.is_set():
            try:
                n = self.run_once()
            except Exception:   # noqa: BLE001  调度循环绝不因单轮异常退出
                logger.exception("dispatcher 调度轮异常")
                n = 0
            self._stop.wait(self._tick if n else self._idle_tick)
        logger.info("ingest dispatcher 停止")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ingest-dispatcher", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
