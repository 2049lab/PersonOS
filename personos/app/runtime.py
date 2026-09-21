"""进程内运行时单例:持久 DB + 用户注册表 + MAAS 客户端 + 异步任务基建。

多租户形态:全局单件持连接/客户端/线程池;每 user 一套绑定好的 stores(rt.for_user
缓存,LRU 上限)——业务代码拿到的 evidence/atoms/cells 天然只属于该 user,隔离由
store 层的 user_id 绑定保证,不存在"忘带过滤"的泄露面。

pod 无状态化(可水平扩容/滚动发布):跨请求会话态全部外置——
- 任务登记 → MySQL tasks 表(任意副本可轮询;内存 dict 重部署即丢);
- 未闭合段 + 会话写锁 → Redis(配置 REDIS_CLUSTER 时;否则进程内单副本实现);
- 写入状态机无实例态,writer_for 每次新建;常驻 dict 只剩 _uctx(有 LRU 上限)。
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from .. import obs
from ..clients.maas import MaasClient
from ..clients.mllm import mllm as _mllm
from ..config import settings
from .admission import AdmissionGate, TaskOverloaded
from .ingest_worker import Dispatcher, SessionConsumer
from ..logging_setup import setup_logging
from ..models import now
from ..online.profile_consolidate import run_user_consolidation, should_consolidate
from ..online.rerank import MaasReranker
from ..storage.profile_store import ProfileStore
from ..online.write_path import MAX_SEGMENT_TURNS, SessionWriter
from ..storage.atom_store import AtomStore
from ..storage.cell_store import CellStore
from ..storage.chain_store import ChainStore
from ..storage.db import Database
from ..storage.evidence_store import EvidenceStore
from ..storage.media_store import media_store_from_settings
from ..storage.msg_queue import MemoryMsgQueue, MsgQueue, RedisMsgQueue
from ..storage.redis_client import get_redis
from ..storage.seg_store import MemorySegStore, RedisSegStore, SegStore
from ..storage.session_lock import LOCK_TTL_S, MemorySessionLock, RedisSessionLock, SessionLock
from ..storage.task_store import TaskStore
from ..storage.user_store import UserStore

_UCTX_CAP = 1024          # 用户上下文 LRU 上限;stores 重建零成本,超限逐最旧
_SWEEP_INTERVAL_S = 300   # 任务表僵尸收割/过期清理的触发间隔
_MAX_PENDING = 200        # 后台在途任务上限:满则拒(503),见 admission.AdmissionGate


class UserContext:
    """某 user 的绑定上下文:stores 自动过滤到该 user(写入状态机已无实例态)。"""

    def __init__(self, user_id: str, db: Database):
        self.user_id = user_id
        self.evidence = EvidenceStore(db, user_id=user_id)
        self.atoms = AtomStore(db, user_id=user_id)
        self.cells = CellStore(db, user_id=user_id)
        self.chains = ChainStore(db, user_id=user_id)


class Runtime:
    """进程内单例:持久 DB + 用户注册表 + MAAS 客户端 + 跨副本会话态 + 异步任务基建。"""

    def __init__(self):
        setup_logging(settings.log_dir)
        self.db = Database()
        self.users = UserStore(self.db)
        # 默认(无 user 体系时的单租户视图)——兼容旧脚本;服务 API 一律走 for_user
        self.evidence = EvidenceStore(self.db)
        self.atoms = AtomStore(self.db)
        self.cells = CellStore(self.db)
        self.maas = MaasClient()
        self.reranker = MaasReranker(self.maas)   # R2 精排工位(失败自动保序,见 MaasReranker)
        self.task_store = TaskStore(self.db)
        # 图片输入:MLLM 看图客户端(未配 key 时 available=False,写入侧自动降级为纯文本);
        # OSS 存储懒建(首次带图 ingest 时装配,凭证缺失则 media_store 保持 None,原图不留底但不阻塞)。
        self.mllm = _mllm
        self._media_store = None
        self._media_guard = threading.Lock()
        # —— 跨副本会话态(seg/锁):首次使用时才装配(懒建池,import 不触网) ——
        self._warn_redis_env_mismatch()
        self._state_guard = threading.Lock()
        self._seg_store: SegStore | None = None
        self._session_lock: SessionLock | None = None
        # —— 异步基建:后台线程池(状态流转全部落 tasks 表)+ 在途上限闸 ——
        # 遗留后台任务池(submit_task;ingest 已改队列消费,此池仅零星低频任务)。不用 4 的小值
        self._executor = obs.ContextThreadPoolExecutor(max_workers=16, thread_name_prefix="rt-bg")
        self._admission = AdmissionGate(_MAX_PENDING)
        self._sweep_at = 0.0
        # —— 有序消费任务系统(Phase B):按类型分池 + 会话队列 + 调度器 ——
        # ingest_exec 跑 dispatcher 派的 drain 作业。recall 不再单开池:sync 端点已占一个 anyio
        # 线程,再 submit 到别池阻塞等 = 双线程占用无收益;改为当前线程内跑,仅用 recall_gate 限流隔离。
        # 保上下文池:每个 drain 任务在提交线程(dispatcher,无活跃 span)的复制上下文里跑,
        # 隔离 worker 线程复用残留的 OTel 上下文——否则并发下前一任务的 span 残留会污染后一任务,
        # 使 build_cell 的 episode_weave/atom_extract/chain_assign 脱离 ingest 根 → 孤儿 trace。
        self.ingest_exec = obs.ContextThreadPoolExecutor(max_workers=settings.ingest_pool_size,
                                                         thread_name_prefix="ingest")
        # 视频消费独立池:clip 处理是重活(落盘 + 本地人脸/声纹推理 + 2-3min MLLM),与文本共用
        # 一个池会把 worker 全占满、把文本消费饿死。调度按会话队头 kind 分派到这里。
        self.video_exec = obs.ContextThreadPoolExecutor(max_workers=settings.video_pool_size,
                                                        thread_name_prefix="video")
        self.recall_gate = AdmissionGate(settings.recall_pool_size)
        self._msgq: MsgQueue | None = None
        self._consumer: SessionConsumer | None = None
        self._dispatcher: Dispatcher | None = None
        # —— 画像整理池(不走队列):关段后触发 → 提交本池 → per-user 单飞锁(pseudo-session
        # "profile")。上一次没跑完,下次触发抢锁失败即返回,再下次触发自愈(幂等,读全部新 cell)。
        self.profile_exec = obs.ContextThreadPoolExecutor(max_workers=settings.profile_pool_size,
                                                          thread_name_prefix="profile")
        # user -> 绑定上下文(LRU 上限;超限逐最旧)
        self._uctx_guard = threading.Lock()
        self._uctx: OrderedDict[str, UserContext] = OrderedDict()
        # —— 视频身份 backends:进程级懒加载单例(重模型 InsightFace/ECAPA,全 drain 线程共享一份;
        # env PERSONOS_VIDEO_BACKEND gate,缺省 real)。draft 会话草稿按 user 缓存
        # (Redis 实现无状态、Memory 实现须持久于同实例)。
        self._video_backends: dict | None = None
        self._video_guard = threading.Lock()
        self._draft_guard = threading.Lock()
        self._draft_stores: "OrderedDict[str, Any]" = OrderedDict()

    # —— 多租户 ——
    def for_user(self, user_id: str) -> UserContext:
        """取某 user 的绑定上下文(LRU 缓存):其 stores 只作用于该 user。"""
        with self._uctx_guard:
            ctx = self._uctx.get(user_id)
            if ctx is None:
                ctx = UserContext(user_id, self.db)
                self._uctx[user_id] = ctx
                if len(self._uctx) > _UCTX_CAP:
                    self._uctx.popitem(last=False)
            else:
                self._uctx.move_to_end(user_id)
            return ctx

    def ctx_by_token(self, token: str) -> UserContext | None:
        """凭调用方 token 定位其上下文;未知 token → None(由 API 层回 401)。"""
        uid = self.users.user_id_by_token(token)
        return self.for_user(uid) if uid else None

    def close_user(self, user_id: str) -> None:
        # stores 共享全局池化连接,无 per-user 连接可关;仅从 LRU 中显式摘除
        with self._uctx_guard:
            self._uctx.pop(user_id, None)

    def writer_for(self, user_id: str, session_id: str,
                   max_turns: int = MAX_SEGMENT_TURNS) -> SessionWriter:
        """构一个写入状态机(无实例态:段状态在 seg_store,跨请求/跨副本共享)。"""
        ctx = self.for_user(user_id)
        return SessionWriter(self.maas, self.maas, ctx.evidence, ctx.cells, ctx.atoms,
                             session_id=session_id, user_id=user_id, max_turns=max_turns,
                             seg_store=self._seg(), chain_store=ctx.chains,
                             media_store=self._media(), mllm=self.mllm)

    def _media(self):
        """OSS 存储懒建:凭证缺失/oss2 缺失 → 返回 None(带图 ingest 仍能看图,只是原图不留底)。"""
        if self._media_store is None:
            with self._media_guard:
                if self._media_store is None:
                    try:
                        self._media_store = media_store_from_settings(settings)
                    except Exception as e:   # noqa: BLE001
                        logger.warning(f"OSS 媒体存储未装配(图片将不留底): {e}")
                        self._media_store = False   # 标记已尝试,避免每次重试
        return self._media_store or None

    # —— 跨副本会话态 ——
    def _seg(self) -> SegStore:
        if self._seg_store is None:
            with self._state_guard:
                if self._seg_store is None:
                    self._seg_store, self._session_lock = self._make_state()
        return self._seg_store

    def session_lock(self, user_id: str, session_id: str):
        """per-(user,session) 写锁:同一会话的 ingest 与 session-end 抢同一把,永不交错。

        Redis 实现 = 跨副本互斥;未配 REDIS_CLUSTER 时进程内(单副本)。返回上下文管理器。
        """
        if self._session_lock is None:
            with self._state_guard:
                if self._session_lock is None:
                    self._seg_store, self._session_lock = self._make_state()
        return self._session_lock(user_id, session_id)

    # —— 视频身份消费依赖 ——
    def video_backends(self) -> dict:
        """视频身份 backends 进程级懒加载单例(重模型只加载一次,全 drain 线程共享)。
        profile 取 env PERSONOS_VIDEO_BACKEND(**缺省 real**;mock 仅供测试——它会编造剧本
        与假人脸向量,生产跑 mock = 往真库写伪造记忆)。"""
        if self._video_backends is None:
            with self._video_guard:
                if self._video_backends is None:
                    from ..identity.backends.factory import make_backends
                    self._video_backends = make_backends()   # env gate 在 make_backends 内
                    logger.info("视频身份 backends 装配完成(进程单例)")
        return self._video_backends

    def _draft_for(self, user_id: str):
        """会话草稿存储(按 user 缓存):Redis 实现无状态(缓存仅省构造),Memory 实现须同实例持久。"""
        with self._draft_guard:
            d = self._draft_stores.get(user_id)
            if d is None:
                from ..identity.draft import MemoryDraftStore, RedisDraftStore
                d = (RedisDraftStore(get_redis(), user_id) if settings.redis_cluster
                     else MemoryDraftStore(user_id))
                self._draft_stores[user_id] = d
                if len(self._draft_stores) > _UCTX_CAP:
                    self._draft_stores.popitem(last=False)
            return d

    def video_deps(self, user_id: str):
        """装配视频消费的一整套依赖(VideoDeps):backends=进程单例,其余 per-user。"""
        from ..identity.cloud import CloudEngine
        from ..identity.store import CharacterStore
        from ..online.video_ingest import VideoDeps
        ctx = self.for_user(user_id)
        store = CharacterStore(self.db, user_id)
        return VideoDeps(
            store=store, cloud=CloudEngine(store), draft=self._draft_for(user_id),
            backends=self.video_backends(), media_store=self._media(), maas=self.maas,
            evidence=ctx.evidence, cells=ctx.cells, atoms=ctx.atoms, chains=ctx.chains)

    def visual_deps(self, user_id: str):
        """装配召回侧视觉改写的依赖(VisualDeps)。

        与视频消费共用同一份 backends 单例(重模型全进程只此一份)和同一套身份资产,
        区别是**不写草稿**——看图改写是读路径,不产生任何身份变更。
        draft 传 None:只用到 registry.candidate_card,那条路径不碰 draft。
        """
        from ..identity.cloud import CloudEngine
        from ..identity.registry import AnchorRegistry
        from ..identity.store import CharacterStore
        from ..online.visual_query import VisualDeps
        store = CharacterStore(self.db, user_id)
        cloud = CloudEngine(store)
        return VisualDeps(store=store, cloud=cloud, backends=self.video_backends(),
                          registry=AnchorRegistry(store, cloud, None,
                                                  media_store=self._media()))

    def _make_state(self) -> tuple[SegStore, SessionLock]:
        """按配置装配 (seg_store, session_lock)。Redis 故障不吞:首次使用时如实抛。"""
        if settings.redis_cluster:
            client = get_redis()
            return RedisSegStore(client), RedisSessionLock(client)
        return MemorySegStore(), MemorySessionLock()

    # —— 有序消费任务系统(Phase B) ——
    def msg_queue(self) -> MsgQueue:
        """会话消息队列(懒建):Redis 实现=跨副本;未配 REDIS_CLUSTER 时进程内单副本。"""
        if self._msgq is None:
            with self._state_guard:
                if self._msgq is None:
                    self._msgq = (RedisMsgQueue(get_redis()) if settings.redis_cluster
                                  else MemoryMsgQueue())
        return self._msgq

    def _get_consumer(self) -> SessionConsumer:
        if self._consumer is None:
            # 先在锁外把依赖建好:msg_queue()/_seg() 各自会取 _state_guard,
            # 若在本方法持锁时再调它们 → 非重入 Lock 自锁(踩过的坑)
            mq = self.msg_queue()
            self._seg()   # 确保 session_lock 已装配(与 seg 同源)
            with self._state_guard:
                if self._consumer is None:
                    self._consumer = SessionConsumer(
                        mq, self._session_lock, self.writer_for,
                        max_drain=settings.max_drain_per_cycle,
                        max_retries=settings.max_ingest_retries,
                        renew_interval_s=LOCK_TTL_S / 3,   # 心跳续锁间隔 = 锁 TTL 的 1/3
                        after_drain=self.trigger_profile,  # 关段后触发画像整理(不阻塞 ingest)
                        video_deps=self.video_deps,        # 视频消息消费依赖(backends 懒加载)
                        task_store=self.task_store)        # clip 永久失败留痕(可查/可告警)
        return self._consumer

    # —— 画像整理触发(不走队列,一 user 一把单飞锁) ——
    def trigger_profile(self, user_id: str, session_id: str = "", scenario: str = "") -> None:
        """SessionConsumer 关段后回调:算触发条件,达标则提交 profile 池。

        在 ingest 线程内调用,必须快且绝不抛(画像失败只影响新鲜度,不能拖垮 ingest 消费)。
        scenario:触发它的那批 ingest 的业务场景描述(per-request 场景透传给画像整理,不落库)。
        """
        try:
            ctx = self.for_user(user_id)
            ps = ProfileStore(self.db, user_id)
            cur = ps.current()
            cells = ctx.cells.cells_after(cur.up_to_cell_id if cur else "")
            if not cells:
                return                                   # 没有新 cell(段未闭合)→ 不触发
            ep_chars = sum(len(c.episode or "") for c in cells)
            version_count = cur.version if cur else 0    # 版本不删,号即计数
            if not should_consolidate(n_new=len(cells), ep_chars=ep_chars,
                                      version_count=version_count,
                                      ep_chars_trigger=settings.profile_ep_chars_trigger):
                return
            # 直接提交:池 worker 数即真并发上限,超出的排队(绝不丢)。per-user 单飞锁去重
            # (重复提交只是廉价空跑)。此前用 gate 满即 return 丢弃,而会话末次关段没有"下次"
            # 自愈 → 并发 > 池容量时稳定漏做若干 user 的画像,故去掉丢弃语义。
            self.profile_exec.submit(self._run_user_profile, user_id, scenario)
        except Exception:   # noqa: BLE001
            logger.exception(f"画像触发失败 user={user_id}(不影响 ingest)")

    def _run_user_profile(self, user_id: str, scenario: str = "") -> None:
        """在 profile 池执行:抢 per-user 单飞锁 → 整理出版本 → 释放。抢不到即返回(下次触发自愈)。"""
        try:
            token = self._session_lock.try_acquire(user_id, "profile") if self._session_lock else None
            if not token:
                return                                   # 另一个整理在跑,交给它(幂等)
            try:
                ctx = self.for_user(user_id)
                run_user_consolidation(self.maas, cells_store=ctx.cells, atoms_store=ctx.atoms,
                                       profile_store=ProfileStore(self.db, user_id),
                                       today=now().date(), scenario=scenario)
            finally:
                self._session_lock.release(user_id, "profile", token)
        except Exception:   # noqa: BLE001
            logger.exception(f"画像整理失败 user={user_id}")

    def enqueue_message(self, user_id: str, session_id: str, payload: dict,
                        *, kind: str = "ingest") -> tuple[str, int]:
        """入队一条会话消息(ingest/session_end),立即返回 (msg_id, seq)。消费由 dispatcher 异步驱动。"""
        return self.msg_queue().enqueue(user_id, session_id, payload, kind=kind)

    def queue_depth(self, user_id: str, session_id: str) -> int:
        return self.msg_queue().depth(user_id, session_id)

    def drain_once(self, user_id: str, session_id: str):
        """同步消费一个会话(单副本/测试/本地脚本用;不依赖后台 dispatcher)。"""
        return self._get_consumer().drain_session(user_id, session_id)

    def start_dispatcher(self) -> None:
        """启动后台调度器(FastAPI 启动钩子调用;测试/脚本不调 → 用 drain_once 手动驱动)。"""
        if self._dispatcher is None:
            self._dispatcher = Dispatcher(
                self.msg_queue(), self._get_consumer(), self.ingest_exec,
                settings.ingest_pool_size,
                tick_s=settings.dispatcher_tick_s, idle_tick_s=settings.dispatcher_idle_tick_s,
                video_pool=self.video_exec, video_cap=settings.video_pool_size)
        self._dispatcher.start()

    def stop_dispatcher(self) -> None:
        if self._dispatcher is not None:
            self._dispatcher.stop()

    @staticmethod
    def _warn_redis_env_mismatch() -> None:
        """启动即亮明会话态落点;非 sit 环境仍用 sit 默认 Redis 集群时大声告警。

        REDIS_CLUSTER 代码默认 sns-redis-sit(sit 部署零配置),代价是 prod 忘了
        覆盖会静默连 sit 集群——这个告警把"忘了"从静默变成启动日志里的明喇叭。
        """
        env = os.environ.get("XHS_ENV", "")
        if not settings.redis_cluster:
            logger.warning("REDIS_CLUSTER 未配置:进程内单副本模式(多副本部署不可用!)")
            return
        logger.info(f"跨副本会话态:Redis 集群 {settings.redis_cluster}"
                    f"(env={env or 'local'},key 前缀 {env or 'local'}:personos:*)")
        if env and env not in ("sit", "local") and settings.redis_cluster == "sns-redis-sit":
            logger.warning(f"XHS_ENV={env} 但 Redis 仍是 SIT 默认集群 sns-redis-sit!"
                           f"prod 部署须在环境变量覆盖 REDIS_CLUSTER(prod 集群中段名)")

    # —— 异步基建(状态落 tasks 表,跨副本可查) ——
    def submit_task(self, kind: str, user_id: str, session_id: str, fn) -> str:
        """把 fn 丢后台线程池执行,状态落 tasks 表供任意副本查询。返回 task_id。

        在途满时抛 TaskOverloaded(API 层转 503)——先拒绝再建行,不留卡 pending 的孤儿行。
        """
        if not self._admission.try_enter():
            raise TaskOverloaded(f"后台任务在途已满(≤{_MAX_PENDING}),请稍后重试")
        tid = uuid.uuid4().hex
        self.task_store.create(tid, kind, user_id, session_id)
        self._maybe_sweep()

        def run():
            self.task_store.mark_running(tid)
            try:
                res = fn()
                self.task_store.mark_done(tid, res)
            except Exception as e:  # noqa: BLE001  后台失败登记进任务表,不崩线程池
                logger.exception(f"后台任务失败 kind={kind} user={user_id} session={session_id}")
                self.task_store.mark_error(tid, str(e))
            finally:
                self._admission.leave()

        try:
            self._executor.submit(run)
        except Exception:   # 提交失败(如进程收尾 shutdown):名额必须归还
            self._admission.leave()
            raise
        return tid

    def _maybe_sweep(self) -> None:
        """低频触发任务表清扫(僵尸收割+过期清理);竞态无害(幂等操作)。"""
        if time.time() - self._sweep_at < _SWEEP_INTERVAL_S:
            return
        self._sweep_at = time.time()
        try:
            self.task_store.sweep()
        except Exception:  # noqa: BLE001  清扫失败不影响提交,下个周期重试
            logger.exception("任务表清扫失败")

    def get_task(self, task_id: str) -> dict | None:
        return self.task_store.get(task_id)


rt = Runtime()
