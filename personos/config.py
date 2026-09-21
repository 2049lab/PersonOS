"""集中配置:从环境变量/.env 读取,禁止在代码里硬编码密钥。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录(本文件在 personos/ 下,上一级即根)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    maas_base_url: str
    chat_key: str
    chat_model: str
    app_id: str
    embedding_key: str
    embedding_model: str
    embedding_dim: int
    rerank_key: str      # R2 精排模型 key(空 = 复用 chat_key,网关同一鉴权)
    rerank_model: str
    maas_chat_timeout: float   # chat 读超时(非流式+大 max_tokens,生成 60s+ 是常态,要宽)
    maas_io_timeout: float     # embed/rerank 读超时(快接口,紧一点好尽早暴露故障)
    deep_write: bool           # 深轨 remember 写回开关(关掉做评测对照;关时深轨只读)
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    # Redis(corvus,经 redinfra 服务发现接入):空串 = 未启用(seg/写锁退进程内单副本实现)。
    # 值为集群名中段,SDK 自动拼 corvus- 前缀(平台 UI「redis@<这段>@online」取中段)。
    redis_cluster: str
    # —— 有序消费任务系统(Phase B)——
    # 按类型分池,互不挤占;dispatcher 从会话队列 FIFO drain。压测时可调小 pool 便于 mock。
    ingest_pool_size: int      # ingest 消费池并发上限(drain 作业)
    # 视频消费池并发上限(独立于文本池:clip 重活不饿死文本)。**视频并发的唯一旋钮**——
    # video_ingest 里曾有第二道信号量(4),两道闸管同一件事、生效的永远是更严的那道,已删。
    # 选值要算的账:单 clip 峰值占一个临时文件(上限=PERSONOS_VIDEO_UPSTREAM_MAX_BYTES,默认
    # 64MB);外链路径这文件从下载一直握到 harvest 结束,跨整个剧本 MLLM(约 90s)。
    # 故最坏临时盘 ≈ video_pool_size × 64MB —— 50 即 3.2GB,须确认 pod 磁盘容得下。
    video_pool_size: int
    recall_pool_size: int      # recall 独立池并发上限(快照读,与 ingest 隔离)
    profile_pool_size: int     # 画像池并发上限(consolidate 作业;一 user 一把单飞锁)
    profile_ep_chars_trigger: int  # 单次 consolidate 输入上界:新 cell 的 episode 累计字数达此即触发
    # 单次抢锁最多消费的消息数(公平阀门:话痨会话消够这么多就归还池名额,回看板与别人平等重排)。
    # **必须显著小于 max_queue_depth**,否则阀门形同虚设:队列最多堆 max_queue_depth 条,drain
    # 会先在"队列空"处 break,永远走不到这个上限,结果就是"占住名额直到该会话排空"。
    # 曾经 20 > 15 正是这种失效态——视频场景下单个会话最坏能独占一个名额十几小时
    # (15 批 × 每批 20 clip × 约 216s)。取 5:视频最坏占用降到约 1/3,文本侧开销可忽略。
    max_drain_per_cycle: int
    max_ingest_retries: int    # 一条消息连续失败达此次数 → 判毒消息,跳过(推进游标)+ ERROR 告警
    max_queue_depth: int       # 单会话队列积压上限,超则 ingest 背压 503
    dispatcher_tick_s: float   # 调度器有活时轮询间隔
    dispatcher_idle_tick_s: float  # 调度器空闲时轮询间隔(省 CPU)
    log_dir: Path
    # MiniMax(评测等场景的第二 LLM 供应商;Anthropic 兼容接口)。未配置时不可用。
    minimax_base_url: str
    minimax_api_key: str
    minimax_chat_model: str
    # MLLM(qwen3.5-omni-plus 等;OpenAI 兼容多模态,图片理解用)。key 空 = 图片理解不可用。
    mllm_endpoint: str
    mllm_model: str
    mllm_key: str
    mllm_timeout: float
    # OSS(阿里云;原图对象存储)。access_key 空 = 图片存储不可用(纯文本链路不受影响)。
    oss_access_key_id: str
    oss_access_key_secret: str
    oss_bucket: str
    oss_endpoint: str      # 上传/回源域名(服务端在 VPC 内走 -internal)
    oss_region: str
    oss_prefix: str        # 对象 key 前缀,与其它业务隔离(如 "personos/")
    oss_url_expires_seconds: int   # 溯源返图签名 URL 有效期(秒)
    # Langfuse(公司 xray LLM 可观测)。pk/sk 空 = 关闭(埋点全无操作,不上报)。
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str             # 上报 host(…/langfuse-microapp)
    langfuse_environment: str      # prod / sit / local(链路环境标)
    langfuse_release: str          # 版本标(可空)


def _abs(p: str) -> Path:
    """相对路径一律相对项目根,避免受运行目录影响。"""
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _ensure_apollo_cache_writable() -> None:
    """redkms 底层 pyapollo 默认把缓存目录建在 site-packages 内(dirname(__file__)/config),
    容器非 root 无写权限 → PermissionError。pyapollo 只认 cache_file_path 参数,而 redkms 不透传,
    故 import 前 monkeypatch ApolloClient.__init__,把缓存重定向到可写目录(默认 /tmp)。幂等。"""
    try:
        from pyapollo import apollo_client as _ac
    except Exception:                             # noqa: BLE001  未装 pyapollo(走 env 的部署)→ 无需 patch
        return
    if getattr(_ac.ApolloClient, "_personos_patched", False):
        return
    cache_dir = os.environ.get("PERSONOS_APOLLO_CACHE", "/tmp/personos-apollo-config")
    _orig = _ac.ApolloClient.__init__

    def _init(self, *a, **kw):
        kw.setdefault("cache_file_path", cache_dir)   # 覆盖 site-packages 内的默认路径
        _orig(self, *a, **kw)

    _ac.ApolloClient.__init__ = _init
    _ac.ApolloClient._personos_patched = True


def get_secret(env_name: str, kms_key: str | None = None, *, required: bool = True) -> str:
    """读密钥:env 优先(本地/CI/.env 明文),缺失则走 KMS(redkms)兜底。

    向前兼容:只要部署仍注入明文 env,就永远走 env 分支、redkms 根本不 import;
    把某个 env 从部署移除后,才回退到 KMS 解密(线上无明文的目标态)。key 别名默认同 env 名。
    - required=True:env 与 KMS 都取不到 → 抛错(启动即失败,防带残缺密钥上线);
    - required=False:取不到返回 ""(可选能力,如 rerank/mllm/oss/langfuse 缺失则降级)。
    """
    v = os.environ.get(env_name)
    if v:
        return v
    # pytest 下绝不碰 KMS(测试零外部依赖),行为等同旧的缺省空串
    if os.environ.get("PERSONOS_TEST_GUARD") == "1":
        return ""
    key = kms_key or env_name
    try:
        _ensure_apollo_cache_writable()          # 修 pyapollo 缓存目录权限坑(容器非 root)
        from redkms import get_secret_value      # 懒导:走 env 的部署无需装 redkms
        val = get_secret_value(key)
    except Exception as e:                        # noqa: BLE001
        if required:
            raise RuntimeError(f"密钥 {env_name} 在 env 缺失且 KMS({key}) 读取失败: {e}") from e
        return ""
    if not val and required:
        raise RuntimeError(f"密钥 {env_name} 在 env 与 KMS({key}) 均无值")
    return val or ""


# —— 环境护栏:XHS_ENV 一值两用(基建连接选址 + Redis 键前缀)。填错的两种后果:
#   ① 非法值(拼错)→ EDS 查表 KeyError,readyz 挂,尚能拦;
#   ② 合法但错环境(prod 配成 sit)→ 静默连到 sit 基建、用 sit: 键前缀,跨环境串库,无任何报错。
# 本护栏把 ② 变成"起不来":容器模式(XHS_K8S)下强制 XHS_ENV 合法,且 REDIS_CLUSTER 不含别环境标记。
_VALID_ENVS = ("sit", "staging", "prod")
# 各环境在资源命名里的可识别标记(staging 亦作 beta)
_ENV_MARKERS = {"sit": ("sit",), "staging": ("staging", "beta"), "prod": ("prod",)}


def _has_marker(text: str, marker: str) -> bool:
    """资源名里是否出现被分隔符界定的环境标记(避免 transit 命中 sit)。"""
    return re.search(rf"(?<![a-z0-9]){marker}(?![a-z0-9])", text.lower()) is not None


def _check_env_consistency(env: str, redis_cluster: str) -> None:
    """纯校验(可单测):XHS_ENV 须合法,且 REDIS_CLUSTER 不得含"别的环境"标记。

    只查 Redis——它是唯一靠 XHS_ENV 前缀隔离、配错会静默撞库的资源(命中别环境标记 = 疑似复制配置忘改)。
    不查 MySQL:staging 共用 prod DB 主机、靠独立 database/账号隔离(主机名带 prod 是正常的),按主机名判环境会误伤。
    """
    if env not in _VALID_ENVS:
        raise RuntimeError(
            f"XHS_ENV 必须显式设为 {_VALID_ENVS} 之一(当前: {env!r})——"
            "它同时决定基建连接选址与 Redis 键前缀,缺省会静默连到 sit 基建并撞库")
    foreign = sorted({e for e, ms in _ENV_MARKERS.items() if e != env
                      for m in ms if _has_marker(redis_cluster, m)})
    if foreign:
        raise RuntimeError(
            f"环境不一致:XHS_ENV={env} 但 REDIS_CLUSTER={redis_cluster!r} 含其它环境标记 {foreign}——"
            "疑似复制了别环境的配置未改,拒绝启动以防跨环境串库")


def _maybe_guard_online(s: "Settings") -> None:
    """仅容器/线上(XHS_K8S 有值)强制护栏;本地与 pytest 不设该变量,行为不变。"""
    if not os.environ.get("XHS_K8S"):
        return
    _check_env_consistency(os.environ.get("XHS_ENV", ""), s.redis_cluster)


def load_settings() -> Settings:
    s = Settings(
        maas_base_url=os.environ.get("MAAS_BASE_URL", "https://maas.devops.xiaohongshu.com/v1"),
        chat_key=get_secret("MAAS_CHAT_KEY"),
        chat_model=os.environ.get("MAAS_CHAT_MODEL", "deepseek-v4-pro"),
        app_id=os.environ.get("MAAS_APP_ID", "qs-api"),
        embedding_key=get_secret("MAAS_EMBEDDING_KEY"),
        embedding_model=os.environ.get("MAAS_EMBEDDING_MODEL", "qwen3-embedding-8b"),
        embedding_dim=int(os.environ.get("MAAS_EMBEDDING_DIM", "4096")),
        rerank_key=get_secret("MAAS_RERANK_KEY", required=False),   # 空 → 客户端回落 chat_key
        rerank_model=os.environ.get("MAAS_RERANK_MODEL", "qwen3-reranker-0.6b"),
        maas_chat_timeout=float(os.environ.get("MAAS_CHAT_TIMEOUT", "120")),
        maas_io_timeout=float(os.environ.get("MAAS_IO_TIMEOUT", "30")),
        deep_write=os.environ.get("PERSONOS_DEEP_WRITE", "1").lower() not in ("0", "false", "no"),
        # MySQL(RedHub): sit/prod 由部署平台注入不同 MYSQL_* 环境变量,本地 .env 覆盖。
        # 默认值指向 SIT(仅 host/port/user 无敏感信息);密码绝不硬编码。
        mysql_host=os.environ.get("MYSQL_HOST", "redhub-clb-2049lab-sit-db1.int.xiaohongshu.com"),
        mysql_port=int(os.environ.get("MYSQL_PORT", "5066")),
        mysql_user=os.environ.get("MYSQL_USER", "personos"),
        mysql_password=get_secret("MYSQL_PASSWORD"),
        mysql_database=os.environ.get("MYSQL_DATABASE", "personos"),
        # 默认指向 SIT 集群(与 MYSQL_* 默认值同规);prod 部署平台注入同名 env 覆盖
        redis_cluster=os.environ.get("REDIS_CLUSTER", "sns-redis-sit"),
        ingest_pool_size=int(os.environ.get("PERSONOS_INGEST_POOL", "50")),
        # 8 而非 50:视频并发的瓶颈不在并发度——单 clip 的 ~91s 里 ~67s 是等上游 MLLM 返回
        # (纯 IO 等待,不吃本地 CPU),真吃资源的只有 harvest 的 ~16s。8 路兼顾三笔账:
        # 临时盘 8×64MB=512MB、上游 8 路视频并发不至于触发限流、本地推理不打满 CPU。
        # 吞吐 ≈ 8 clip/91s,按 2min clip 算单 pod 可跟住约 10 路实时流;要更高请加 pod。
        video_pool_size=int(os.environ.get("PERSONOS_VIDEO_POOL", "8")),
        recall_pool_size=int(os.environ.get("PERSONOS_RECALL_POOL", "50")),
        profile_pool_size=int(os.environ.get("PERSONOS_PROFILE_POOL", "50")),
        profile_ep_chars_trigger=int(os.environ.get("PERSONOS_PROFILE_EP_CHARS", "15000")),
        max_drain_per_cycle=int(os.environ.get("PERSONOS_MAX_DRAIN", "5")),
        max_ingest_retries=int(os.environ.get("PERSONOS_MAX_INGEST_RETRIES", "5")),
        max_queue_depth=int(os.environ.get("PERSONOS_MAX_QUEUE_DEPTH", "15")),
        dispatcher_tick_s=float(os.environ.get("PERSONOS_DISPATCH_TICK", "0.05")),
        dispatcher_idle_tick_s=float(os.environ.get("PERSONOS_DISPATCH_IDLE_TICK", "0.5")),
        log_dir=_abs(os.environ.get("PERSONOS_LOG_DIR", "logs")),
        minimax_base_url=os.environ.get("MINIMAX_BASE_URL", ""),
        minimax_api_key=os.environ.get("MINIMAX_API_KEY", ""),
        minimax_chat_model=os.environ.get("MINIMAX_CHAT_MODEL", "MiniMax-M3"),
        # MLLM:endpoint/model/key 皆可配;未配 key 时图片理解降级(返回空,不阻塞写入)
        mllm_endpoint=os.environ.get(
            "MAAS_MLLM_ENDPOINT",
            "https://maas.devops.xiaohongshu.com/openai/openai/qwen/v1/chat/completions"),
        mllm_model=os.environ.get("MAAS_MLLM_MODEL", "qwen3.5-omni-plus"),
        mllm_key=get_secret("MAAS_MLLM_KEY", required=False),
        mllm_timeout=float(os.environ.get("MAAS_MLLM_TIMEOUT", "120")),
        # OSS:原图对象存储;沿用 .env 既有 OSS_* 键名
        oss_access_key_id=get_secret("OSS_ACCESS_KEY_ID", required=False),
        oss_access_key_secret=get_secret("OSS_ACCESS_KEY_SECRET", required=False),
        oss_bucket=os.environ.get("OSS_BUCKET", ""),
        oss_endpoint=os.environ.get("OSS_ENDPOINT", ""),
        oss_region=os.environ.get("OSS_REGION", "cn-shanghai"),
        oss_prefix=os.environ.get("OSS_PREFIX", "personos/"),
        oss_url_expires_seconds=int(os.environ.get("OSS_URL_EXPIRES_SECONDS", "3600")),
        langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=get_secret("LANGFUSE_SECRET_KEY", required=False),
        langfuse_host=os.environ.get("LANGFUSE_BASE_URL", "https://xray-langfuse.devops.xiaohongshu.com/langfuse-microapp"),
        langfuse_environment=os.environ.get("LANGFUSE_ENVIRONMENT", os.environ.get("XHS_ENV", "local")),
        langfuse_release=os.environ.get("LANGFUSE_RELEASE", ""),
    )
    _maybe_guard_online(s)
    return s


settings = load_settings()
