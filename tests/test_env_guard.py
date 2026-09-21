"""环境护栏:XHS_ENV 填错(非法值 / 与 Redis 集群名矛盾)时拒绝启动,防跨环境串库。

只查 Redis(靠 XHS_ENV 前缀隔离、会静默撞库);不查 MySQL——staging 共用 prod DB 主机、
靠独立 database/账号隔离,主机名带 prod 是正常的,按主机名判环境会误伤。

纯校验函数 _check_env_consistency 直接测(不依赖 XHS_K8S);
_maybe_guard_online 测"仅容器模式生效、本地/测试放行"。
"""

from __future__ import annotations

import pytest

from personos.config import Settings, _check_env_consistency, _maybe_guard_online


# —— 纯校验:合法组合放行 ——

@pytest.mark.parametrize("env,cluster", [
    ("sit", "sns-redis-sit"),
    ("staging", "sns-redis-staging"),
    ("staging", "shequ-2049lab"),      # staging 共用 prod Redis(无环境标记的集群名)
    ("prod", "shequ-2049lab"),         # prod 用同一集群,靠 XHS_ENV 前缀隔离
    ("prod", "sns-redis-prod"),
])
def test_consistent_env_passes(env, cluster):
    _check_env_consistency(env, cluster)   # 不抛即通过


# —— 纯校验:非法 XHS_ENV ——

@pytest.mark.parametrize("bad", ["", "production", "PROD", "prd", "beta", "local"])
def test_invalid_env_rejected(bad):
    with pytest.raises(RuntimeError, match="XHS_ENV 必须"):
        _check_env_consistency(bad, "shequ-2049lab")


# —— 纯校验:合法但错环境(Redis 集群名带别环境标记 = 复制配置忘改)——

def test_prod_env_with_sit_redis_rejected():
    """最要命的场景:prod 部署却连着 sit 的 Redis 集群。"""
    with pytest.raises(RuntimeError, match="环境不一致.*REDIS_CLUSTER"):
        _check_env_consistency("prod", "sns-redis-sit")


def test_sit_env_with_prod_redis_rejected():
    with pytest.raises(RuntimeError, match="环境不一致"):
        _check_env_consistency("sit", "sns-redis-prod")


# —— MySQL 主机名不参与判断:staging 跑在 prod DB 主机上是合法的 ——

def test_staging_on_prod_db_host_is_fine():
    """真实 staging 配置:MYSQL_HOST 是 prod 主机,但 Redis 集群无 sit/prod 标记 → 放行。"""
    _check_env_consistency("staging", "shequ-2049lab")


def test_staging_beta_marker_is_own_not_foreign():
    """staging 亦作 beta:beta 标记是自己的,不应判为别环境。"""
    _check_env_consistency("staging", "sns-redis-beta")


def test_marker_boundary_no_false_positive():
    """transit 里的 'sit' 不是环境标记,不应误伤。"""
    _check_env_consistency("prod", "corvus-transit-cache")


# —— 护栏开关:仅容器模式生效 ——

def _mk(cluster: str) -> Settings:
    """构造一个 redis_cluster 可控的 Settings(其余字段填占位,不参与校验)。"""
    kw = {}
    for name, f in Settings.__dataclass_fields__.items():
        t = f.type
        kw[name] = 0 if t == "int" else 0.0 if t == "float" else False if t == "bool" else ""
    kw["redis_cluster"] = cluster
    from pathlib import Path
    kw["log_dir"] = Path("logs")
    return Settings(**kw)


def test_guard_skipped_without_xhs_k8s(monkeypatch):
    """本地/测试:未设 XHS_K8S → 放行,即便配置矛盾也不抛(不影响开发)。"""
    monkeypatch.delenv("XHS_K8S", raising=False)
    monkeypatch.setenv("XHS_ENV", "prod")
    _maybe_guard_online(_mk("sns-redis-sit"))   # 矛盾但跳过 → 不抛


def test_guard_enforced_in_container(monkeypatch):
    """容器模式:设了 XHS_K8S → 强制校验,矛盾即抛。"""
    monkeypatch.setenv("XHS_K8S", "1")
    monkeypatch.setenv("XHS_ENV", "prod")
    with pytest.raises(RuntimeError, match="环境不一致"):
        _maybe_guard_online(_mk("sns-redis-sit"))


def test_guard_container_valid_passes(monkeypatch):
    monkeypatch.setenv("XHS_K8S", "1")
    monkeypatch.setenv("XHS_ENV", "staging")
    _maybe_guard_online(_mk("shequ-2049lab"))   # staging 用共享集群 → 不抛
