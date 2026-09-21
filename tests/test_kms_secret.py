"""config.get_secret:env 优先 / KMS 兜底 / required 抛错。

用 fake redkms 注入 sys.modules,不依赖真 KMS。pytest 下(PERSONOS_TEST_GUARD=1)默认
短路走 env——本文件按需 delenv 该守卫来验证 KMS 兜底分支。
"""

from __future__ import annotations

import sys
import types

import pytest

from personos.config import get_secret


def _fake_redkms(monkeypatch, value=None, boom=False):
    """注入假的 redkms 模块;value=None 且非 boom 时 get_secret_value 返回空串。"""
    m = types.ModuleType("redkms")
    def _gsv(key):
        if boom:
            raise RuntimeError(f"KMS 不可达: {key}")
        return value or ""
    m.get_secret_value = _gsv
    monkeypatch.setitem(sys.modules, "redkms", m)


def test_env_first_wins(monkeypatch):
    monkeypatch.setenv("PERSONOS_X", "from-env")
    _fake_redkms(monkeypatch, value="from-kms")   # 即便 KMS 有值,env 优先
    assert get_secret("PERSONOS_X") == "from-env"


def test_test_guard_skips_kms(monkeypatch):
    """pytest 守卫下:env 缺失也不碰 KMS,直接空串(required 亦不抛)——测试零外部依赖。"""
    monkeypatch.setenv("PERSONOS_TEST_GUARD", "1")
    monkeypatch.delenv("PERSONOS_Y", raising=False)
    called = {"n": 0}
    m = types.ModuleType("redkms")
    m.get_secret_value = lambda k: called.__setitem__("n", called["n"] + 1) or "x"
    monkeypatch.setitem(sys.modules, "redkms", m)
    assert get_secret("PERSONOS_Y", required=True) == ""
    assert called["n"] == 0                         # KMS 根本没被调用


def test_kms_fallback_when_env_absent(monkeypatch):
    monkeypatch.delenv("PERSONOS_TEST_GUARD", raising=False)   # 关守卫,走 KMS 分支
    monkeypatch.delenv("PERSONOS_Z", raising=False)
    _fake_redkms(monkeypatch, value="from-kms")
    assert get_secret("PERSONOS_Z") == "from-kms"


def test_kms_key_alias_override(monkeypatch):
    monkeypatch.delenv("PERSONOS_TEST_GUARD", raising=False)
    monkeypatch.delenv("PERSONOS_W", raising=False)
    seen = {}
    m = types.ModuleType("redkms")
    m.get_secret_value = lambda k: seen.setdefault("key", k) or "v"
    monkeypatch.setitem(sys.modules, "redkms", m)
    get_secret("PERSONOS_W", kms_key="kms.custom.alias")
    assert seen["key"] == "kms.custom.alias"          # 用传入别名,而非 env 名


def test_required_raises_when_both_missing(monkeypatch):
    monkeypatch.delenv("PERSONOS_TEST_GUARD", raising=False)
    monkeypatch.delenv("PERSONOS_MISS", raising=False)
    _fake_redkms(monkeypatch, value="")               # KMS 也空
    with pytest.raises(RuntimeError, match="均无值"):
        get_secret("PERSONOS_MISS", required=True)


def test_required_raises_when_kms_errors(monkeypatch):
    monkeypatch.delenv("PERSONOS_TEST_GUARD", raising=False)
    monkeypatch.delenv("PERSONOS_ERR", raising=False)
    _fake_redkms(monkeypatch, boom=True)
    with pytest.raises(RuntimeError, match="读取失败"):
        get_secret("PERSONOS_ERR", required=True)


def test_optional_returns_empty_on_miss(monkeypatch):
    monkeypatch.delenv("PERSONOS_TEST_GUARD", raising=False)
    monkeypatch.delenv("PERSONOS_OPT", raising=False)
    _fake_redkms(monkeypatch, boom=True)              # 取不到也不抛
    assert get_secret("PERSONOS_OPT", required=False) == ""


def test_apollo_cache_redirected_to_writable(monkeypatch, tmp_path):
    """pyapollo 默认缓存在 site-packages(容器非 root 不可写)→ patch 重定向到可写目录。"""
    pytest.importorskip("pyapollo")
    from pyapollo import apollo_client as ac
    from personos.config import _ensure_apollo_cache_writable
    monkeypatch.setenv("PERSONOS_APOLLO_CACHE", str(tmp_path / "apollo"))
    monkeypatch.setattr(ac.ApolloClient, "_personos_patched", False, raising=False)
    _ensure_apollo_cache_writable()
    c = ac.ApolloClient(app_id="x", config_server_url="http://127.0.0.1:1")
    assert str(tmp_path) in c._cache_file_path        # 不在 site-packages,落到可写目录
