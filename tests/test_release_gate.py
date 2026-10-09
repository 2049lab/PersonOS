"""The release gate must distinguish public providers from internal identifiers."""

from pathlib import Path

import pytest

from scripts import check_no_internal_refs as gate

# Build blocked fixtures at runtime so this test does not ship the identifiers
# that the release gate is intended to catch in real source files.
GATEWAY = "".join(("ma", "as"))


@pytest.mark.parametrize("text", [
    f"https://workspace.cn-beijing.{GATEWAY}.aliyuncs.com/api/v1/rerank",
    f"https://workspace.cn-beijing.{GATEWAY.upper()}.ALIYUNCS.COM/api/v1/rerank",
])
def test_public_provider_domain_is_allowed(tmp_path, monkeypatch, text):
    source = tmp_path / "provider.py"
    source.write_text(text, encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_files", lambda: [source])

    assert gate.scan() == {}


@pytest.mark.parametrize("text", [
    f"use the {GATEWAY} gateway",
    f"{GATEWAY.upper()}_API_KEY=value",
    f"https://{GATEWAY}.internal.example/api",
    f"https://{GATEWAY}.aliyuncs.com.evil.example/api",
    f"https://{GATEWAY}.aliyuncs.company/api",
    f"https://{GATEWAY}.aliyuncs.com-attacker.example/api",
    f"https://{GATEWAY}.aliyuncs.com/api and {GATEWAY.upper()}_API_KEY=value",
])
def test_internal_identifier_and_domain_lookalikes_are_blocked(tmp_path, monkeypatch, text):
    source = tmp_path / "provider.py"
    source.write_text(text, encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_files", lambda: [source])

    hits = gate.scan()
    assert len(hits) == 1
    assert next(iter(hits.values())) == [(Path("provider.py"), 1, text[:110])]
