"""开源泄漏闸:仓库里不得残留任何公司内部标识。

为什么要自动化:内部标识不是只藏在显眼的配置里 —— 它散在 docstring 的"移植自 mneme"、
注释里的"RedHub 代理"、依赖名 redkms、env 名 XHS_ENV、异常消息的内网域名。人工 review
必然漏,而开源是**不可逆**的:推上去被爬了就收不回来。

用法:
    python scripts/check_no_internal_refs.py            # 报告命中,有命中则退出码 1
    python scripts/check_no_internal_refs.py --list     # 按模式分组列清单(当工作清单用)

作为 pre-commit 钩子常驻。改造期间它会一直红 —— 这是预期的,红的条目就是待办清单;
到发布前必须全绿。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 模式 → 为什么它不能出现。写清楚理由,免得后人以为是洁癖而随手加白名单。
PATTERNS: dict[str, str] = {
    r"xiaohongshu": "公司域名",
    r"\bxhs\b|XHS_": "公司简称/env 前缀",
    r"redkms|redenv|redmetrics|redinfra|python-infra-framework|\bhyx\b":
        "内网私有包(公网装不上)",
    r"RedHub|redhub": "内网数据库代理",
    r"\bcorvus\b": "内网 Redis 集群实现",
    r"\bmaas\b|MAAS_": "内网模型网关",
    r"\bapollo\b|pyapollo": "内网配置中心",
    r"xray-langfuse|xray": "内网可观测平台",
    r"mneme|meme-backend|wallace": "内部项目代号",
    r"DMS 工单|内部系统开发安全规范": "内部流程/规范文档名",
}

# 跳过:二进制、产物、本文件自身(它当然含这些词)
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "baseline", "data", "logs",
             ".venv", "venv", "node_modules", ".idea"}
SKIP_SUFFIX = {".pyc", ".gz", ".png", ".jpg", ".jpeg", ".ico", ".lock", ".json"}
SKIP_FILES = {"check_no_internal_refs.py"}


def _files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.name in SKIP_FILES or p.suffix in SKIP_SUFFIX:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


def scan() -> dict[str, list[tuple[Path, int, str]]]:
    hits: dict[str, list[tuple[Path, int, str]]] = defaultdict(list)
    compiled = {pat: re.compile(pat, re.IGNORECASE) for pat in PATTERNS}
    for f in _files():
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat, rx in compiled.items():
                if rx.search(line):
                    hits[pat].append((f.relative_to(ROOT), i, line.strip()[:110]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="列出全部命中(当工作清单用)")
    a = ap.parse_args()

    hits = scan()
    if not hits:
        print("✅ 无内部标识残留")
        return 0

    total = sum(len(v) for v in hits.values())
    files = {f for v in hits.values() for f, _, _ in v}
    print(f"❌ 命中 {total} 处内部标识,分布在 {len(files)} 个文件\n")
    for pat, items in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        by_file: dict[Path, int] = defaultdict(int)
        for f, _, _ in items:
            by_file[f] += 1
        print(f"── {PATTERNS[pat]}  ({len(items)} 处 / {len(by_file)} 文件)  /{pat}/")
        for f, n in sorted(by_file.items(), key=lambda kv: -kv[1])[:8 if not a.list else 999]:
            print(f"     {n:>4}  {f}")
        if not a.list and len(by_file) > 8:
            print(f"          …另有 {len(by_file) - 8} 个文件(--list 看全部)")
        if a.list:
            for f, i, line in items[:40]:
                print(f"          {f}:{i}  {line}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
