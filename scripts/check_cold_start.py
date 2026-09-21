"""冷启动闸:空环境下 import 必须成功,一行配置都不给。

这是"零配置可用"承诺的唯一硬保障。挡的是**导入期求值** —— 模块在 import 阶段就去读
密钥/连数据库/建客户端。一旦有,`pip install personos` 后的第一行 `from personos import
Memory` 就炸,再漂亮的 README 也没用。

为什么不能只测 `import personos`:包的 `__init__.py` 可能几乎是空的,那样这个检查
永远绿、永远抓不到问题(实测过——空环境下 `import personos` 通过,而
`import personos.online.write_path` 直接抛"密钥缺失")。所以这里**显式列出代表性模块**,
覆盖配置、存储、写入、召回四条线。

用法:python scripts/check_cold_start.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 代表性模块:每条覆盖一类导入期风险。新增子系统时请一并加进来。
MODULES = [
    ("personos", "包入口(Memory 门面最终在这里)"),
    ("personos.config", "配置层——导入期求值的高发区"),
    ("personos.models", "纯数据模型,任何时候都该能导"),
    ("personos.storage.db", "存储层:不得在 import 时连库"),
    ("personos.online.write_path", "写入链路"),
    ("personos.online.retrieval", "召回链路"),
]


def main() -> int:
    py = sys.executable
    # env -i 的等价物:只留解释器能启动所必需的,其余一律清空
    clean = {"PATH": os.environ.get("PATH", ""), "HOME": "/tmp/personos-coldstart"}
    if sys.platform == "win32":
        clean["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")

    failed = []
    for mod, why in MODULES:
        r = subprocess.run([py, "-c", f"import {mod}"], env=clean, cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"✅ {mod}")
        else:
            tail = (r.stderr or "").strip().splitlines()
            print(f"❌ {mod}  ({why})\n     {tail[-1] if tail else '未知错误'}")
            failed.append(mod)

    if failed:
        print(f"\nCOLD_START FAIL — {len(failed)}/{len(MODULES)} 个模块在空环境下无法导入。"
              f"\n零配置承诺不成立:这些模块在 import 阶段就要求配置。"
              f"\n典型原因:模块级 `settings = load_settings()`、把 settings 当默认参数、"
              f"模块级客户端单例。")
        return 1
    print(f"\nCOLD_START PASS — {len(MODULES)} 个模块均可在空环境下导入")
    return 0


if __name__ == "__main__":
    sys.exit(main())
