"""容器/生产启动入口:`python -m personos`。

单进程 uvicorn(不 reload),host/port 从 env 读(默认 0.0.0.0:8080,对齐 XHS 平台约定)。
本地 Debug 打断点用 run_dev.py;热重载用 `uvicorn server.app:app --reload`。
"""

from __future__ import annotations

import os

import uvicorn


def main():
    host = os.environ.get("PERSONOS_HOST", "0.0.0.0")
    port = int(os.environ.get("PERSONOS_PORT", "8080"))
    log_level = os.environ.get("PERSONOS_LOG_LEVEL", "info").lower()
    # 延迟 import:确保 MYSQL_* / PERSONOS_LOG_DIR 等 env 先于 settings/rt 初始化生效
    from server.app import app
    from personos.config import settings
    from loguru import logger

    logger.info(f"PersonOS 启动 http://{host}:{port} · "
                f"db=mysql://{settings.mysql_host}/{settings.mysql_database} · "
                f"log_dir={settings.log_dir}")
    uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)


if __name__ == "__main__":
    main()
