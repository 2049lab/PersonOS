"""Container/production entrypoint: `python -m personos`.

Single-process uvicorn (no reload); host/port come from env (default 0.0.0.0:8080,
matching the deployment platform's convention). For local development with hot
reload use `uvicorn server.app:app --reload`.
"""

from __future__ import annotations

import os

import uvicorn


def main():
    host = os.environ.get("PERSONOS_HOST", "0.0.0.0")
    port = int(os.environ.get("PERSONOS_PORT", "8080"))
    log_level = os.environ.get("PERSONOS_LOG_LEVEL", "info").lower()
    # Import late so env vars like MYSQL_* / PERSONOS_LOG_DIR are in place before
    # settings/rt are initialized.
    from server.app import app
    from personos.config import settings
    from loguru import logger

    logger.info(f"PersonOS starting http://{host}:{port} · "
                f"log_dir={settings.log_dir}")
    uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)


if __name__ == "__main__":
    main()
