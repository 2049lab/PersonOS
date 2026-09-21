"""Media storage: original images, video clips and voice samples.

Two backends with an identical surface — the local filesystem by default, an
object store when configured. Callers hold whichever one they were given and
never branch on it.
"""

from __future__ import annotations

from loguru import logger

from ._common import (
    MediaNotFoundError, MediaStoreError, StoredImage, StoredVideo, content_key,
)
from .local import LocalMediaStore

__all__ = [
    "LocalMediaStore",
    "MediaStoreError",
    "MediaNotFoundError",
    "StoredImage",
    "StoredVideo",
    "content_key",
    "media_store_from_settings",
]


def media_store_from_settings(settings):
    """Build the configured media store.

    Object storage only when explicitly selected *and* usable. Falling back to
    local on incomplete object-store settings would be worse than failing: the
    writes would succeed, quietly landing somewhere nobody is looking.
    """
    if (settings.media_backend or "local").lower() == "oss":
        from .oss import OSSMediaStore

        return OSSMediaStore(
            bucket=settings.oss_bucket,
            endpoint=settings.oss_endpoint,
            access_key_id=settings.oss_access_key_id,
            access_key_secret=settings.oss_access_key_secret,
            region=settings.oss_region,
            prefix=settings.oss_prefix,
            url_expires_seconds=settings.oss_url_expires_seconds,
        )
    root = settings.media_root
    logger.debug(f"media storage: local filesystem at {root}")
    return LocalMediaStore(root=root, prefix="", base_url=settings.media_base_url)
