"""Shared pieces of the media stores: validation, content addressing, types.

Both backends must agree on all of this. If the local store accepted a file the
object store rejects — or produced a different key for the same bytes — then
switching backends would change what gets stored, which is exactly the kind of
silent divergence a "just packaging" change must not introduce.

Format detection is by magic number, never by the declared content type: a
client can be wrong or say nothing, the bytes cannot.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger

# Content type -> extension. Image input supports these common formats.
_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/heic": ".heic",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

_MAX_IMAGE_BYTES = 25 * 1024 * 1024   # per-image ceiling

# Video clips: the caller records, cuts into segments of roughly two minutes, and
# uploads them. The multimodal model fetches each one itself through a signed URL,
# so we never hand it the bytes.
_VIDEO_EXTENSIONS = {"video/mp4": ".mp4", "video/quicktime": ".mov"}
_MAX_VIDEO_BYTES = int(os.environ.get("PERSONOS_VIDEO_MAX_BYTES", str(200 * 1024 * 1024)))


class MediaStoreError(ValueError):
    """The media cannot be stored safely: unsupported type, over the size limit, or
    content that does not match what was declared.
    """


class MediaNotFoundError(LookupError):
    """No bytes could be read for this object key."""


@dataclass(frozen=True)
class StoredImage:
    """The result of storing one image."""
    key: str          # the object key, which is also evidence.content_ref
    sha256: str
    content_type: str
    byte_size: int


@dataclass(frozen=True)
class StoredVideo:
    """The result of storing one clip.

    The bytes live in the object store; the database and the multimodal model only
    ever use the key and a signed URL.
    """
    key: str
    sha256: str
    content_type: str
    byte_size: int


def _detect_video_type(header: bytes) -> str | None:
    """Identify mp4 or mov by magic number.

    ISO BMFF puts 'ftyp' at offset 4, and a 'qt' brand means mov.
    """
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12]
        return "video/quicktime" if brand[:2] == b"qt" else "video/mp4"
    return None


def _detect_image_type(header: bytes) -> str | None:
    """Identify the image type by magic number, so a file declared jpg that is
    actually something else cannot slip through.
    """
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        if header[8:12] in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return "image/heic"
    return None




def content_key(prefix: str, owner: str, sha256: str, extension: str,
                subdir: str = "") -> str:
    """Content-addressed, per-owner: {prefix}/{owner}/[subdir/]{YYYY}/{MM}/{sha[:2]}/{sha}{ext}

    Identical bytes produce an identical key, so re-uploading the same media is
    a no-op on every backend. The date segments keep any single directory from
    growing without bound.
    """
    root = prefix.strip("/")
    head = f"{root}/" if root else ""
    sub = f"{subdir.strip('/')}/" if subdir else ""
    now = datetime.now(timezone.utc)
    return f"{head}{owner}/{sub}{now:%Y/%m}/{sha256[:2]}/{sha256}{extension}"


def validate_image(data: bytes, content_type: str = "") -> tuple[str, str]:
    """Return (detected mime, extension) or raise. Shared by both backends."""
    if not data:
        raise MediaStoreError("image payload is empty")
    if len(data) > _MAX_IMAGE_BYTES:
        raise MediaStoreError(f"image exceeds the {_MAX_IMAGE_BYTES} byte limit")
    detected = _detect_image_type(data[:32])
    if detected is None:
        raise MediaStoreError(f"unrecognised image format (declared: {content_type or 'none'})")
    return detected, _EXTENSIONS[detected]


def validate_video(data: bytes, content_type: str = "") -> tuple[str, str]:
    if not data:
        raise MediaStoreError("video payload is empty")
    if len(data) > _MAX_VIDEO_BYTES:
        raise MediaStoreError(f"video exceeds the {_MAX_VIDEO_BYTES} byte limit")
    detected = _detect_video_type(data[:32])
    if detected is None:
        raise MediaStoreError(f"unrecognised video format (declared: {content_type or 'none'})")
    return detected, _VIDEO_EXTENSIONS[detected]


def validate_audio(data: bytes) -> None:
    if not data or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise MediaStoreError("audio is not WAV (expected a RIFF/WAVE header)")
