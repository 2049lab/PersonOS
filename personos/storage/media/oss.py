"""Object-store backend (S3-compatible / Alibaba Cloud OSS).

Optional: `pip install personos[oss]`. Without it the local filesystem backend
is used instead, and nothing else in the pipeline notices.
"""

from __future__ import annotations

import hashlib

from loguru import logger

from ._common import (
    MediaNotFoundError,
    StoredImage,
    StoredVideo,
    content_key,
    validate_audio,
    validate_image,
    validate_video,
)


class OSSMediaStore:
    """Content-addressed media storage on Alibaba Cloud OSS, plus signed URLs."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint: str,
        access_key_id: str,
        access_key_secret: str,
        region: str = "cn-shanghai",
        prefix: str = "personos/",
        url_expires_seconds: int = 3600,
        public_endpoint: str | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("OSS bucket must not be empty")
        if not endpoint:
            raise ValueError("OSS endpoint must not be empty")
        if not access_key_id or not access_key_secret:
            raise ValueError("OSS access key id/secret must not be empty")
        self.bucket = bucket
        self.endpoint = endpoint
        # A client device cannot reach an "-internal" endpoint, so signed GET URLs
        # have to use the public domain. If one is not configured explicitly we
        # derive it from the internal endpoint.
        self.public_endpoint = (public_endpoint or endpoint.replace("-internal", "")).strip()
        self.region = region
        self.prefix = prefix
        self.url_expires_seconds = url_expires_seconds
        self._access_key_id = access_key_id
        self._access_key_secret = access_key_secret
        self._bucket_client = None
        self._sign_bucket_client = None

    def _client(self):
        if self._bucket_client is not None:
            return self._bucket_client
        try:
            import oss2  # type: ignore
        except ImportError as exc:
            raise ImportError("uploading to OSS requires the 'oss2' package") from exc
        endpoint = self.endpoint if self.endpoint.startswith(("http://", "https://")) \
            else "https://" + self.endpoint
        auth = oss2.AuthV4(self._access_key_id, self._access_key_secret)
        self._bucket_client = oss2.Bucket(auth, endpoint, self.bucket, region=self.region)
        return self._bucket_client

    def _sign_client(self):
        """A separate client on the public domain, used only to sign GET URLs a client can reach."""
        if self._sign_bucket_client is not None:
            return self._sign_bucket_client
        try:
            import oss2  # type: ignore
        except ImportError as exc:
            raise ImportError("uploading to OSS requires the 'oss2' package") from exc
        endpoint = self.public_endpoint if self.public_endpoint.startswith(("http://", "https://")) \
            else "https://" + self.public_endpoint
        auth = oss2.AuthV4(self._access_key_id, self._access_key_secret)
        self._sign_bucket_client = oss2.Bucket(auth, endpoint, self.bucket, region=self.region)
        return self._sign_bucket_client

    def _object_key(self, sha256: str, extension: str, owner: str) -> str:
        return content_key(self.prefix, owner, sha256, extension)

    def save_image(self, data: bytes, *, owner: str, content_type: str = "") -> StoredImage:
        """Validate and store an image.

        Storage is content-addressed, so identical bytes are never re-uploaded.
        Returns a StoredImage whose key is the content_ref.
        """
        detected, extension = validate_image(data, content_type)
        sha256 = hashlib.sha256(data).hexdigest()
        key = self._object_key(sha256, extension, owner)
        bucket = self._client()
        if not bucket.object_exists(key):
            bucket.put_object(key, data, headers={"Content-Type": detected})
            logger.info(f"OSS stored image owner={owner} key={key} bytes={len(data)}")
        else:
            logger.info(f"OSS already has these bytes (content-addressed, upload skipped) key={key}")
        return StoredImage(key=key, sha256=sha256, content_type=detected, byte_size=len(data))

    def _clip_key(self, sha256: str, extension: str, owner: str) -> str:
        return content_key(self.prefix, owner, sha256, extension, subdir="clip")

    def save_video(self, data: bytes, *, owner: str, content_type: str = "") -> StoredVideo:
        """Validate and store a clip, content-addressed so identical bytes are never
        re-uploaded.

        Returns a StoredVideo whose key is what sign_url turns into the URL handed
        to the multimodal model.
        """
        detected, extension = validate_video(data, content_type)
        sha256 = hashlib.sha256(data).hexdigest()
        key = self._clip_key(sha256, extension, owner)
        bucket = self._client()
        if not bucket.object_exists(key):
            bucket.put_object(key, data, headers={"Content-Type": detected})
            logger.info(f"OSS stored clip owner={owner} key={key} bytes={len(data)}")
        else:
            logger.info(f"OSS already has this clip (content-addressed, upload skipped) key={key}")
        return StoredVideo(key=key, sha256=sha256, content_type=detected, byte_size=len(data))

    def save_audio(self, data: bytes, *, owner: str) -> str:
        """Store a voiceprint sample as 16k mono wav and return its OSS key, which is
        what lets arbitration recognize the speaker by ear.

        Content-addressed, under the voice/ subprefix.
        """
        validate_audio(data)
        sha256 = hashlib.sha256(data).hexdigest()
        key = content_key(self.prefix, owner, sha256, ".wav", subdir="voice")
        bucket = self._client()
        if not bucket.object_exists(key):
            bucket.put_object(key, data, headers={"Content-Type": "audio/wav"})
        return key

    def object_size(self, key: str) -> int:
        """The object's size in bytes, via a head request rather than a download.

        Used to size-check a clip the caller uploaded.
        """
        return int(self._client().head_object(key).content_length)

    def sign_url(self, key: str) -> str:
        """Sign a time-limited GET URL on the public domain that a client can reach.

        Used to hand an image back when showing where a memory came from.
        """
        return self._sign_client().sign_url("GET", key, self.url_expires_seconds, slash_safe=True)

    def read_bytes(self, key: str) -> bytes:
        """Fetch the original bytes back, for when the image pipeline needs the full image."""
        try:
            return self._client().get_object(key).read()
        except Exception as exc:
            raise MediaNotFoundError(key) from exc

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client().object_exists(key))
        except Exception:
            logger.warning(f"OSS object_exists lookup failed key={key}", exc_info=True)
            return False
