"""Object-store backend (S3-compatible / Alibaba Cloud OSS).

Optional: `pip install personos[oss]`. Without it the local filesystem backend
is used instead, and nothing else in the pipeline notices.
"""

from __future__ import annotations

import hashlib

from loguru import logger

from ._common import (
    MediaNotFoundError, MediaStoreError, StoredImage, StoredVideo,
    content_key, validate_audio, validate_image, validate_video,
)


class OSSMediaStore:
    """内容寻址的阿里云 OSS 图片存储 + 签名 URL 服务。"""

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
        # 客户端(手机)连不到 -internal,签 GET URL 必须用公网域名;未显式配则从内网域名推导。
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
            raise ImportError("OSS 上传需要 'oss2' 包") from exc
        endpoint = self.endpoint if self.endpoint.startswith(("http://", "https://")) \
            else "https://" + self.endpoint
        auth = oss2.AuthV4(self._access_key_id, self._access_key_secret)
        self._bucket_client = oss2.Bucket(auth, endpoint, self.bucket, region=self.region)
        return self._bucket_client

    def _sign_client(self):
        """独立的公网域名客户端,仅用于签发客户端可达的 GET URL。"""
        if self._sign_bucket_client is not None:
            return self._sign_bucket_client
        import oss2  # type: ignore
        endpoint = self.public_endpoint if self.public_endpoint.startswith(("http://", "https://")) \
            else "https://" + self.public_endpoint
        auth = oss2.AuthV4(self._access_key_id, self._access_key_secret)
        self._sign_bucket_client = oss2.Bucket(auth, endpoint, self.bucket, region=self.region)
        return self._sign_bucket_client

    def _object_key(self, sha256: str, extension: str, owner: str) -> str:
        return content_key(self.prefix, owner, sha256, extension)

    def save_image(self, data: bytes, *, owner: str, content_type: str = "") -> StoredImage:
        """校验 + 存图(内容寻址,相同字节不重传),返回 StoredImage(key 即 content_ref)。"""
        detected, extension = validate_image(data, content_type)
        sha256 = hashlib.sha256(data).hexdigest()
        key = self._object_key(sha256, extension, owner)
        bucket = self._client()
        if not bucket.object_exists(key):
            bucket.put_object(key, data, headers={"Content-Type": detected})
            logger.info(f"OSS 存图 owner={owner} key={key} bytes={len(data)}")
        else:
            logger.info(f"OSS 命中已存在(内容寻址,跳过上传) key={key}")
        return StoredImage(key=key, sha256=sha256, content_type=detected, byte_size=len(data))

    def _clip_key(self, sha256: str, extension: str, owner: str) -> str:
        return content_key(self.prefix, owner, sha256, extension, subdir="clip")

    def save_video(self, data: bytes, *, owner: str, content_type: str = "") -> StoredVideo:
        """校验 + 存 clip(内容寻址,相同字节不重传)。返回 StoredVideo(key 供 sign_url 喂 Omni)。"""
        detected, extension = validate_video(data, content_type)
        sha256 = hashlib.sha256(data).hexdigest()
        key = self._clip_key(sha256, extension, owner)
        bucket = self._client()
        if not bucket.object_exists(key):
            bucket.put_object(key, data, headers={"Content-Type": detected})
            logger.info(f"OSS 存 clip owner={owner} key={key} bytes={len(data)}")
        else:
            logger.info(f"OSS clip 命中已存在(内容寻址,跳过上传) key={key}")
        return StoredVideo(key=key, sha256=sha256, content_type=detected, byte_size=len(data))

    def save_audio(self, data: bytes, *, owner: str) -> str:
        """存声纹样本 wav(16k 单声道),返回 OSS key(供仲裁听声辨人)。内容寻址,voice/ 子前缀。"""
        validate_audio(data)
        sha256 = hashlib.sha256(data).hexdigest()
        key = content_key(self.prefix, owner, sha256, ".wav", subdir="voice")
        bucket = self._client()
        if not bucket.object_exists(key):
            bucket.put_object(key, data, headers={"Content-Type": "audio/wav"})
        return key

    def object_size(self, key: str) -> int:
        """OSS 对象字节数(head,不下载)。供调用方上传的 clip 做大小校验(定案 §5.3)。"""
        return int(self._client().head_object(key).content_length)

    def sign_url(self, key: str) -> str:
        """签发客户端可达的 GET URL(公网域名,带时效)。用于溯源返图。"""
        return self._sign_client().sign_url("GET", key, self.url_expires_seconds, slash_safe=True)

    def read_bytes(self, key: str) -> bytes:
        """取回原图字节。供看图管线(MLLM)在需要时拉原图。"""
        try:
            return self._client().get_object(key).read()
        except Exception as exc:
            raise MediaNotFoundError(key) from exc

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client().object_exists(key))
        except Exception:
            logger.warning(f"OSS object_exists 查询失败 key={key}", exc_info=True)
            return False
