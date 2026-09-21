"""原图对象存储(阿里云 OSS)。

移植裁剪自 meme-backend 的 media_store,只保留 personos 图片输入需要的部分:
- content-addressed:对象 key 用内容 sha256,相同图片永不重复上传;
- 每 user 隔离:owner 段进 key 前缀,存储布局层面即隔离(与 Redis/MySQL 的 user 前缀同思路);
- 内外网域名分离:上传/回源走内网 endpoint(VPC 内更快),签发给客户端的 GET URL 用公网域名。

原图是**真相层**、永久留底;DB(evidence.content_ref)只存 OSS 对象 key。
凭证未配时构造即抛错——由调用方决定是否降级(纯文本链路不依赖本模块)。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger

# 内容类型 → 扩展名(与 meme-backend 白名单一致;图片输入先支持这几种常见格式)
_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/heic": ".heic",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

_MAX_IMAGE_BYTES = 25 * 1024 * 1024   # 单图上限,与 meme-backend 对齐

# 视频 clip:调用方录制切段(≈2min)上传;Omni 走签名 URL 直取(不下载字节)。
_VIDEO_EXTENSIONS = {"video/mp4": ".mp4", "video/quicktime": ".mov"}
_MAX_VIDEO_BYTES = int(os.environ.get("PERSONOS_VIDEO_MAX_BYTES", str(200 * 1024 * 1024)))


class MediaStoreError(ValueError):
    """图片无法被安全存储(类型不支持/超限/内容与声明不符)。"""


class MediaNotFoundError(LookupError):
    """对象 key 取不到字节。"""


@dataclass(frozen=True)
class StoredImage:
    """一次存图的产物。"""
    key: str          # OSS 对象 key(即 evidence.content_ref)
    sha256: str
    content_type: str
    byte_size: int


@dataclass(frozen=True)
class StoredVideo:
    """一次存 clip 的产物(字节在 OSS,库/Omni 只用 key + 签名 URL)。"""
    key: str
    sha256: str
    content_type: str
    byte_size: int


def _detect_video_type(header: bytes) -> str | None:
    """按魔数识别 mp4/mov(ISO BMFF:偏移 4 起为 'ftyp';qt 品牌 → mov)。"""
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12]
        return "video/quicktime" if brand[:2] == b"qt" else "video/mp4"
    return None


def _detect_image_type(header: bytes) -> str | None:
    """按魔数识别图片类型(防「声明 jpg 实为别的」)。"""
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
        """内容寻址 + 每 user 隔离:{prefix}/{owner}/{YYYY}/{MM}/{sha[:2]}/{sha}{ext}。"""
        root = self.prefix.strip("/")
        head = f"{root}/" if root else ""
        now = datetime.now(timezone.utc)
        return f"{head}{owner}/{now:%Y/%m}/{sha256[:2]}/{sha256}{extension}"

    def save_image(self, data: bytes, *, owner: str, content_type: str = "") -> StoredImage:
        """校验 + 存图(内容寻址,相同字节不重传),返回 StoredImage(key 即 content_ref)。"""
        if not data:
            raise MediaStoreError("图片内容为空")
        if len(data) > _MAX_IMAGE_BYTES:
            raise MediaStoreError(f"图片超过 {_MAX_IMAGE_BYTES} 字节上限")
        detected = _detect_image_type(data[:32])
        if detected is None:
            raise MediaStoreError(f"无法识别的图片格式(声明 {content_type or '未知'})")
        # 以魔数探测结果为准(客户端声明可能不准/缺失)
        extension = _EXTENSIONS[detected]
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
        """clip 独立子前缀 clip/,与用户上传图不混:{prefix}/{owner}/clip/{YYYY}/{MM}/{sha[:2]}/{sha}{ext}。"""
        root = self.prefix.strip("/")
        head = f"{root}/" if root else ""
        now = datetime.now(timezone.utc)
        return f"{head}{owner}/clip/{now:%Y/%m}/{sha256[:2]}/{sha256}{extension}"

    def save_video(self, data: bytes, *, owner: str, content_type: str = "") -> StoredVideo:
        """校验 + 存 clip(内容寻址,相同字节不重传)。返回 StoredVideo(key 供 sign_url 喂 Omni)。"""
        if not data:
            raise MediaStoreError("视频内容为空")
        if len(data) > _MAX_VIDEO_BYTES:
            raise MediaStoreError(f"视频超过 {_MAX_VIDEO_BYTES} 字节上限")
        detected = _detect_video_type(data[:32])
        if detected is None:
            raise MediaStoreError(f"无法识别的视频格式(声明 {content_type or '未知'})")
        extension = _VIDEO_EXTENSIONS[detected]
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
        if not data or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise MediaStoreError("音频非 WAV(RIFF/WAVE)")
        sha256 = hashlib.sha256(data).hexdigest()
        root = self.prefix.strip("/")
        head = f"{root}/" if root else ""
        now = datetime.now(timezone.utc)
        key = f"{head}{owner}/voice/{now:%Y/%m}/{sha256[:2]}/{sha256}.wav"
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


def media_store_from_settings(settings) -> OSSMediaStore:
    """从 Settings 构造 OSS 存储;凭证缺失时抛错(调用方决定降级)。"""
    return OSSMediaStore(
        bucket=settings.oss_bucket,
        endpoint=settings.oss_endpoint,
        access_key_id=settings.oss_access_key_id,
        access_key_secret=settings.oss_access_key_secret,
        region=settings.oss_region,
        prefix=settings.oss_prefix,
        url_expires_seconds=settings.oss_url_expires_seconds,
    )
