"""OSS 签名 URL 有效期配置单测:OSS_URL_EXPIRES_SECONDS 透传到 OSSMediaStore。

默认 3600(1 小时);media_store_from_settings 负责透传——签名 URL 是每次请求现签的,
有效期只由该配置决定(构造函数不触 oss2,可离线测)。
"""

from __future__ import annotations

import dataclasses

from personos.storage.media_store import media_store_from_settings


@dataclasses.dataclass
class _Cfg:
    oss_bucket: str = "b"
    oss_endpoint: str = "oss-internal.cn-shanghai.aliyuncs.com"
    oss_access_key_id: str = "ak"
    oss_access_key_secret: str = "sk"
    oss_region: str = "cn-shanghai"
    oss_prefix: str = "personos/"
    oss_url_expires_seconds: int = 3600


def test_default_expiry_one_hour():
    store = media_store_from_settings(_Cfg())
    assert store.url_expires_seconds == 3600


def test_env_expiry_passthrough():
    store = media_store_from_settings(_Cfg(oss_url_expires_seconds=86400))
    assert store.url_expires_seconds == 86400
    assert store.public_endpoint == "oss.cn-shanghai.aliyuncs.com"   # 公网域名推导不受影响
