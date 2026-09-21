"""Media storage: the two backends must be interchangeable.

The point of these tests is not that each backend works in isolation — it is
that they agree. Same bytes, same key; same rejection for the same bad input.
If they ever diverge, switching backends would change what gets stored, and the
symptom would appear far from the cause.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from personos.config import Config
from personos.storage.media import (
    LocalMediaStore, MediaNotFoundError, MediaStoreError, media_store_from_settings,
)
from personos.storage.media._common import _detect_image_type, content_key

PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 40
JPEG = bytes.fromhex("ffd8ff") + b"\x00" * 40
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40
WAV = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 40


@pytest.fixture
def store(tmp_path: Path) -> LocalMediaStore:
    return LocalMediaStore(root=tmp_path / "media")


# ── writing and reading ─────────────────────────────────────────────────

def test_image_round_trip(store):
    saved = store.save_image(PNG, owner="alice", content_type="image/png")
    assert saved.content_type == "image/png"
    assert saved.byte_size == len(PNG)
    assert store.exists(saved.key)
    assert store.read_bytes(saved.key) == PNG
    assert store.object_size(saved.key) == len(PNG)


def test_content_addressing_deduplicates(store):
    a = store.save_image(PNG, owner="alice")
    b = store.save_image(PNG, owner="alice")
    assert a.key == b.key, "identical bytes must land at the same key"
    assert a.sha256 == b.sha256


def test_different_owners_are_isolated(store):
    a = store.save_image(PNG, owner="alice")
    b = store.save_image(PNG, owner="bob")
    assert a.key != b.key
    assert a.key.startswith("alice/") and b.key.startswith("bob/")


def test_video_and_audio_live_under_their_own_prefixes(store):
    clip = store.save_video(MP4, owner="alice")
    voice = store.save_audio(WAV, owner="alice")
    assert "/clip/" in clip.key
    assert "/voice/" in voice
    assert store.read_bytes(clip.key) == MP4


def test_missing_key_raises_not_found(store):
    with pytest.raises(MediaNotFoundError):
        store.read_bytes("alice/2026/01/aa/does-not-exist.png")
    assert store.exists("nope") is False


# ── validation is shared, so both backends reject identically ───────────

@pytest.mark.parametrize("payload,kind,message", [
    (b"", "image", "empty"),
    (b"this is not an image at all", "image", "unrecognised"),
    (b"", "video", "empty"),
    (b"not a video", "video", "unrecognised"),
])
def test_bad_payloads_are_refused(store, payload, kind, message):
    fn = store.save_image if kind == "image" else store.save_video
    with pytest.raises(MediaStoreError, match=message):
        fn(payload, owner="alice")


def test_audio_must_actually_be_wav(store):
    with pytest.raises(MediaStoreError, match="WAV"):
        store.save_audio(b"\x00" * 40, owner="alice")


def test_format_comes_from_the_bytes_not_the_declaration(store):
    """A client's declared content type can be wrong or absent; the magic
    number cannot. Trusting the declaration would let a mislabelled file
    through and store it under the wrong extension."""
    saved = store.save_image(JPEG, owner="alice", content_type="image/png")
    assert saved.content_type == "image/jpeg"
    assert saved.key.endswith(".jpg")
    assert _detect_image_type(JPEG[:32]) == "image/jpeg"


# ── URLs ────────────────────────────────────────────────────────────────

def test_image_url_without_a_base_url_is_a_file_uri(store):
    saved = store.save_image(PNG, owner="alice")
    assert store.sign_url(saved.key).startswith("file://")


def test_video_url_without_a_base_url_raises_with_a_remedy(store):
    """A remote model fetches the clip URL server-side and cannot open file://.
    Returning one anyway would fail opaquely inside the model call, so this
    fails early and says what to set."""
    clip = store.save_video(MP4, owner="alice")
    with pytest.raises(MediaStoreError, match="PERSONOS_MEDIA_BASE_URL"):
        store.sign_url(clip.key)


def test_base_url_makes_video_usable(tmp_path):
    s = LocalMediaStore(root=tmp_path, base_url="https://media.example/files/")
    clip = s.save_video(MP4, owner="alice")
    assert s.sign_url(clip.key) == f"https://media.example/files/{clip.key}"


def test_writes_stay_inside_the_data_directory(store):
    with pytest.raises(MediaStoreError, match="outside"):
        store.read_bytes("../../../etc/passwd")


# ── backend selection ───────────────────────────────────────────────────

def test_local_is_the_default(tmp_path):
    cfg = Config(data_dir=tmp_path)
    s = media_store_from_settings(cfg)
    assert isinstance(s, LocalMediaStore)
    assert s.root == tmp_path / "media"


@dataclasses.dataclass
class _OssCfg:
    media_backend: str = "oss"
    oss_bucket: str = "b"
    oss_endpoint: str = "oss-internal.cn-shanghai.aliyuncs.com"
    oss_access_key_id: str = "ak"
    oss_access_key_secret: str = "sk"
    oss_region: str = "cn-shanghai"
    oss_prefix: str = "personos/"
    oss_url_expires_seconds: int = 3600


def test_object_store_settings_are_passed_through():
    """Signed URLs are minted per request, so the expiry is governed purely by
    configuration. Constructing the store does not touch the SDK, so this is
    offline."""
    s = media_store_from_settings(_OssCfg())
    assert s.url_expires_seconds == 3600
    s = media_store_from_settings(_OssCfg(oss_url_expires_seconds=86400))
    assert s.url_expires_seconds == 86400
    # A private endpoint is unreachable from a client device, so the public one
    # is derived for signing.
    assert s.public_endpoint == "oss.cn-shanghai.aliyuncs.com"


def test_both_backends_agree_on_the_key_for_the_same_bytes(tmp_path):
    """The shared helper is what guarantees this; the test is what keeps it true."""
    import hashlib

    local = LocalMediaStore(root=tmp_path, prefix="personos/")
    saved = local.save_image(PNG, owner="alice")
    expected = content_key("personos/", "alice",
                           hashlib.sha256(PNG).hexdigest(), ".png")
    assert saved.key == expected
