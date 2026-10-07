"""ElevenLabs generation with a content-hash cache and a local spend ledger. The API key is read from the environment only
(ELEVENLABS_API_KEY) and is never printed, logged or written.

    from gen import tts, sfx, shift_voice
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).parent
CACHE = HERE / "cache"
LEDGER = HERE / "ledger.jsonl"
CAP = 6000
API = "https://api.elevenlabs.io/v1"
SFX_CREDITS_PER_S = 10     # measured: the response's character-cost header shows 10 credits per second of sound effect
CONF = json.loads((HERE / "voices.json").read_text())
CACHE.mkdir(exist_ok=True)


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l).get("credits", 0) for l in LEDGER.read_text().splitlines() if l.strip())


def _log(rec: dict) -> None:
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _headers() -> dict:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    return {"xi-api-key": key}


def _cost_headers(r: requests.Response) -> dict:
    return {k: v for k, v in r.headers.items() if any(w in k.lower() for w in ("cost", "credit", "character", "usage"))}


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


def tts(voice_id: str, text: str, tag: str = "") -> Path:
    settings = CONF["voice_settings"]
    out = CACHE / f"tts_{_key(voice_id, CONF['model'], text, settings)}.mp3"
    if out.exists():
        return out
    credits = len(text)
    if spent() + credits > CAP:
        raise RuntimeError(f"spend cap {CAP} would be exceeded ({spent():.0f} + {credits})")
    r = requests.post(f"{API}/text-to-speech/{voice_id}", params={"output_format": "mp3_44100_128"}, headers=_headers(),
                      json={"text": text, "model_id": CONF["model"], "voice_settings": settings}, timeout=180)
    if r.status_code != 200:
        _log({"kind": "tts_error", "voice": voice_id, "chars": len(text), "status": r.status_code, "credits": 0, "body": r.text[:200]})
        raise RuntimeError(f"TTS failed: HTTP {r.status_code} {r.text[:200]}")
    out.write_bytes(r.content)
    credits = int(_cost_headers(r).get("character-cost", credits))
    _log({"kind": "tts", "tag": tag, "voice": voice_id, "chars": len(text), "credits": credits, "cost_headers": _cost_headers(r), "file": out.name})
    return out


def sfx(prompt: str, seconds: float, tag: str = "", loop: bool = False) -> Path:
    out = CACHE / f"sfx_{_key(prompt, seconds, loop)}.mp3"
    if out.exists():
        return out
    credits = seconds * SFX_CREDITS_PER_S
    if spent() + credits > CAP:
        raise RuntimeError(f"spend cap {CAP} would be exceeded ({spent():.0f} + {credits})")
    body = {"text": prompt, "duration_seconds": seconds, "prompt_influence": 0.5}
    if loop:
        body["loop"] = True
    r = requests.post(f"{API}/sound-generation", params={"output_format": "mp3_44100_128"}, headers=_headers(), json=body, timeout=180)
    if r.status_code != 200:
        _log({"kind": "sfx_error", "seconds": seconds, "status": r.status_code, "credits": 0, "body": r.text[:200]})
        raise RuntimeError(f"SFX failed: HTTP {r.status_code} {r.text[:200]}")
    out.write_bytes(r.content)
    credits = int(_cost_headers(r).get("character-cost", credits))
    _log({"kind": "sfx", "tag": tag, "seconds": seconds, "credits": credits, "cost_headers": _cost_headers(r), "file": out.name})
    return out


def shift_voice(src: Path, semitones: float, formant: float = 1.0, robot: bool = False) -> Path:
    """Pitch (and a little formant) shift with Praat's PSOLA, so the voice stays natural; robot adds a light chorus + comb."""
    out = CACHE / f"vc_{_key(src.name, semitones, formant, robot)}.wav"
    if out.exists():
        return out
    import parselmouth
    from parselmouth.praat import call
    wav = CACHE / f"_in_{src.stem}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "44100", str(wav)], check=True)
    snd = parselmouth.Sound(str(wav))
    if semitones or formant != 1.0:
        f0 = snd.to_pitch(pitch_floor=75, pitch_ceiling=600).selected_array["frequency"]
        med = float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 200.0
        snd = call(snd, "Change gender", 75, 600, formant, med * 2 ** (semitones / 12), 1.0, 1.0)
    tmp = CACHE / f"_vc_{src.stem}.wav"
    snd.save(str(tmp), "WAV")
    af = "highpass=f=90,loudnorm=I=-20:TP=-2:LRA=7"
    if robot:
        af = "chorus=0.7:0.9:28|40:0.25|0.2:0.3|0.25:1.5|2,aecho=0.8:0.85:7:0.22," + af
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-af", af, "-ar", "44100", str(out)], check=True)
    wav.unlink(missing_ok=True); tmp.unlink(missing_ok=True)
    return out
