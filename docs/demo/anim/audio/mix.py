"""Voices, SFX, ambience and the final mix for halloween.mp4.

    python docs/demo/anim/audio/mix.py audition   # step 2 audition mp3
    python docs/demo/anim/audio/mix.py voices      # render every dialogue line (cached), fit it to its window
    python docs/demo/anim/audio/mix.py sfx         # generate every SFX / ambience (cached)
    python docs/demo/anim/audio/mix.py mix         # ducked mix -> out/halloween_mix.wav (-16 LUFS, <= -1 dBTP)
    python docs/demo/anim/audio/mix.py mux         # AAC 192k into out/halloween.mp4 (silent copy kept as halloween_silent.mp4)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from gen import CACHE, CONF, HERE, shift_voice, sfx, spent, tts

OUT = HERE.parent.parent / "out"
SR = 44100
CUES = json.loads((HERE / "cues.json").read_text())
MAX_TEMPO = 1.12


def run(*a):
    subprocess.run([str(x) for x in a], check=True)


def decode(path: Path) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def dur(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip())


# ───────────── audition (step 2) ─────────────
def silence(seconds: float, path: Path) -> Path:
    run("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", seconds, path)
    return path


def beep(path: Path) -> Path:
    run("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=880:d=0.18:r=44100", "-af", "afade=t=out:st=0.12:d=0.06,volume=0.5", "-ac", "1", path)
    return path


def audition() -> None:
    OUT.mkdir(exist_ok=True)
    tmp = CACHE / "_aud"; tmp.mkdir(exist_ok=True)
    gap, short, bp = silence(0.7, tmp / "gap.wav"), silence(0.35, tmp / "short.wav"), beep(tmp / "beep.wav")
    parts, index, n = [], [], 0
    for ch, cfg in CONF["characters"].items():
        for name, vid in cfg["candidates"].items():
            n += 1
            clips = [shift_voice(tts(vid, line, f"audition:{ch}:{name}"), cfg["semitones"], cfg["formant"], cfg.get("robot", False)) for line in CONF["audition_lines"][ch]]
            parts += [bp, gap]
            for i, c in enumerate(clips):
                parts += [c] + ([short] if i < len(clips) - 1 else [])
            parts += [gap]
            index.append(f"{n:2d}. {ch:7s} voice={name:8s} pitch=+{cfg['semitones']}st formant x{cfg['formant']}{'  +robot chorus/comb' if cfg.get('robot') else ''}\n      " + " | ".join(CONF["audition_lines"][ch]))
    parts += [bp, bp, gap]
    index.append("SFX (after two beeps): " + ", ".join(CONF["audition_sfx"]))
    for k, (prompt, secs) in CONF["audition_sfx"].items():
        parts += [sfx(prompt, secs, f"audition:{k}"), gap]
    norm = []
    for i, p in enumerate(parts):
        q = tmp / f"n{i:03d}.wav"
        run("ffmpeg", "-y", "-loglevel", "error", "-i", p, "-ac", "1", "-ar", "44100", q); norm.append(q)
    lst = tmp / "list.txt"; lst.write_text("".join(f"file '{q}'\n" for q in norm))
    run("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c:a", "libmp3lame", "-b:a", "160k", OUT / "voice_audition.mp3")
    (OUT / "voice_audition.txt").write_text("Voice audition order (a beep precedes each candidate; two beeps precede the SFX samples)\n\n" + "\n".join(index) + "\n")
    print("audition ready;", f"credits spent so far: {spent():.0f}")


# ───────────── step 3: dialogue ─────────────
def voices() -> dict:
    """Render and fit every dialogue line. Returns {line_id: {speaker: {file, start, window, natural, tempo, fits}}}."""
    info: dict = {}
    for s in CUES["says"]:
        who_list = [s["speaker"]] + s.get("also", [])
        win = s["end"] - s["start"]
        for who in who_list:
            key = f"{s['id']}:{who}" if s.get("also") else s["id"]
            if who == "biscuit":
                continue
            cast_who, text = CONF["lines"][key]
            name, vid = CONF["cast"][cast_who]; cfg = CONF["characters"][cast_who]
            raw = tts(vid, text, f"line:{key}:{name}")
            vc = shift_voice(raw, cfg["semitones"], cfg["formant"], cfg.get("robot", False))
            d = dur(vc); tempo = max(1.0, d / win); fits = tempo <= MAX_TEMPO + 1e-6; tempo = min(tempo, MAX_TEMPO)
            out = CACHE / f"fit_{vc.stem}_{tempo:.3f}.wav"
            if not out.exists():
                run("ffmpeg", "-y", "-loglevel", "error", "-i", vc, "-af", f"atempo={tempo:.4f}" if tempo > 1.001 else "anull", "-ar", SR, "-ac", 1, out)
            info.setdefault(s["id"], {})[who] = {"file": out.name, "start": s["start"], "window": round(win, 3), "natural": round(d, 3), "tempo": round(tempo, 3), "fits": fits, "fitted_len": round(dur(out), 3), "offscreen": s.get("offscreen", False)}
    (CACHE / "lines.json").write_text(json.dumps(info, indent=1))
    bad = [(k, w, v) for k, d in info.items() for w, v in d.items() if not v["fits"]]
    print("lines rendered:", sum(len(d) for d in info.values()), "| not fitting within atempo", MAX_TEMPO, ":", [(k, w, v["natural"], v["window"]) for k, w, v in bad] or "none")
    print(f"credits spent so far: {spent():.0f}")
    return info


# ───────────── SFX ─────────────
def sfx_all() -> dict[str, Path]:
    out = {k: sfx(p, sec, f"sfx:{k}") for k, (p, sec) in CONF["sfx"].items()}
    for k, (p, sec) in CONF["ambience"].items():
        out["amb_" + k] = sfx(p, sec, f"amb:{k}", loop=True)
    print(f"sfx ready ({len(out)}); credits spent so far: {spent():.0f}")
    return out


def music_check(path: Path) -> dict:
    """Crude musicality test: pitched (harmonic) fraction + onset periodicity. We can't listen, so report the numbers."""
    import parselmouth
    wav = CACHE / "_music.wav"; run("ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", 1, "-ar", SR, wav)
    snd = parselmouth.Sound(str(wav)); hnr = snd.to_harmonicity_cc(time_step=0.05); h = hnr.values[0]; h = h[h > -100]
    x = decode(path); hop = 512; frames = len(x) // hop; env = np.array([np.abs(x[i * hop:(i + 1) * hop]).mean() for i in range(frames)]); on = np.maximum(0, np.diff(env)); on -= on.mean()
    ac = np.correlate(on, on, "full")[len(on) - 1:]; ac /= ac[0] + 1e-9; lag = np.arange(len(ac)) * hop / SR; m = (lag > 0.25) & (lag < 2.0)
    return {"harmonic_fraction": round(float((h > 6).mean()), 2), "mean_hnr_db": round(float(h.mean()), 1), "beat_peak": round(float(ac[m].max()), 2), "beat_period_s": round(float(lag[m][ac[m].argmax()]), 2)}


# ───────────── mix ─────────────
SFX_MAP = {  # event name -> (sfx key, gain dB, offset s)
    "door_burst": ("door_burst", -4, -0.05), "thunder": ("thunder", -3, 0), "thunder2": ("thunder", -9, 0.1), "page_turn_cover": ("page_turn", -8, 0), "page_turn_morning": ("page_turn", -8, 0),
    "ketchup_squirt": ("ketchup", -6, 0.1), "sheets_toss": ("sheets_toss", -8, 0), "ghost_poof": ("poof", -10, 0), "wrapper_drop": ("wrapper_drop", -10, 0), "alert_red": ("red_alert", -10, 0),
    "repair_land": ("chime_ok", -11, 0), "card_fold": ("chip_whoosh", -16, 0), "save_stamp": ("stamp_thump", -4, 0), "antenna_flash": ("ping", -12, 0), "lens_zoom": ("lens_whir", -10, 0),
    "night_view_on": ("night_hum", -12, 0), "sheet_pull_off": ("cloth_pull", -12, 0), "sheet_hang": ("tick", -14, 0), "sheet_drop": ("cloth_pull", -16, 0), "sheet_pull_on": ("cloth_pull", -14, 0), "sheet_pull_on2": ("cloth_pull", -14, 0),
    "sheet_lift": ("cloth_pull", -14, 0), "wrapper_peel": ("wrapper_stick", -10, 0), "chip_fly": ("chip_whoosh", -17, 0), "chip_land": ("chip_land", -17, 0), "name_sure": ("stamp_thump", -20, 0), "name_guess": ("tick", -20, 0),
    "card_appear": ("chip_land", -22, 0), "bell": ("bell_jingle", -13, 0), "snore": ("snore", -18, 0), "end_card": ("chime_ok", -10, 0.3), "candy_grab": ("candy_rustle", -9, 0), "idea_spark": ("ping", -16, 0),
}


def place(buf: np.ndarray, x: np.ndarray, t: float, gain_db: float = 0.0) -> None:
    i = int(round(t * SR)); i = max(i, 0)
    if i >= len(buf):
        return
    n = min(len(x), len(buf) - i); buf[i:i + n] += x[:n] * 10 ** (gain_db / 20)


def sfx_clip(path: Path, peak: float = 0.7) -> np.ndarray:
    x = decode(path); m = np.abs(x).max() + 1e-9
    return x * (peak / m)


def envelope(x: np.ndarray, atk: float = 0.03, rel: float = 0.35) -> np.ndarray:
    win = int(SR * 0.02); e = np.sqrt(np.convolve(x * x, np.ones(win) / win, "same")); e = np.clip(e / (np.percentile(e[e > 1e-4], 90) + 1e-9), 0, 1)
    out = np.zeros_like(e); a, r = np.exp(-1 / (atk * SR)), np.exp(-1 / (rel * SR)); y = 0.0
    for i in range(0, len(e), 8):                                  # coarse-step smoothing keeps this fast
        v = e[i]; y = a * y + (1 - a) * v if v > y else r * y + (1 - r) * v
        out[i:i + 8] = y
    return out


def tile(x: np.ndarray, n: int, xf: float = 1.0) -> np.ndarray:
    k = int(xf * SR); out = np.zeros(n, dtype=np.float32); pos = 0; fade = np.linspace(0, 1, k, dtype=np.float32)
    while pos < n:
        seg = x.copy(); seg[:k] *= fade; seg[-k:] *= fade[::-1]
        m = min(len(seg), n - pos); out[pos:pos + m] += seg[:m]; pos += len(seg) - k
    return out


def mix_all(use_music: bool | None = None) -> dict:
    info = json.loads((CACHE / "lines.json").read_text()); S = sfx_all(); N = int((CUES["duration"] + 1.0) * SR)
    voice = np.zeros(N, np.float32); fx = np.zeros(N, np.float32); amb = np.zeros(N, np.float32); mus = np.zeros(N, np.float32)
    for lid, d in info.items():
        for who, v in d.items():
            x = decode(CACHE / v["file"])
            if v["offscreen"]:
                tmp = CACHE / "_off.wav"; run("ffmpeg", "-y", "-loglevel", "error", "-i", CACHE / v["file"], "-af", "lowpass=f=2600,aecho=0.8:0.6:60|120:0.3|0.2,volume=0.7", "-ar", SR, "-ac", 1, tmp); x = decode(tmp)
            place(voice, x, v["start"] + 0.06, -2 if who != "pebble" else -3)
    # biscuit: SFX only, on the dialogue lines' windows
    for s in CUES["says"]:
        if s["speaker"] == "biscuit":
            key = CONF["biscuit_sfx"][s["id"]]; place(voice, sfx_clip(S[key], .8), s["start"] + .05, -2)
    # SFX from the cue sheet
    clips = {k: sfx_clip(p) for k, p in S.items() if not k.startswith("amb_")}
    for e in CUES["events"]:
        n = e["name"]
        if n == "script_act":
            if "doorbell" in e["text"]: place(fx, clips["doorbell"], e["t"] - .05, -4)
            else: place(fx, clips["pen_scribble"], e["t"], -24)
        elif n in SFX_MAP:
            k, g, off = SFX_MAP[n]; place(fx, clips[k], e["t"] + off, g)
    for s in CUES["says"]:
        if s["speaker"] != "biscuit": place(fx, clips["pen_scribble"], s["start"] + .1, -26)
    # candy wrappers sticking after each grab; ink wipe over the whole rewrite
    for e in [e for e in CUES["events"] if e["name"] == "candy_grab"]: place(fx, clips["wrapper_stick"], e["t"] + .5, -11)
    ws = [e["t"] for e in CUES["events"] if e["name"] == "wipe_start"][0]; we = [e["t"] for e in CUES["events"] if e["name"] == "wipe_end"][0]
    seg = tile(clips["ink_wipe"], int((we - ws + .4) * SR), .4); place(fx, seg, ws, -10)
    # ambience: night rain until the page turn to morning, birds after, both crossfaded
    night_end, flip = 110.4, 111.3
    rain = tile(decode(S["amb_rain"]), N, 1.5); birds = tile(decode(S["amb_birds"]), N, 1.5); t = np.arange(N) / SR
    rain_env = np.clip((t - 3.2) / 2.0, 0, 1) * np.clip((flip - t) / (flip - night_end), 0, 1) * (1 + 0.8 * ((t > 65.4) & (t < 72.8)))
    bird_env = np.clip((t - night_end) / (flip - night_end), 0, 1) * np.clip((156.5 - t) / 2.0, 0, 1)
    amb += rain * rain_env * 10 ** (-27 / 20) + birds * bird_env * 10 ** (-24 / 20)
    if use_music:
        mus += tile(decode(S["amb_music"]), N, 1.5) * np.clip(t / 2, 0, 1) * np.clip((CUES["duration"] - t) / 3, 0, 1) * 10 ** (-27 / 20)
    # ducking: dialogue envelope pushes effects / ambience / music down
    env = envelope(voice); duck = lambda depth: 1 - depth * env
    mixed = voice + fx * duck(.35) + amb * duck(.55) + mus * duck(.6)
    tmp = CACHE / "_premix.wav"
    import soundfile as sf
    sf.write(str(tmp), mixed[:int(CUES["duration"] * SR)], SR, subtype="FLOAT")
    # two-pass loudnorm to -16 LUFS, true peak <= -1 dBTP
    ln = "loudnorm=I=-16:TP=-1.2:LRA=11"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(tmp), "-af", ln + ":print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
    m = json.loads(r[r.rindex("{"):r.rindex("}") + 1])
    af = f"{ln}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
    final = OUT / "halloween_mix.wav"
    run("ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-af", af + ",alimiter=limit=0.89:level=false", "-ar", SR, "-ac", 2, final)
    r2 = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(final), "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    summ = r2[r2.rindex("Summary:"):]
    I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1)); tp = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summ).group(1)); lra = float(re.search(r"LRA:\s+([\d.]+) LU", summ).group(1))
    res = {"integrated_lufs": I, "true_peak_dbtp": tp, "lra": lra, "premix_input_lufs": m["input_i"], "music": bool(use_music)}
    print(res); (CACHE / "mix_report.json").write_text(json.dumps(res, indent=1)); return res


def mux(video_silent: Path, video_out: Path) -> None:
    run("ffmpeg", "-y", "-loglevel", "error", "-i", video_silent, "-i", OUT / "halloween_mix.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", video_out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "audition": audition()
    elif cmd == "voices": voices()
    elif cmd == "sfx": sfx_all()
    elif cmd == "mix": mix_all(use_music="--music" in sys.argv)
    elif cmd == "mux": mux(OUT / "halloween_silent.mp4", OUT / "halloween.mp4")
    elif cmd == "music": print(music_check(sfx_all()["amb_music"]))
