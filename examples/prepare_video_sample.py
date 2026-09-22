"""Fetch a sample video and cut it into clips, for the video example.

    python examples/prepare_video_sample.py            # default: 3 clips of 60s
    python examples/prepare_video_sample.py --clips 5 --seconds 45
    python examples/prepare_video_sample.py --source /path/to/your/own.mp4

**The sample is downloaded, never redistributed.** It comes from M3-Bench
(ByteDance-Seed), which is licensed CC BY-NC-SA-4.0: non-commercial, share-alike.
This project is Apache-2.0, which permits commercial use, so vendoring those
files into the repository would make its licensing incoherent and would quietly
impose non-commercial terms on everyone who clones it. Downloading on demand
keeps the two licences separate: you obtain the sample under its own terms.

    M3-Bench — Seeing, Listening, Remembering, and Reasoning: A Multimodal
    Agent with Long-Term Memory (arXiv:2508.09736)
    https://huggingface.co/datasets/ByteDance-Seed/M3-Bench

Any video with people in it works just as well; pass --source to use your own
and skip the download entirely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# A living-room recording: several people, some speaking, entering and leaving.
# Chosen because person identity is the thing being demonstrated, and because
# at ~400 MB it is among the smaller files in the set.
SOURCE_URL = ("https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/"
              "resolve/main/videos/robot/living_room_22.mp4")
SOURCE_NAME = "living_room_22.mp4"
SAMPLE_DIR = Path.home() / ".personos" / "samples"


def _download(dest: Path) -> Path:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already downloaded: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    print("About to download a sample video from M3-Bench (ByteDance-Seed),")
    print("licensed CC BY-NC-SA-4.0 — non-commercial use, attribution required.")
    print(f"  {SOURCE_URL}")
    print("  roughly 400 MB, saved to", dest)
    if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        raise SystemExit("cancelled — use --source to point at your own video instead")

    # Streamed to a temporary name and renamed at the end: an interrupted
    # download must not leave a truncated file that later looks complete.
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", SOURCE_URL, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {pct:3d}%  {done / 1e6:6.0f} / {total / 1e6:.0f} MB",
                          end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def _cut(source: Path, out_dir: Path, *, n_clips: int, seconds: float) -> list[Path]:
    """Cut consecutive clips by re-muxing, so nothing is re-encoded.

    Consecutive matters for this example: identity is built up across clips, and
    the interesting behaviour is someone recognised in a later clip being tied
    to the same person seen earlier.
    """
    try:
        import av
    except ImportError as e:
        raise SystemExit("cutting clips needs PyAV: pip install 'personos[video]'") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for index in range(n_clips):
        start = index * seconds
        target = out_dir / f"clip{index:03d}.mp4"
        if target.exists() and target.stat().st_size > 0:
            print(f"  clip {index} already cut: {target.name}")
            written.append(target)
            continue

        with av.open(str(source)) as src, av.open(str(target), "w") as dst:
            in_v = src.streams.video[0]
            in_a = src.streams.audio[0] if src.streams.audio else None
            out_v = dst.add_stream_from_template(in_v)
            out_a = dst.add_stream_from_template(in_a) if in_a else None

            src.seek(int(start * av.time_base), backward=True, any_frame=False)
            streams = [s for s in (in_v, in_a) if s is not None]
            first_pts: dict[int, int] = {}
            for packet in src.demux(streams):
                if packet.dts is None:
                    continue
                t = float(packet.pts * packet.time_base) if packet.pts is not None else 0.0
                if t < start:
                    continue
                if t >= start + seconds:
                    break
                # Rebase timestamps so each clip starts at zero; players and
                # decoders otherwise see a file that begins minutes in.
                idx = packet.stream.index
                if idx not in first_pts:
                    first_pts[idx] = packet.pts or 0
                packet.pts = (packet.pts or 0) - first_pts[idx]
                packet.dts = (packet.dts or 0) - first_pts[idx]
                packet.stream = out_v if packet.stream.type == "video" else out_a
                dst.mux(packet)

        size = target.stat().st_size / 1e6
        print(f"  clip {index}: {target.name}  {size:.0f} MB  ({start:.0f}s → {start + seconds:.0f}s)")
        written.append(target)

    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=3, help="how many consecutive clips")
    ap.add_argument("--seconds", type=float, default=60.0, help="length of each clip")
    ap.add_argument("--source", help="use your own video instead of downloading the sample")
    ap.add_argument("--out", default=str(SAMPLE_DIR / "clips"))
    args = ap.parse_args()

    source = Path(args.source) if args.source else _download(SAMPLE_DIR / SOURCE_NAME)
    if not source.exists():
        raise SystemExit(f"no such file: {source}")

    print(f"\nCutting {args.clips} clips of {args.seconds:.0f}s from {source.name}")
    clips = _cut(source, Path(args.out), n_clips=args.clips, seconds=args.seconds)

    print(f"\nReady: {len(clips)} clips in {args.out}")
    print(f"Next:  python examples/video.py --clips-dir {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
