# The film

`anim/` is the source of the README short — an original, code-drawn animation
rendered frame by frame with p5.js + p5.brush (`index.html`, `main.js`,
`story*.js`, `rigs.js`, `set.js`, `ui2d.js`); `STORYBOARD.md` is the script.
`audio/` holds the cue sheet, the ElevenLabs voice/SFX generator and the mixer.
Generated media (`out/`, `audio/cache/`) is git-ignored.

```bash
pip install playwright && python -m playwright install chromium   # + ffmpeg
python docs/demo/anim/render.py            # -> out/halloween_silent.mp4
ELEVENLABS_API_KEY=... python docs/demo/anim/audio/gen.py          # voices + SFX (cached)
python docs/demo/anim/audio/mix.py          # -> out/halloween.mp4
```

The end card loads `docs/assets/brand/lockup-tagline.svg` — the same file the
README uses, so the two cannot drift apart.
