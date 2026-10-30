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

The end card preloads `docs/assets/brand/mark.svg` and `lockup-tagline.svg`.
It draws the canonical particle mark with a short fade and settle, then holds
the original geometry still. A missing brand asset rejects setup instead of
silently producing a blank mark. The README uses the light and dark covers
from the same `docs/assets/brand/` directory.

These source changes affect future renders. Existing published videos retain
their historical end cards until they are rendered and published again.
