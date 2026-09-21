# Examples

One story, told in three chapters of increasing input richness. The same three
calls every time — `add`, `end_session`, `search` — with progressively harder
things being remembered.

| | Input | What it needs |
|---|---|---|
| [quickstart.py](quickstart.py) | a week of conversation | one API key |
| [images.py](images.py) | conversation + a photo | a vision model |
| [video.py](video.py) | video clips | `personos[identity]`, a video-capable model |

## 1. A week of Alice's life — text

```bash
export PERSONOS_LLM_API_KEY=sk-...
python examples/quickstart.py
```

Alice talks on Monday, Wednesday, Friday. The example prints the story as it
is written, then — after **each day** — the profile the library distilled of
her without being asked. Watch it go from "(nothing yet)" to an occupation, a
diet, a city and a preference for blunt answers.

The week contains a correction — 15 fish, then 13, then 11. Watch the two
questions at the end:

```
Q: How many neon tetras do I have now?
A: 11 ... the original count was 15; two died ... two more had jumped out.

Q: Did the number of fish change over time?
A: originally 15 ... two died, reducing it to 13 ... corrected to 11.
```

Both are right, and they are right *at the same time*. A store that overwrote
on update could answer the first but not the second.

The last question — where should the team eat — is never discussed in any
episode. It is the profile that answers it, injected into recall automatically.

## 2. Images

```bash
export PERSONOS_MLLM_API_KEY=sk-...      # may be the same key
export PERSONOS_MLLM_MODEL=gpt-4o
python examples/images.py
```

An image is described once at write time and then behaves like any other
evidence — searchable by words, citable, traceable. The original bytes are
kept too, on the local filesystem unless object storage is configured.

Run it **without** `PERSONOS_MLLM_API_KEY` to see the other half: the image is
still stored, the text path is untouched, and the result carries

```
warning: image stored but not understood: no multimodal model is configured...
```

That is the difference between degrading and failing, and it is worth seeing
once.

## 3. Video and person identity

```bash
pip install 'personos[identity]'          # ~2 GB of model dependencies
export PERSONOS_VIDEO_BACKEND=real
export PERSONOS_MLLM_MODEL=...            # a model that accepts video

python examples/prepare_video_sample.py   # downloads a sample, cuts 3 clips
python examples/video.py
```

This is the part no other open memory framework does. Text and images get
*described*; a person gets *resolved*. Faces, body shots and voice samples
accumulate across clips into character entities, committed when the session
closes.

Real output from the sample recording:

```
People this recording produced:
  Bob                      evidence=['face', 'body', 'voice']
  (name not yet known)     evidence=['voice']
  Robot                    evidence=['voice']   [the wearer — the camera itself]

Q: Who appears in this recording, and what did each of them do?
A: Bob entered holding a basketball, removed his shoes, and began dribbling
   indoors, which ruined Person #1's homework and favourite notebook...
   Person #1, seated at the dining table, told him to stop...
```

Two details worth noticing. **Bob is named because the dialogue names him** —
no one enrolled him. The second person is never named on camera, so she keeps
a stable handle instead, and questions about her still work. And the
**wearer** — the camera itself — is its own character, because "who did this"
has to be answerable about the recorder too.

### Where the sample comes from

`prepare_video_sample.py` downloads one recording from
[M3-Bench](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)
(ByteDance-Seed) and cuts consecutive clips from it.

It is downloaded rather than shipped. M3-Bench is CC BY-NC-SA-4.0 —
non-commercial, share-alike — and this project is Apache-2.0, which permits
commercial use. Committing those files would make the repository's licensing
incoherent and quietly impose non-commercial terms on everyone who clones it.
Fetching on demand keeps the two separate: you obtain the sample under its own
terms.

> Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with
> Long-Term Memory — arXiv:2508.09736

Any video with people in it works:

```bash
python examples/prepare_video_sample.py --source /path/to/your/own.mp4
```

### Clips have to be reachable by the model

The model service fetches each clip **by URL, from its side**, rather than
receiving the bytes. With local media storage that means it needs a publicly
reachable prefix:

```bash
export PERSONOS_MEDIA_BASE_URL=https://your-host/media
# or skip the question entirely:
export PERSONOS_MEDIA_BACKEND=oss
```

Without one the library says so before starting, instead of failing somewhere
inside the model call.

### It is not fast

Roughly 200 seconds per 60-second clip: most of that is the model watching the
video, not local computation. Budget accordingly.
