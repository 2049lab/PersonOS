"""Record and replay for the LLM and embedder: freeze one real run into a
cassette that can be replayed entirely offline.

Why it is needed: open-sourcing means a large-scale migration (splitting the
package, swapping providers, changing the storage backend), and the recall path
is LLM-driven — "all the tests pass" only proves nothing crashed, it does
**not** prove the pipeline is unchanged. The only way to prove that is to take
one real pre-migration run as the reference, replay the same model responses
after the migration, and compare the structured output.

**Indexed by prompt hash, not by call order.** That is the key design decision:
- Order-based indexing (the way the existing FakeLLM works) falls out of
  alignment the moment anyone reorders an unrelated call, which is noisy;
- Under hash indexing, **a miss is itself the most valuable signal** — it says
  precisely that "the text of some prompt changed", and prompt text is part of
  the algorithm, so changing it *is* a behaviour change. That is exactly what
  we want to catch.

When the same prompt is called several times (a retry, say), the recorded
responses are returned in order, and the last one repeats once they run out.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class CassetteMiss(BaseException):
    """A cassette miss. **Deliberately inherits BaseException, not Exception.**

    The pipeline being recorded is full of its own `except Exception` degradation
    paths (a failed R0 rewrite falls back to the original query, a crashed deep
    path falls back to the fast path, and so on). Those fallbacks are correct in
    production, but during replay they would swallow "the cassette did not
    match" and make the failure surface downstream wearing a completely
    unrelated face. This happened once: a miss on R0's chat call was swallowed
    by rewrite_query, which fell back to the original query, so the error was
    reported as "embed miss" and sent everyone chasing the wrong thing for an
    entire round. Inheriting BaseException punches through every
    `except Exception`, so the failure lands where it actually occurred.
    """


def _key(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class Cassette:
    """One cassette: prompt hash -> ordered list of responses."""

    def __init__(self, data: dict | None = None):
        self.chat: dict[str, list[str]] = dict(data.get("chat", {})) if data else {}
        self.embed: dict[str, list[list[float]]] = dict(data.get("embed", {})) if data else {}
        # Keys that missed during replay, used to report "which prompt changed".
        self.misses: list[str] = []

    @classmethod
    def load(cls, path: str | Path) -> "Cassette":
        return cls(read_json(path))

    def save(self, path: str | Path) -> None:
        write_json(path, {"chat": self.chat, "embed": self.embed})

    def stats(self) -> dict:
        return {"chat_prompts": len(self.chat),
                "chat_calls": sum(len(v) for v in self.chat.values()),
                "embed_batches": len(self.embed)}


def read_json(path: str | Path):
    """Read a baseline artefact, transparently decompressing .gz. The bulk of
    these artefacts is 4096-dimensional vectors, which compress to roughly a
    fifth; uncompressed, the snapshot alone is 22MB, too large to commit."""
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    if p.suffix == ".gz":
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        p.write_text(text, encoding="utf-8")


class RecordingLLM:
    """Wraps the real LLM, forwarding verbatim and recording every
    (messages, temperature, max_tokens) -> response."""

    def __init__(self, inner, cassette: Cassette):
        self._inner, self._c = inner, cassette

    def chat(self, messages, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        out = self._inner.chat(messages, temperature=temperature, max_tokens=max_tokens)
        self._c.chat.setdefault(_key(messages, temperature, max_tokens), []).append(out)
        return out


class RecordingEmbedder:
    def __init__(self, inner, cassette: Cassette):
        self._inner, self._c = inner, cassette

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._inner.embed(texts)
        self._c.embed[_key(texts)] = np.asarray(vecs, dtype=np.float32).tolist()
        return vecs


class ReplayLLM:
    """Reads the cassette only, never the network. A miss is recorded and
    raised, because "the prompt changed" has to be an explicit failure rather
    than a silent degradation."""

    def __init__(self, cassette: Cassette, *, strict: bool = True):
        self._c, self._strict = cassette, strict
        self._cursor: dict[str, int] = {}

    def chat(self, messages, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        k = _key(messages, temperature, max_tokens)
        seq = self._c.chat.get(k)
        if not seq:
            self._c.misses.append(k)
            if self._strict:
                raise CassetteMiss(
                    f"chat miss (the prompt has changed) key={k}\n"
                    f"  temperature={temperature} max_tokens={max_tokens}\n"
                    f"  first 300 chars of the last message: {str(messages[-1].get('content'))[:300]}")
            return ""
        i = self._cursor.get(k, 0)
        self._cursor[k] = i + 1
        return seq[min(i, len(seq) - 1)]


class ReplayEmbedder:
    def __init__(self, cassette: Cassette, *, dim: int = 4096, strict: bool = True):
        self._c, self._dim, self._strict = cassette, dim, strict

    def embed(self, texts: list[str]) -> np.ndarray:
        k = _key(texts)
        v = self._c.embed.get(k)
        if v is None:
            self._c.misses.append(k)
            if self._strict:
                raise CassetteMiss(
                    f"embed miss key={k}, {len(texts)} texts, texts[0]={texts[0][:120]!r}")
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        return np.asarray(v, dtype=np.float32)
