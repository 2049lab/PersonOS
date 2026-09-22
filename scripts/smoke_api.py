"""End-to-end smoke test of the public API against real models and a real
database: all seven endpoints walked one by one, with contract assertions.

Run: PYTHONPATH=. python scripts/smoke_api.py
(requires a working model API key in .env; everything runs inside a user
namespace unique to this run and is cleaned up at the end, so a shared
database is left with no residue)

Covers: healthz/readyz probes -> register (including the 409 and 401 paths)
-> ingest x3 -> task polling -> session/end -> recall (fast/deep/empty)
-> trace (atom/evidence/404) -> tasks and trace isolation across users
-> invalid mode 400 -> upstream-failure 502 fallback -> /health counters.

The number of W1 segments is an LLM judgement (short segments may be merged),
so it is only printed for observation and never asserted. Segmentation quality
is measured by the LoCoMo benchmark instead.
"""
import json
import os
import sys
import time

os.environ["PERSONOS_LOG_DIR"] = "/tmp/personos_smoke/logs"

from fastapi.testclient import TestClient   # noqa: E402

from server.app import app          # noqa: E402
from server import runtime as rt_mod   # noqa: E402

# A user suffix unique to this run, so a re-run never collides with stale data
# (register 409); everything is cleaned up at the end.
SUFFIX = time.strftime("%m%d-%H%M%S")
AGENT, OTHER = f"smoke-agent-{SUFFIX}", f"smoke-other-{SUFFIX}"


def cleanup():
    """Delete this run's user data (every business table plus users), so a
    shared database is left with no residue."""
    from personos.storage.db.ddl import TABLE_NAMES
    db = rt_mod.rt.db
    for uid in (AGENT, OTHER):
        # Same wipe as Memory.reset: every table that carries user data. Redis
        # seg/lock keys are cleared when the session ends, and any leftover key
        # carries a TTL and expires itself.
        for t in TABLE_NAMES:
            if t == "users":
                continue
            db.execute(f"DELETE FROM {t} WHERE user_id=%s", (uid,))
        db.execute("DELETE FROM users WHERE user_id=%s", (uid,))
    print("cleaned up this run's user data (no residue left behind)", flush=True)


# atexit: guarantees cleanup even on a mid-run crash (e.g. an upstream outage),
# and equally on a normal exit.
import atexit   # noqa: E402
atexit.register(cleanup)


PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""), flush=True)


# Every /api/v1 request must be AK/SK signed (no env bypass by design). Give this
# run its own keypair and sign like a real caller would.
from server.signing import sign_request   # noqa: E402

SMOKE_AK, SMOKE_SK = "smoke-ak", "smoke-secret-key"
os.environ["PERSONOS_AKSK_MAP"] = json.dumps({SMOKE_AK: SMOKE_SK})

c = TestClient(app)
c.__enter__()          # Fire the lifespan startup (without it the ingest dispatcher
                       # never starts and queued writes are never consumed).
atexit.register(lambda: c.__exit__(None, None, None))


def req(method, path, **kw):
    """c.request + AK/SK signature headers (content-type and x-user-token are
    signed when present, per the scheme's minimum set). The canonical string
    covers path and query separately, so split them before signing."""
    from urllib.parse import parse_qsl, urlsplit

    parts = urlsplit(path)
    headers = {k.lower(): v for k, v in (kw.pop("headers", None) or {}).items()}
    body = b""
    if "json" in kw:
        body = json.dumps(kw.pop("json")).encode()
        headers["content-type"] = "application/json"
    headers.update(sign_request(SMOKE_AK, SMOKE_SK, method, parts.path,
                                query_items=parse_qsl(parts.query),
                                headers=headers, body=body))
    return c.request(method, parts.path, params=parts.query or None,
                     content=body or None, headers=headers, **kw)


def data(r):
    """Unwrap the {code, data, msg} envelope the API puts around every payload."""
    j = r.json()
    if isinstance(j, dict) and {"code", "data", "msg"} <= set(j):
        return j.get("data") or {}
    return j


def errmsg(r):
    """The human-readable failure reason, wherever the envelope put it."""
    j = r.json()
    return str((j or {}).get("msg") or (j or {}).get("error") or "")

print("== 0. probes (outside /api/v1, unauthenticated) ==", flush=True)
r = req("GET", "/healthz"); check("GET /healthz 200", r.status_code == 200, r.text[:80])
r = req("GET", "/readyz"); check("GET /readyz 200", r.status_code == 200, r.text[:80])

print("== 1. register ==", flush=True)
r = req("POST", "/api/v1/users/register", json={"user_id": AGENT})
check("register 201", r.status_code == 201, r.text[:120])
tok = data(r)["token"]
H = {"X-User-Token": tok}

r = req("POST", "/api/v1/users/register", json={"user_id": AGENT})
check("duplicate registration 409", r.status_code == 409, r.text[:80])

r = req("POST", "/api/v1/recall", json={"session_id": "s", "query": "x"})
check("no token 401", r.status_code == 401, r.text[:80])
r = req("POST", "/api/v1/recall", json={"session_id": "s", "query": "x"}, headers={"X-User-Token": "tok_bogus"})
check("bad token 401", r.status_code == 401, r.text[:80])

print("== 2. ingest (batched atoms: one batch on topic, one batch changing topic) ==", flush=True)
BATCHES = [
    [{"speaker": "user", "text": "I moved to Jurong West last weekend; the rent is about a hundred dollars more than my old place"},
     {"speaker": "user", "text": "the main reason for moving is that the new place is closer to the office, it saves me half an hour of commuting"}],
    [{"speaker": "user", "text": "by the way, I have a dentist appointment next Wednesday morning, please remind me"}],
]
# Batch validation (the 400 paths; these do not consume a task slot)
r = req("POST", "/api/v1/ingest", json={"session_id": "chat-001", "messages": []}, headers=H)
check("empty batch 400", r.status_code == 400, r.text[:80])
r = req("POST", "/api/v1/ingest", json={"session_id": "chat-001",
                                   "messages": [{"speaker": "user", "text": f"filler line {i}"} for i in range(21)]},
           headers=H)
check("more than 20 messages 400", r.status_code == 400, r.text[:80])
r = req("POST", "/api/v1/ingest", json={"session_id": "chat-001",
                                   "messages": [{"speaker": "user", "text": "  "}]}, headers=H)
check("empty text with no image 400", r.status_code == 400, r.text[:80])
r = req("POST", "/api/v1/ingest", json={"session_id": "chat-001", "speaker": "user", "text": "old single-message format"}, headers=H)
check("old single-message format rejected (422, breaking change)", r.status_code == 422, r.text[:80])

receipts = []
for msgs in BATCHES:
    r = req("POST", "/api/v1/ingest", json={"session_id": "chat-001", "messages": msgs}, headers=H)
    check(f"ingest 202 (batch of {len(msgs)})", r.status_code == 202 and data(r).get("accepted") is True, r.text[:100])
    receipts.append(data(r))
check("receipts carry msg_id + strictly increasing seq",
      all(rc.get("msg_id") for rc in receipts) and
      [rc["seq"] for rc in receipts] == sorted(rc["seq"] for rc in receipts),
      str([{k: rc.get(k) for k in ("msg_id", "seq", "queue_depth")} for rc in receipts])[:160])


def wait_drained(session_id, seq, timeout=300):
    """Poll /queue/status until the session's cursor has passed seq and the
    backlog is empty — the HTTP counterpart of the SDK's flush()."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = req("GET", f"/api/v1/queue/status?session_id={session_id}", headers=H)
        st = data(r)
        if st.get("depth") == 0 and st.get("cursor", 0) >= seq:
            return st, None
        time.sleep(2)
    return None, f"timeout (last status: {st if 'st' in dir() else 'n/a'})"


st, err = wait_drained("chat-001", receipts[-1]["seq"])
check("both batches consumed (queue/status cursor passed their seq)", st is not None, err or "")

print("== 3. session/end (close the trailing segment) ==", flush=True)
r = req("POST", "/api/v1/session/end", json={"session_id": "chat-001"}, headers=H)
check("session/end 202", r.status_code == 202, r.text[:80])
st, err = wait_drained("chat-001", data(r)["seq"])
check("session/end consumed (trailing segment closed)", st is not None, err or "")
r = req("GET", "/api/v1/episodes", headers=H)
check("episodes 200", r.status_code == 200, r.text[:100])
eps = data(r).get("items") or []
check("the session produced at least one episode", len(eps) >= 1, str(eps)[:160])
print(f"  [observed] episodes={data(r).get('total')} - short batches may merge into one W1 segment "
      f"(the split is the LLM's call, not a defect)", flush=True)

print("== 3b. sync=true: the API waits for the drain (same contract as the SDK) ==", flush=True)
r = req("POST", "/api/v1/ingest", headers=H, json={
    "session_id": "chat-sync", "sync": True,
    "messages": [{"speaker": "user", "text": "for the sync test: my favorite bookstore is Grassroots"}]})
check("ingest sync=true 202 with consumed=true", r.status_code == 202 and data(r).get("consumed") is True,
      r.text[:120])
r = req("POST", "/api/v1/session/end", headers=H, json={"session_id": "chat-sync", "sync": True})
check("session/end sync=true 202 with consumed=true",
      r.status_code == 202 and data(r).get("consumed") is True, r.text[:120])
# sync returned, so the memory must already be there — no task polling needed.
r = req("POST", "/api/v1/recall", headers=H,
           json={"session_id": "chat-sync", "query": "what is my favorite bookstore?", "mode": "fast"})
check("recall right after sync write answers (read-your-own-writes over HTTP)",
      r.status_code == 200 and "grassroots" in data(r).get("answer", "").lower(),
      data(r).get("answer", "")[:120])

print("== 4. recall fast (fast path) ==", flush=True)
r = req("POST", "/api/v1/recall",
           json={"session_id": "chat-001", "query": "Where do I live now, and why did I move?", "mode": "fast"}, headers=H)
check("recall fast 200", r.status_code == 200, r.text[:200])
o = data(r)
check("all fields present (query/mode/verdict/critique/retried/answer/cited_cells/memories)",
      all(k in o for k in ("query", "mode", "verdict", "critique", "retried", "answer",
                           "cited_cells", "memories")))
check("mode echoed back as fast", o["mode"] == "fast")
check("verdict within the enum", o["verdict"] in ("ok", "answer_defect", "insufficient_material"),
      str(o["verdict"]))
check("answer mentions Jurong West", "jurong" in o["answer"].lower(), o["answer"][:120])
if o["memories"]:
    m = o["memories"][0]
    check("memory carries atom_id/evidence_refs with inlined evidence",
          m.get("atom_id") and m.get("evidence_refs") and isinstance(m.get("evidence"), list),
          str(m)[:160])
    ev0 = (m.get("evidence") or [{}])[0]
    check("evidence entry carries holder/content", ev0.get("holder") and ev0.get("content"), str(ev0)[:120])
    smoke_atom_id, smoke_ev_id = m["atom_id"], m["evidence_refs"][0]
else:
    smoke_atom_id = smoke_ev_id = None
    check("memory carries atom_id/evidence_refs with inlined evidence", False,
          "memories was empty, cannot verify")
print(f"  answer: {o['answer'][:220]}", flush=True)

print("== 5. recall deep (straight to the deep path) ==", flush=True)
r = req("POST", "/api/v1/recall", json={"session_id": "chat-001", "query": "What do I have scheduled next Wednesday?", "mode": "deep"}, headers=H)
check("recall deep 200", r.status_code == 200, r.text[:200])
o_deep = data(r)
check("deep answer mentions the dentist", "dent" in o_deep["answer"].lower(), o_deep["answer"][:160])
print(f"  answer: {o_deep['answer'][:220]}", flush=True)

print("== 6. recall on an unanswerable question (the honesty obligation) ==", flush=True)
r = req("POST", "/api/v1/recall", json={"session_id": "chat-001", "query": "What is my cat's name?"}, headers=H)
check("unanswerable question 200", r.status_code == 200, r.text[:120])
o_e = data(r)
# Which negative verdict the reviewer lands on is an LLM judgement:
# insufficient_material (nothing relevant exists) or answer_defect (materials
# exist but none answer it — with unrelated cells in the store, both are
# honest). What must never vary is the honesty itself: no fabricated name.
check("verdict is a negative one (an honest refusal, not a fabrication)",
      o_e["verdict"] in ("insufficient_material", "answer_defect"), str(o_e["verdict"]))
check("no fabricated content is cited", not o_e.get("cited_cells") or o_e["verdict"] == "answer_defect",
      str(o_e.get("cited_cells"))[:120])
if o_e["verdict"] == "insufficient_material":
    check("nothing is returned in memories", o_e["memories"] == [], f"len={len(o_e['memories'])}")
# Whether the deep path answered or the answer was assembled deterministically,
# it must say "not found" and what was searched. Only the family of negative
# phrasings is matched; no exact wording is hardcoded.
honest = any(k in o_e["answer"].lower() for k in
             ("not found", "no record", "don't have", "do not have", "not mentioned",
              "no information", "cannot determine", "can't determine", "unable to",
              "not recorded", "never mentioned"))
check("answer states the situation plainly (a negative conclusion plus what was searched)",
      honest, o_e["answer"][:200])
print(f"  answer: {o_e['answer'][:260]}", flush=True)

print("== 7. trace, both directions ==", flush=True)
if smoke_atom_id:
    r = req("GET", f"/api/v1/trace/{smoke_atom_id}", headers=H)
    check("trace atom 200 node=memory", r.status_code == 200 and data(r).get("node") == "memory", r.text[:160])
    check("trace atom drills down to non-empty evidence", len(data(r).get("evidence", [])) >= 1)
if smoke_ev_id:
    r = req("GET", f"/api/v1/trace/{smoke_ev_id}", headers=H)
    check("trace evidence 200 node=evidence",
          r.status_code == 200 and data(r).get("node") == "evidence", r.text[:160])
    n = data(r)
    check("trace evidence pair non-empty", isinstance(n.get("pair"), list) and len(n["pair"]) >= 1)
    check("trace evidence cited_by carries citations", isinstance(n.get("cited_by"), list) and len(n["cited_by"]) >= 1)
r = req("GET", "/api/v1/trace/atom_bogus", headers=H)
check("trace on an unknown id 404", r.status_code == 404, r.text[:80])

print("== 8. tasks / trace isolation across users ==", flush=True)
r = req("POST", "/api/v1/users/register", json={"user_id": OTHER})
tok2 = data(r)["token"]
r = req("GET", "/api/v1/tasks/task_bogus", headers={"X-User-Token": tok2})
check("unknown task 404", r.status_code == 404, r.text[:80])
r = req("GET", "/api/v1/queue/status?session_id=chat-001", headers={"X-User-Token": tok2})
check("someone else's queue view is empty (depth 0, cursor 0)",
      r.status_code == 200 and data(r).get("depth") == 0 and data(r).get("cursor") == 0,
      r.text[:120])
if smoke_atom_id:
    r = req("GET", f"/api/v1/trace/{smoke_atom_id}", headers={"X-User-Token": tok2})
    check("someone else's memory trace 404", r.status_code == 404, r.text[:80])

print("== 9. invalid mode 400 ==", flush=True)
r = req("POST", "/api/v1/recall", json={"session_id": "s", "query": "x", "mode": "bogus"}, headers=H)
check("mode=bogus 400", r.status_code == 400 and "mode" in errmsg(r), r.text[:120])

print("== 10. upstream-failure 502 fallback ==", flush=True)
orig_chat, orig_embed = rt_mod.rt.llm.chat, rt_mod.rt.embedder.embed


def boom(*a, **kw):
    raise RuntimeError("simulated upstream outage")


rt_mod.rt.llm.chat = boom
rt_mod.rt.embedder.embed = boom
try:
    r = req("POST", "/api/v1/recall", json={"session_id": "chat-001", "query": "Where do I live?"}, headers=H)
    check("upstream failure -> 502 JSON (not a bare 500)",
          r.status_code == 502 and "unavailable" in errmsg(r), r.text[:160])
finally:
    rt_mod.rt.llm.chat, rt_mod.rt.embedder.embed = orig_chat, orig_embed
r = req("POST", "/api/v1/recall", json={"session_id": "chat-001", "query": "Where do I live?"}, headers=H)
check("recall 200 again after recovery", r.status_code == 200, r.text[:120])

print("== 11. /health counters ==", flush=True)
r = req("GET", "/api/v1/health")
h = data(r)
check("health 200 with non-zero counts",
      r.status_code == 200 and h["users"] >= 2 and h["evidence"] >= 3 and h["cells"] >= 1, str(h))

print(f"\n===== smoke result: {len(PASS)} passed / {len(FAIL)} failed =====", flush=True)
if FAIL:
    for f_ in FAIL:
        print(f"  FAIL {f_}")
sys.exit(1 if FAIL else 0)   # atexit cleans up on the way out
