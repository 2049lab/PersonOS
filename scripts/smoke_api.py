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
import os
import sys
import time

os.environ["PERSONOS_LOG_DIR"] = "/tmp/personos_smoke/logs"

from fastapi.testclient import TestClient   # noqa: E402

from server.app import app          # noqa: E402
from personos import runtime as rt_mod   # noqa: E402

# A user suffix unique to this run, so a re-run never collides with stale data
# (register 409); everything is cleaned up at the end.
SUFFIX = time.strftime("%m%d-%H%M%S")
AGENT, OTHER = f"smoke-agent-{SUFFIX}", f"smoke-other-{SUFFIX}"


def cleanup():
    """Delete this run's user data (the four business tables plus users), so a
    shared database is left with no residue."""
    db = rt_mod.rt.db
    for uid in (AGENT, OTHER):
        # tasks: the task registry. Redis seg/lock keys are cleared when the
        # session ends, and any leftover key carries a TTL and expires itself.
        for t in ("evidence", "atoms", "atom_chains", "memcells", "session_context", "tasks"):
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


c = TestClient(app)

print("== 0. probes (outside /api/v1, unauthenticated) ==", flush=True)
r = c.get("/healthz"); check("GET /healthz 200", r.status_code == 200, r.text[:80])
r = c.get("/readyz"); check("GET /readyz 200", r.status_code == 200, r.text[:80])

print("== 1. register ==", flush=True)
r = c.post("/api/v1/users/register", json={"user_id": AGENT})
check("register 201", r.status_code == 201, r.text[:120])
tok = r.json()["token"]
H = {"X-User-Token": tok}

r = c.post("/api/v1/users/register", json={"user_id": AGENT})
check("duplicate registration 409", r.status_code == 409, r.text[:80])

r = c.post("/api/v1/recall", json={"session_id": "s", "query": "x"})
check("no token 401", r.status_code == 401, r.text[:80])
r = c.post("/api/v1/recall", json={"session_id": "s", "query": "x"}, headers={"X-User-Token": "tok_bogus"})
check("bad token 401", r.status_code == 401, r.text[:80])

print("== 2. ingest (batched atoms: one batch on topic, one batch changing topic) ==", flush=True)
BATCHES = [
    [{"speaker": "user", "text": "I moved to Jurong West last weekend; the rent is about a hundred dollars more than my old place"},
     {"speaker": "user", "text": "the main reason for moving is that the new place is closer to the office, it saves me half an hour of commuting"}],
    [{"speaker": "user", "text": "by the way, I have a dentist appointment next Wednesday morning, please remind me"}],
]
# Batch validation (the 400 paths; these do not consume a task slot)
r = c.post("/api/v1/ingest", json={"session_id": "chat-001", "messages": []}, headers=H)
check("empty batch 400", r.status_code == 400, r.text[:80])
r = c.post("/api/v1/ingest", json={"session_id": "chat-001",
                                   "messages": [{"speaker": "user", "text": f"filler line {i}"} for i in range(21)]},
           headers=H)
check("more than 20 messages 400", r.status_code == 400, r.text[:80])
r = c.post("/api/v1/ingest", json={"session_id": "chat-001",
                                   "messages": [{"speaker": "user", "text": "  "}]}, headers=H)
check("empty text with no image 400", r.status_code == 400, r.text[:80])
r = c.post("/api/v1/ingest", json={"session_id": "chat-001", "speaker": "user", "text": "old single-message format"}, headers=H)
check("old single-message format rejected (422, breaking change)", r.status_code == 422, r.text[:80])

task_ids = []
for msgs in BATCHES:
    r = c.post("/api/v1/ingest", json={"session_id": "chat-001", "messages": msgs}, headers=H)
    check(f"ingest 202 (batch of {len(msgs)})", r.status_code == 202 and r.json().get("accepted") is True, r.text[:100])
    task_ids.append(r.json().get("task_id"))


def wait_task(tid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/v1/tasks/{tid}", headers=H)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        t = r.json()
        if t["status"] in ("done", "error"):
            return t, None
        time.sleep(2)
    return None, "timeout"


results = []
for tid in task_ids:
    t, err = wait_task(tid)
    ok = t is not None and t["status"] == "done"
    check(f"task {tid[:8]} done", ok, err or str((t or {}).get("error") or "")[:200])
    results.append(t)
check("every batch's task.result carries evidence_ids with a matching count",
      all((t or {}).get("result", {}).get("evidence_ids")
          and len(t["result"]["evidence_ids"]) == len(BATCHES[i])
          for i, t in enumerate(results) if t))
r3 = (results[1] or {}).get("result", {}) if results[1] and results[1]["status"] == "done" else {}
check("topic-changing batch has a non-empty boundary (there is a preceding segment to compare against)",
      r3.get("boundary") is not None, str(r3.get("boundary"))[:120])

print("== 3. session/end (close the trailing segment) ==", flush=True)
r = c.post("/api/v1/session/end", json={"session_id": "chat-001"}, headers=H)
check("session/end 202", r.status_code == 202, r.text[:80])
t, err = wait_task(r.json()["task_id"])
check("session/end task done", t is not None and t["status"] == "done", err or str((t or {}).get("error"))[:200])
if t and t["status"] == "done":
    res = t["result"]
    check("closed_now carries cell_id/topic/atoms",
          res.get("closed_now") and res["closed_now"].get("cell_id") and res["closed_now"].get("atoms", 0) >= 1,
          str(res.get("closed_now"))[:160])
    # Observation only (never fails the run): whether W1 splits "moving" and
    # "dentist" into two segments is the LLM's call; short segments may merge.
    print(f"  [observed] cells_total={res.get('cells_total')} - three short messages may well "
          f"collapse into a single W1 segment (not a defect)", flush=True)

print("== 4. recall fast (fast path) ==", flush=True)
r = c.post("/api/v1/recall",
           json={"session_id": "chat-001", "query": "Where do I live now, and why did I move?", "mode": "fast"}, headers=H)
check("recall fast 200", r.status_code == 200, r.text[:200])
o = r.json()
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
r = c.post("/api/v1/recall", json={"session_id": "chat-001", "query": "What do I have scheduled next Wednesday?", "mode": "deep"}, headers=H)
check("recall deep 200", r.status_code == 200, r.text[:200])
o_deep = r.json()
check("deep answer mentions the dentist", "dent" in o_deep["answer"].lower(), o_deep["answer"][:160])
print(f"  answer: {o_deep['answer'][:220]}", flush=True)

print("== 6. recall on an unanswerable question (the honesty obligation) ==", flush=True)
r = c.post("/api/v1/recall", json={"session_id": "chat-001", "query": "What is my cat's name?"}, headers=H)
check("unanswerable question 200", r.status_code == 200, r.text[:120])
o_e = r.json()
check("verdict=insufficient_material (an honest refusal, not a fabrication)",
      o_e["verdict"] == "insufficient_material", str(o_e["verdict"]))
if o_e["verdict"] == "insufficient_material":
    check("nothing is returned in memories", o_e["memories"] == [], f"len={len(o_e['memories'])}")
    # Whether the deep path answered or the answer was assembled deterministically,
    # it must say "not found" and what was searched. Only the family of negative
    # phrasings is matched; no exact wording is hardcoded.
    honest = any(k in o_e["answer"].lower() for k in
                 ("not found", "no record", "don't have", "do not have", "not mentioned",
                  "no information", "cannot determine", "can't determine", "unable to"))
    check("answer states the situation plainly (a negative conclusion plus what was searched)",
          honest, o_e["answer"][:200])
print(f"  answer: {o_e['answer'][:260]}", flush=True)

print("== 7. trace, both directions ==", flush=True)
if smoke_atom_id:
    r = c.get(f"/api/v1/trace/{smoke_atom_id}", headers=H)
    check("trace atom 200 node=memory", r.status_code == 200 and r.json().get("node") == "memory", r.text[:160])
    check("trace atom drills down to non-empty evidence", len(r.json().get("evidence", [])) >= 1)
if smoke_ev_id:
    r = c.get(f"/api/v1/trace/{smoke_ev_id}", headers=H)
    check("trace evidence 200 node=evidence",
          r.status_code == 200 and r.json().get("node") == "evidence", r.text[:160])
    n = r.json()
    check("trace evidence pair non-empty", isinstance(n.get("pair"), list) and len(n["pair"]) >= 1)
    check("trace evidence cited_by carries citations", isinstance(n.get("cited_by"), list) and len(n["cited_by"]) >= 1)
r = c.get("/api/v1/trace/atom_bogus", headers=H)
check("trace on an unknown id 404", r.status_code == 404, r.text[:80])

print("== 8. tasks / trace isolation across users ==", flush=True)
r = c.post("/api/v1/users/register", json={"user_id": OTHER})
tok2 = r.json()["token"]
r = c.get(f"/api/v1/tasks/{task_ids[0]}", headers={"X-User-Token": tok2})
check("someone else's task 404", r.status_code == 404, r.text[:80])
if smoke_atom_id:
    r = c.get(f"/api/v1/trace/{smoke_atom_id}", headers={"X-User-Token": tok2})
    check("someone else's memory trace 404", r.status_code == 404, r.text[:80])

print("== 9. invalid mode 400 ==", flush=True)
r = c.post("/api/v1/recall", json={"session_id": "s", "query": "x", "mode": "bogus"}, headers=H)
check("mode=bogus 400", r.status_code == 400 and "mode" in r.json().get("error", ""), r.text[:120])

print("== 10. upstream-failure 502 fallback ==", flush=True)
orig_chat, orig_embed = rt_mod.rt.llm.chat, rt_mod.rt.embedder.embed


def boom(*a, **kw):
    raise RuntimeError("simulated upstream outage")


rt_mod.rt.llm.chat = boom
rt_mod.rt.embedder.embed = boom
try:
    r = c.post("/api/v1/recall", json={"session_id": "chat-001", "query": "Where do I live?"}, headers=H)
    check("upstream failure -> 502 JSON (not a bare 500)",
          r.status_code == 502 and "unavailable" in r.json().get("error", ""), r.text[:160])
finally:
    rt_mod.rt.llm.chat, rt_mod.rt.embedder.embed = orig_chat, orig_embed
r = c.post("/api/v1/recall", json={"session_id": "chat-001", "query": "Where do I live?"}, headers=H)
check("recall 200 again after recovery", r.status_code == 200, r.text[:120])

print("== 11. /health counters ==", flush=True)
r = c.get("/api/v1/health")
h = r.json()
check("health 200 with non-zero counts",
      r.status_code == 200 and h["users"] >= 2 and h["evidence"] >= 3 and h["cells"] >= 1, str(h))

print(f"\n===== smoke result: {len(PASS)} passed / {len(FAIL)} failed =====", flush=True)
if FAIL:
    for f_ in FAIL:
        print(f"  FAIL {f_}")
sys.exit(1 if FAIL else 0)   # atexit cleans up on the way out
