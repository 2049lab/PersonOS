"""对外 API 全链路冒烟(真实 MAAS,共享 SIT 库):7 个接口逐一走查 + 契约断言。

运行:PYTHONPATH=. ~/miniconda3/envs/personos/bin/python scripts/smoke_api.py
(需 .env 里有可用 MAAS key;跑在运行唯一 user 命名空间,结束清理,共享库零残留)

覆盖:healthz/readyz 探针 → register(含 409/401 路径)→ ingest×3 → tasks 轮询
→ session/end → recall(fast/deep/empty 题)→ trace(atom/evidence/404)
→ tasks/trace 跨 user 隔离 → 非法 mode 400 → 上游故障 502 兜底 → /health 统计。

W1 分段数是 LLM 判断(短段可能合并),只作观测打印、不做硬断言——分段质量看 LoCoMo bench 统计。
"""
import os
import sys
import time

os.environ["PERSONOS_LOG_DIR"] = "/tmp/personos_smoke/logs"

from fastapi.testclient import TestClient   # noqa: E402

from personos.app.server import app          # noqa: E402
from personos.app import runtime as rt_mod   # noqa: E402

# 本次运行唯一 user 后缀:重跑不撞旧数据(register 409),结束统一清理
SUFFIX = time.strftime("%m%d-%H%M%S")
AGENT, OTHER = f"smoke-agent-{SUFFIX}", f"smoke-other-{SUFFIX}"


def cleanup():
    """清理本次冒烟的 user 数据(业务四表 + users),共享库零残留。"""
    db = rt_mod.rt.db
    for uid in (AGENT, OTHER):
        # tasks:任务登记簿(MySQL);Redis seg/lock 键会话结束即清、残留键带 TTL 自灭
        for t in ("evidence", "atoms", "atom_chains", "memcells", "session_context", "tasks"):
            db.execute(f"DELETE FROM {t} WHERE user_id=%s", (uid,))
        db.execute("DELETE FROM users WHERE user_id=%s", (uid,))
    print("已清理本次冒烟 user 数据(共享库零残留)", flush=True)


# atexit:中途崩溃(如上游断供)也保证清理;正常结束同样生效
import atexit   # noqa: E402
atexit.register(cleanup)


PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✓' if cond else '✗ FAIL'} {name}" + (f"  [{detail}]" if detail else ""), flush=True)


c = TestClient(app)

print("== 0. 探针(不在 /api/v1 下,免鉴权)==", flush=True)
r = c.get("/healthz"); check("GET /healthz 200", r.status_code == 200, r.text[:80])
r = c.get("/readyz"); check("GET /readyz 200", r.status_code == 200, r.text[:80])

print("== 1. register ==", flush=True)
r = c.post("/api/v1/users/register", json={"user_id": AGENT})
check("register 201", r.status_code == 201, r.text[:120])
tok = r.json()["token"]
H = {"X-User-Token": tok}

r = c.post("/api/v1/users/register", json={"user_id": AGENT})
check("重复注册 409", r.status_code == 409, r.text[:80])

r = c.post("/api/v1/recall", json={"session_id": "s", "query": "x"})
check("无 token 401", r.status_code == 401, r.text[:80])
r = c.post("/api/v1/recall", json={"session_id": "s", "query": "x"}, headers={"X-User-Token": "tok_bogus"})
check("错 token 401", r.status_code == 401, r.text[:80])

print("== 2. ingest(批原子:同话题一批 + 换话题一批)==", flush=True)
BATCHES = [
    [{"speaker": "user", "text": "我上周末搬到了裕廊西,房租比原来贵了一百多新币"},
     {"speaker": "user", "text": "搬家的主要原因是新地方离公司近,通勤能省半小时"}],
    [{"speaker": "user", "text": "对了,我下周三上午要去看牙医,记得提醒我"}],
]
# 批校验(400 路径,不占任务名额)
r = c.post("/api/v1/ingest", json={"session_id": "chat-001", "messages": []}, headers=H)
check("空批 400", r.status_code == 400, r.text[:80])
r = c.post("/api/v1/ingest", json={"session_id": "chat-001",
                                   "messages": [{"speaker": "user", "text": f"填充第{i}句"} for i in range(21)]},
           headers=H)
check("超 20 条 400", r.status_code == 400, r.text[:80])
r = c.post("/api/v1/ingest", json={"session_id": "chat-001",
                                   "messages": [{"speaker": "user", "text": "  "}]}, headers=H)
check("空文本无图 400", r.status_code == 400, r.text[:80])
r = c.post("/api/v1/ingest", json={"session_id": "chat-001", "speaker": "user", "text": "旧单条格式"}, headers=H)
check("旧单条格式拒绝(422,break change)", r.status_code == 422, r.text[:80])

task_ids = []
for msgs in BATCHES:
    r = c.post("/api/v1/ingest", json={"session_id": "chat-001", "messages": msgs}, headers=H)
    check(f"ingest 202(批 {len(msgs)} 条)", r.status_code == 202 and r.json().get("accepted") is True, r.text[:100])
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
check("每批 task.result 带 evidence_ids 且条数相符",
      all((t or {}).get("result", {}).get("evidence_ids")
          and len(t["result"]["evidence_ids"]) == len(BATCHES[i])
          for i, t in enumerate(results) if t))
r3 = (results[1] or {}).get("result", {}) if results[1] and results[1]["status"] == "done" else {}
check("换话题批 boundary 非空(前有界可比)", r3.get("boundary") is not None, str(r3.get("boundary"))[:120])

print("== 3. session/end(闭合尾段)==", flush=True)
r = c.post("/api/v1/session/end", json={"session_id": "chat-001"}, headers=H)
check("session/end 202", r.status_code == 202, r.text[:80])
t, err = wait_task(r.json()["task_id"])
check("session/end task done", t is not None and t["status"] == "done", err or str((t or {}).get("error"))[:200])
if t and t["status"] == "done":
    res = t["result"]
    check("closed_now 带 cell_id/topic/atoms",
          res.get("closed_now") and res["closed_now"].get("cell_id") and res["closed_now"].get("atoms", 0) >= 1,
          str(res.get("closed_now"))[:160])
    # 观测项(不 fail):W1 是否把「搬家/看牙医」切成两段由 LLM 判,短段可能合并
    print(f"  [观测] cells_total={res.get('cells_total')} —— 3 句短对话 W1 可能并成 1 段(非缺陷)", flush=True)

print("== 4. recall fast(快链)==", flush=True)
r = c.post("/api/v1/recall",
           json={"session_id": "chat-001", "query": "我现在住在哪?为什么搬家?", "mode": "fast"}, headers=H)
check("recall fast 200", r.status_code == 200, r.text[:200])
o = r.json()
check("字段齐全(query/mode/verdict/critique/retried/answer/cited_cells/memories)",
      all(k in o for k in ("query", "mode", "verdict", "critique", "retried", "answer",
                           "cited_cells", "memories")))
check("mode 回显 fast", o["mode"] == "fast")
check("verdict 在枚举内", o["verdict"] in ("ok", "answer_defect", "insufficient_material"),
      str(o["verdict"]))
check("答案含『裕廊西』", "裕廊西" in o["answer"], o["answer"][:120])
if o["memories"]:
    m = o["memories"][0]
    check("memory 带 atom_id/evidence_refs/evidence 内联",
          m.get("atom_id") and m.get("evidence_refs") and isinstance(m.get("evidence"), list),
          str(m)[:160])
    ev0 = (m.get("evidence") or [{}])[0]
    check("evidence 条目带 holder/content", ev0.get("holder") and ev0.get("content"), str(ev0)[:120])
    smoke_atom_id, smoke_ev_id = m["atom_id"], m["evidence_refs"][0]
else:
    smoke_atom_id = smoke_ev_id = None
    check("memory 带 atom_id/evidence_refs/evidence 内联", False, "memories 为空,无法验证")
print(f"  answer: {o['answer'][:220]}", flush=True)

print("== 5. recall deep(深轨直达)==", flush=True)
r = c.post("/api/v1/recall", json={"session_id": "chat-001", "query": "我下周三有什么安排?", "mode": "deep"}, headers=H)
check("recall deep 200", r.status_code == 200, r.text[:200])
o_deep = r.json()
check("deep 答案含『牙』", "牙" in o_deep["answer"], o_deep["answer"][:160])
print(f"  answer: {o_deep['answer'][:220]}", flush=True)

print("== 6. recall empty 题(诚实义务)==", flush=True)
r = c.post("/api/v1/recall", json={"session_id": "chat-001", "query": "我的猫叫什么名字?"}, headers=H)
check("empty 题 200", r.status_code == 200, r.text[:120])
o_e = r.json()
check("verdict=insufficient_material(诚实拒答,非编造)",
      o_e["verdict"] == "insufficient_material", str(o_e["verdict"]))
if o_e["verdict"] == "insufficient_material":
    check("empty 不下发 memories", o_e["memories"] == [], f"len={len(o_e['memories'])}")
    # 客观交代:深轨作答 or 确定性拼装,都应说清"没找到 + 查过什么";只认否定语族,不写死具体字样
    honest = any(k in o_e["answer"] for k in
                 ("未找到", "没有", "未提及", "无相关", "无法确定", "未包含", "未发现", "无法回答"))
    check("answer 客观交代(否定结论 + 查证过程)", honest, o_e["answer"][:200])
print(f"  answer: {o_e['answer'][:260]}", flush=True)

print("== 7. trace 双向 ==", flush=True)
if smoke_atom_id:
    r = c.get(f"/api/v1/trace/{smoke_atom_id}", headers=H)
    check("trace atom 200 node=memory", r.status_code == 200 and r.json().get("node") == "memory", r.text[:160])
    check("trace atom 下钻 evidence 非空", len(r.json().get("evidence", [])) >= 1)
if smoke_ev_id:
    r = c.get(f"/api/v1/trace/{smoke_ev_id}", headers=H)
    check("trace evidence 200 node=evidence",
          r.status_code == 200 and r.json().get("node") == "evidence", r.text[:160])
    n = r.json()
    check("trace evidence pair 非空", isinstance(n.get("pair"), list) and len(n["pair"]) >= 1)
    check("trace evidence cited_by 含引用", isinstance(n.get("cited_by"), list) and len(n["cited_by"]) >= 1)
r = c.get("/api/v1/trace/atom_bogus", headers=H)
check("trace 未知 id 404", r.status_code == 404, r.text[:80])

print("== 8. tasks / trace 跨 user 隔离 ==", flush=True)
r = c.post("/api/v1/users/register", json={"user_id": OTHER})
tok2 = r.json()["token"]
r = c.get(f"/api/v1/tasks/{task_ids[0]}", headers={"X-User-Token": tok2})
check("别人的 task 404", r.status_code == 404, r.text[:80])
if smoke_atom_id:
    r = c.get(f"/api/v1/trace/{smoke_atom_id}", headers={"X-User-Token": tok2})
    check("别人的记忆 trace 404", r.status_code == 404, r.text[:80])

print("== 9. 非法 mode 400 ==", flush=True)
r = c.post("/api/v1/recall", json={"session_id": "s", "query": "x", "mode": "bogus"}, headers=H)
check("mode=bogus 400", r.status_code == 400 and "mode" in r.json().get("error", ""), r.text[:120])

print("== 10. 上游故障 502 兜底 ==", flush=True)
orig_chat, orig_embed = rt_mod.rt.maas.chat, rt_mod.rt.maas.embed


def boom(*a, **kw):
    raise RuntimeError("simulated upstream outage")


rt_mod.rt.maas.chat = boom
rt_mod.rt.maas.embed = boom
try:
    r = c.post("/api/v1/recall", json={"session_id": "chat-001", "query": "我住哪?"}, headers=H)
    check("上游故障 → 502 JSON(非裸 500)",
          r.status_code == 502 and "不可用" in r.json().get("error", ""), r.text[:160])
finally:
    rt_mod.rt.maas.chat, rt_mod.rt.maas.embed = orig_chat, orig_embed
r = c.post("/api/v1/recall", json={"session_id": "chat-001", "query": "我住哪?"}, headers=H)
check("恢复后 recall 200", r.status_code == 200, r.text[:120])

print("== 11. /health 统计 ==", flush=True)
r = c.get("/api/v1/health")
h = r.json()
check("health 200 且计数>0",
      r.status_code == 200 and h["users"] >= 2 and h["evidence"] >= 3 and h["cells"] >= 1, str(h))

print(f"\n===== 冒烟结果: {len(PASS)} 通过 / {len(FAIL)} 失败 =====", flush=True)
if FAIL:
    for f_ in FAIL:
        print(f"  ✗ {f_}")
sys.exit(1 if FAIL else 0)   # 退出时 atexit 自动清理
