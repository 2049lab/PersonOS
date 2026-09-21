"""评测 trace → 自包含 HTML 全流程报告(融合架构步骤2 快链版)。

设计取向:零依赖(无 CDN/无 JS 框架,断网可开)、一切可下钻(每个环节默认显示结论,
<details> 展开细节/原始 LLM 输出)。报告即审计:入库每轮的边界判定、每个闭合 cell 的
topic/episode/atoms、每题的五件套/双路排名/R5 草稿/核判/判分,全部可见。

trace 结构见 run_locomo.py:ingest(exchanges+closed_cells)/store(cells+atoms+evidence)/
qa(rewrite/hits/ranked/review/r5/memories/gold_chain)/summary。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

# category → 题型名:对齐 Mem0 官方评测代码(memory-benchmarks/locomo/prompts.py)与
# locomo 原仓库 task_eval/evaluation.py:1=multi-hop 2=temporal 3=open-domain 4=single-hop 5=adversarial
_CAT_NAME = {1: "多跳", 2: "时间推理", 3: "开放域", 4: "单跳", 5: "对抗(无金标)"}

_STYLE = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font:14px/1.65 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#f6f7f9;color:#1c1e21}
.wrap{max-width:1100px;margin:0 auto;padding:24px 20px 80px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:36px 0 12px;border-left:4px solid #4a6cf7;padding-left:10px}
.meta{color:#667;font-size:13px}
.card{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:14px 16px;margin:12px 0}
.ok{color:#0a7d32;font-weight:600}.bad{color:#c0392b;font-weight:600}.dim{color:#8a93a0}
.tag{display:inline-block;background:#eef1fe;color:#3b5bdb;border-radius:5px;padding:0 7px;font-size:12px;margin-right:4px}
.tag.g{background:#e6f6ea;color:#0a7d32}.tag.r{background:#fdeaea;color:#c0392b}.tag.y{background:#fff6e0;color:#9a6b00}
pre{background:#f4f5f7;border:1px solid #e4e7ec;border-radius:8px;padding:10px;overflow-x:auto;font-size:12px;white-space:pre-wrap;word-break:break-word}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border:1px solid #e4e7ec;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f0f2f5}
details{margin:6px 0} summary{cursor:pointer;color:#3b5bdb;font-size:13px;user-select:none}
.q{background:#fff;border:1px solid #e4e7ec;border-left:5px solid #4a6cf7;border-radius:10px;padding:12px 16px;margin:14px 0}
.q.badq{border-left-color:#c0392b}
.step{border-left:2px solid #d8dde6;margin:6px 0 6px 10px;padding:4px 0 4px 12px}
.small{font-size:12px;color:#667}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
.flow .n{background:#eef1fe;border-radius:6px;padding:2px 10px;font-size:12px;color:#3b5bdb}
.flow .ar{color:#aab}
"""


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _j(v) -> str:
    return _e(json.dumps(v, ensure_ascii=False))


# —— 入库段:每轮的边界判定 + 每个闭合 cell ——

def _boundary_mark(ex: dict) -> str:
    if ex.get("forced_close"):
        return '<span class="tag r">安全阀闭合</span>'
    b = ex.get("boundary")
    if b is None:
        return '<span class="tag">段首句</span>'
    if b.get("should_end"):
        return f'<span class="tag y">切段(conf={b.get("confidence")})</span>'
    return f'<span class="tag g">延续</span>'


def _cell_html(cb: dict) -> str:
    atoms = "".join(
        f'<div class="small">[{_e(a.get("holder"))}] {_e(a.get("text"))} '
        f'<span class="dim">{_e(a.get("type"))}/{_e(a.get("kind"))} {_e(",".join(a.get("domains") or []))} '
        f'{_e((a.get("occurrence") or "")[:10])} · 证据×{len(a.get("evidence") or [])}</span></div>'
        for a in cb.get("atoms") or [])
    gen = cb.get("gen") or {}
    gen_html = "".join(
        f'<details><summary>W2 {k} 原始输出</summary><pre>{_e(v.get("raw"))}</pre></details>'
        for k, v in gen.items())
    return f"""
<div class="card">
  <b>cell {_e(cb.get("topic"))}</b> <span class="tag">{_e(cb.get("t_start") or "?")[:10]}~{_e(cb.get("t_end") or "?")[:10]}</span>
  <span class="tag">{_j(cb.get("domains") or [])}</span> · {len(cb.get("atoms") or [])} atoms
  <div style="margin:4px 0">{_e(cb.get("episode"))}</div>
  {atoms}
  {gen_html}
</div>"""


def _ingest_html(trace: dict) -> str:
    out = []
    for s in trace.get("ingest", []):
        ex_rows = []
        for i, ex in enumerate(s.get("exchanges", []), 1):
            b = ex.get("boundary") or {}
            hint = f' <span class="dim">{_e(b.get("topic_summary"))}</span>' if b.get("should_end") else ""
            ex_rows.append(f"""
<div class="card">
  <div><b>轮 {i}</b> <span class="tag">{_e(ex.get("holder"))}</span>
   <span class="dim">{_e(ex.get("dia"))}</span> {_boundary_mark(ex)}{hint}</div>
  <div style="margin:4px 0">{_e(ex.get("text"))}</div>
</div>""")
        cells_html = "".join(_cell_html(cb) for cb in s.get("closed_cells", []))
        out.append(f"""
<h2>①-{s["session"]} · Session {s["session"]} <span class="dim">@{_e(s["dt"][:16])} · {len(s.get("exchanges", []))} 轮 · {s.get("secs")}s</span></h2>
{''.join(ex_rows)}
<div class="small">本会话闭合 {len(s.get("closed_cells", []))} 个 cell(累计 {s.get("cells_total")}),全局 {s.get("atoms_total")} atoms</div>
{cells_html}""")
    return "".join(out) or '<div class="dim">(--skip-ingest 复用库,本次未重灌;入库细节见上一次完整运行)</div>'


# —— 库全貌 ——

def _store_html(trace: dict) -> str:
    st = trace.get("store", {})
    cells = st.get("cells", [])
    cell_rows = "".join(
        f"<tr><td class=\"small\">{_e(c.get('t_start', '')[:10])}</td><td>{_e(c.get('topic'))}</td>"
        f"<td>{_e(','.join(c.get('domains') or []))}</td><td>{_e(c.get('episode'))[:160]}</td>"
        f"<td>{c.get('n_atoms')}</td></tr>"
        for c in cells)
    atoms = st.get("atoms", [])
    atom_rows = "".join(
        f"<tr><td>{i}</td><td>{_e(a['type'])}/{_e(a['kind'])}</td><td>{_e(a['holder'])}</td>"
        f"<td>{_e(a['text'])}</td><td>{_e(','.join(a['domains']))}</td>"
        f"<td>{_e((a.get('occurrence') or '')[:10])}</td><td>{len(a['evidence'])}</td></tr>"
        for i, a in enumerate(atoms, 1))
    evs = st.get("evidence", [])
    ev_rows = "".join(
        f"<tr><td class=\"small\">{_e(e['at'][:16])}</td><td>{_e(e['holder'])}</td><td>{_e(e['content'])}</td></tr>"
        for e in evs)
    return f"""
<h2>② 记忆库全貌(评测后终态)</h2>
<div class="card"><b>cell 层 {len(cells)} 段</b>(episode 叙事 + 挂 atoms)
<table><tr><th>段起始</th><th>topic</th><th>D域</th><th>episode(截160)</th><th>atoms</th></tr>{cell_rows}</table></div>
<div class="card"><b>原子层 {len(atoms)} 条</b>(检索单元,指向 cell 与证据)
<table><tr><th>#</th><th>类型/K</th><th>holder</th><th>命题文本</th><th>D域</th><th>发生</th><th>证据数</th></tr>{atom_rows}</table></div>
<div class="card"><b>证据层 {len(evs)} 条</b>(原话,不可变真相源)
<table><tr><th>时间</th><th>说话人</th><th>原话</th></tr>{ev_rows}</table></div>"""


# —— QA 段 ——

def _error_kind(q: dict) -> str:
    """错题的初步归因(启发式,最终以 break_point/人工复核为准)。run_all 聚合也用它。"""
    ans = (q.get("answer") or "").strip()
    if ans.lower().startswith("i don't have"):
        verdict = (q.get("review") or {}).get("verdict")
        if verdict == "insufficient_material":
            return "拒答·核判材料不足(查抽取层:该信息是否被抽出;再查检索)"
        return "拒答·核判判defect仍拒答(查R5作答/answerer是否过保守)"
    return "答案与金标不符(查检索排名→R5材料→answerer→judge)"


def _chain_html(q: dict) -> str:
    """金标证据链可视化:每条金标原话走到了哪一环(✓/✗),断点即错误点。"""
    chains = q.get("gold_chain") or []
    if not chains:
        return ""
    rows = []
    for c in chains:
        def mk(ok):
            return '<span class="ok">✓</span>' if ok else '<span class="bad">✗</span>'
        atoms = f'<div class="small">抽取为: {"; ".join(_e(a) for a in c["extracted_atoms"])}</div>' \
            if c.get("extracted_atoms") else ""
        rows.append(f"""<div class="step"><code>{_e(c['dia'])}</code>
<div class="small">链路:{mk(bool(c['ev_id']))}入证据层 → {mk(c['n_extracted']>0)}抽成原子({c['n_extracted']}条)
 → {mk(c['in_hits'])}检索top10 → {mk(c['in_material'])}作答材料 → {mk(c['cited'])}R5引用</div>{atoms}</div>""")
    bp = q.get("break_point")
    bp_html = f'<div class="tag r">断点: {_e(bp)}</div>' if bp else ""
    return f'<div style="margin:4px 0">{bp_html}{"".join(rows)}</div>'


def _hits_table(hits: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{h.get('rank')}</td><td>{_e(h.get('topic'))}</td>"
        f"<td>{h.get('rrf')}</td><td>{h.get('best_sim')}</td>"
        f"<td>{'织写链' if h.get('covers') else '普通格'}</td>"
        f"<td>{_e(h.get('rerank_score'))}</td><td>{len(h.get('atoms') or [])}</td></tr>"
        for h in hits)
    return (f'<table><tr><th>#</th><th>topic</th><th>RRF分</th><th>MaxSim</th>'
            f'<th>单元</th><th>rerank分</th><th>atoms</th></tr>{rows}</table>')


def _errors_html(trace: dict) -> str:
    wrong = [q for q in trace.get("qa", []) if not q.get("judge")]
    if not wrong:
        return '<h2>⑤ 错题分析</h2><div class="card">无错题 🎉</div>'
    out = ['<h2>⑤ 错题分析</h2>']
    kinds: dict[str, int] = {}
    for q in wrong:
        k = q.get("break_point") or _error_kind(q)
        kinds[k] = kinds.get(k, 0) + 1
    stat = " · ".join(f"{k} ×{v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1]))
    out.append(f'<div class="card"><b>共 {len(wrong)} 题判错,断点分布:</b><br>{_e(stat)}'
               '<div class="small">断点=金标证据链最先断的环节(错误点);链路各环见每题展开。</div></div>')
    for i, q in enumerate(wrong, 1):
        mems = "<br>".join(f"- {_e(m['text'])}" for m in q.get("memories", [])[:4]) or "(无)"
        missing = (q.get("review") or {}).get("critique") or ""
        out.append(f"""
<div class="q badq">
  <div><b>E{i} [{_CAT_NAME.get(q["category"], q["category"])}]</b> {_e(q["question"])}
   <span class="tag r">断点: {_e(q.get("break_point") or _error_kind(q))}</span></div>
  <div>金标: <b>{_e(q["gold"])}</b> | evidence: {_j(q["evidence"])}</div>
  {_chain_html(q)}
  <div class="small">核判指正: {_e(missing[:200])}</div>
  <div class="small">采信记忆:<br>{mems}</div>
  <div style="margin-top:4px">答案: {_e((q.get("answer") or "")[:200])} · judge: {_e(q.get("judge_raw"))}</div>
</div>""")
    return "".join(out)


def _mem_html(m: dict) -> str:
    """一条采信记忆 + 其支撑原话(evidence_entries 同口径 Q↔A,报告里可对照)。"""
    ev = m.get("evidence") or []
    ev_html = "".join(
        f'<div class="small">[{_e(e.get("at"))}] {_e(e.get("holder"))}: {_e(e.get("content"))}</div>'
        for e in ev) or '<div class="small dim">(无挂靠证据)</div>'
    return f'<div class="step"><b>记忆:</b> {_e(m.get("text"))}{ev_html}</div>'


def _qa_html(trace: dict) -> str:
    out = []
    for i, q in enumerate(trace.get("qa", []), 1):
        cls = "" if q["judge"] else " badq"
        mark = '<span class="ok">✓ 判对</span>' if q["judge"] else '<span class="bad">✗ 判错</span>'
        mark5 = ('<span class="ok">✓ R5直判对</span>' if q.get("judge_r5")
                 else '<span class="bad">✗ R5直判错</span>')
        rv = q.get("review") or {}
        r5 = q.get("r5") or {}
        rw = q.get("rewrite") or {}
        vcls = {"ok": "g", "answer_defect": "y", "insufficient_material": "r"}.get(
            rv.get("verdict"), "")
        flow = ['<span class="n">R0 五件套</span><span class="ar">→</span>'
                '<span class="n">R1 双路MaxSim+RRF</span><span class="ar">→</span>'
                '<span class="n">R2 精排</span><span class="ar">→</span>'
                '<span class="n">R5 草稿</span><span class="ar">→</span>'
                f'<span class="n">R3\' 核判 {_e(rv.get("verdict"))}'
                f'{"(重答)" if rv.get("retried") else ""}</span><span class="ar">→</span>'
                '<span class="n">answerer</span><span class="ar">→</span>'
                '<span class="n">judge×双口径</span>']
        mem_html = "".join(_mem_html(m) for m in q.get("memories") or []) \
            or '<div class="dim">(无命中记忆)</div>'
        out.append(f"""
<div class="q{cls}">
  <div><b>Q{i} [{_CAT_NAME.get(q["category"], q["category"])}]</b> {mark} · {mark5}
   <span class="tag {vcls}">核判 {_e(rv.get("verdict"))}</span><span class="tag">{q["secs"]}s</span></div>
  <div style="margin:6px 0"><b>题目:</b> {_e(q["question"])}</div>
  <div><b>金标:</b> {_e(q["gold"])} <span class="dim">| evidence: {_j(q["evidence"])}</span></div>
  <div class="flow">{ ''.join(flow) }</div>
  <details><summary>展开:R0 五件套(检索的实际输入)</summary>
    <div class="small">resolved: {_e(rw.get("resolved"))}<br>
    subject: {_e(rw.get("subject"))} | 扩展词: {_j(rw.get("expansions"))} |
    时间窗: {_j(rw.get("time_window"))} | 判域: {_j(rw.get("domains"))}</div>
  </details>
  <details><summary>展开:R1 双路融合序(top10)与 R2 精排序(top10)</summary>
    {_hits_table(q.get("hits") or [])}
    <div class="small" style="margin:6px 0">R2 精排序(rerank_score 空=保序):</div>
    {_hits_table(q.get("ranked") or [])}
  </details>
  <details><summary>展开:R3' 核判与 R5 作答</summary>
    <div class="small">verdict={_e(rv.get("verdict"))} retried={_e(rv.get("retried"))}
     critique={_e(rv.get("critique"))}</div>
    <div class="small">R5 直答(=answerer 的 brief): {_e((r5.get("answer") or "")[:400])}</div>
    <div class="small">R5 引用 cells: {_j(r5.get("cited_cells"))}</div>
  </details>
  <details open><summary>命中记忆(answerer 材料,精排前20 cell)及支撑原话</summary>{mem_html}</details>
  <div style="margin-top:6px"><b>模式A答案:</b> {_e(q["answer"])}</div>
  <div><b>judge(Mem0口径):</b> {_e(q["judge_raw"])}</div>
  <div><b>judge_r5(产品口径):</b> {_e(q.get("judge_r5_raw"))}</div>
</div>""")
    return "".join(out)


def _summary_html(trace: dict) -> str:
    s = trace.get("summary", {})
    by = s.get("by_category", {})
    vs = s.get("verdicts", {})
    n = max(1, s.get("n_questions", 0))
    rows = "".join(f"<tr><td>cat{k} {_CAT_NAME.get(int(k), '')}</td><td>{v[0]}/{v[1]}</td>"
                   f"<td>{100*v[0]/max(1,v[1]):.0f}%</td></tr>" for k, v in by.items())
    return f"""
<h2>④ 汇总</h2>
<div class="card"><b>Mem0口径(answerer): {s.get("score")}/{s.get("n_questions")}
({100*s.get("score", 0)/n:.0f}%) · 产品口径(R5直判): {s.get("score_r5", 0)}/
{s.get("n_questions")} ({100*s.get("score_r5", 0)/n:.0f}%)</b>
<table><tr><th>题型</th><th>对/总(Mem0口径)</th><th>正确率</th></tr>{rows}</table>
<div class="small" style="margin-top:8px">核判判级: ok {vs.get("ok", 0)} ·
answer_defect {vs.get("answer_defect", 0)}(重答 {s.get("retried", 0)} 次) ·
insufficient_material {vs.get("insufficient_material", 0)} · 升深轨 {s.get("escalated", 0)} 题</div>
<div class="small">注意:judge 为单模型单次判定(语义等价类可能误判),以链路展开为准;
基线对比时确认 sessions/选题范围一致;对外对比时声明双口径协议差异。</div></div>"""


def render(trace: dict, out_path: str | Path) -> Path:
    m = trace.get("meta", {})
    body = f"""
<div class="wrap">
<h1>LoCoMo × personos · 评测全流程报告(融合架构快链)</h1>
<div class="meta">对话 {_e(m.get("conv"))} · 前 {_e(m.get("n_sessions"))} 个 session · 模式 {_e(m.get("mode"))} ·
LLM/判分: {_e(m.get("llm"))} / {_e(m.get("judge"))} · answerer: {_e(m.get("answerer"))} ·
embedding: MAAS qwen3 · git {_e(m.get("git"))}</div>
{_summary_html(trace)}
<h2>① 入库流程(W0 证据 → W1 边界 → W2 建 cell)</h2>
{_ingest_html(trace)}
{_store_html(trace)}
<h2>③ QA 评测(每题:R0→R1→R2→R5草稿→R3'核判→answerer→judge×双口径)</h2>
{_qa_html(trace)}
{_errors_html(trace)}
</div>"""
    doc = f"<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">" \
          f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">" \
          f"<title>LoCoMo × personos 评测报告 · {_e(m.get('conv'))}</title><style>{_STYLE}</style></head>" \
          f"<body>{body}</body></html>"
    p = Path(out_path)
    p.write_text(doc, encoding="utf-8")
    return p


def _t(s) -> str:
    """markdown 用纯文本(markdown 不吃 HTML 转义,只防表格竖线)。"""
    return str(s if s is not None else "").replace("|", "\|")


def render_md(trace: dict, out_path: str | Path) -> Path:
    """最终结果报告(markdown,归档/git 友好):双口径总分 + 分题型 + 核判判级 + 错题分析。"""
    m = trace.get("meta", {})
    s = trace.get("summary", {})
    qa = trace.get("qa", [])
    st = trace.get("store", {})
    lines = [f"# LoCoMo × personos 最终评测报告 · {_t(m.get('conv'))}", ""]
    lines.append(f"- 运行: {_t(m.get('run_at', ''))} · 模式 {_t(m.get('mode'))} · "
                 f"LLM/判分 {_t(m.get('llm'))} / {_t(m.get('judge'))} · git `{_t(m.get('git', ''))}`")
    lines.append(f"- answerer: {_t(m.get('answerer', ''))}")
    lines.append(f"- 范围: 前 {m.get('n_sessions')} 个 session · 有效题 {s.get('n_questions')} 道(剔 cat5 对抗题)")
    lines.append(f"- 记忆库终态: {len(st.get('cells', []))} cells / {len(st.get('atoms', []))} atoms / "
                 f"{len(st.get('evidence', []))} 证据")
    n = max(1, len(qa))
    lines.append(f"- 平均耗时 {sum(q.get('secs', 0) for q in qa)/n:.0f}s/题")
    lines.append("")
    lines.append(f"## 总分: **Mem0口径 {s.get('score')}/{s.get('n_questions')} "
                 f"({100*s.get('score', 0)/n:.1f}%) · 产品口径(R5直判) "
                 f"{s.get('score_r5', 0)}/{s.get('n_questions')} "
                 f"({100*s.get('score_r5', 0)/n:.1f}%)**")
    lines.append("")
    lines.append("| 题型 | 对/总(Mem0口径) | 正确率 |")
    lines.append("|---|---|---|")
    for k, v in sorted(s.get("by_category", {}).items()):
        lines.append(f"| cat{k} {_CAT_NAME.get(int(k), '')} | {v[0]}/{v[1]} | {100*v[0]/max(1,v[1]):.0f}% |")
    vs = s.get("verdicts", {})
    lines.append("")
    lines.append(f"核判判级: ok {vs.get('ok', 0)} · answer_defect {vs.get('answer_defect', 0)}"
                 f"(重答 {s.get('retried', 0)} 次) · insufficient_material "
                 f"{vs.get('insufficient_material', 0)} · 升深轨 {s.get('escalated', 0)} 题")
    lines.append("")
    lines.append("## 错题分析")
    wrong = [q for q in qa if not q.get("judge")]
    if not wrong:
        lines.append("无错题。")
    for i, q in enumerate(wrong, 1):
        lines.append(f"### E{i} [{_CAT_NAME.get(q['category'], q['category'])}] {_t(q['question'])}")
        lines.append(f"- 金标: **{_t(q['gold'])}** · evidence: {', '.join(q.get('evidence', []))}")
        lines.append(f"- 断点: **{_t(q.get('break_point') or _error_kind(q))}**")
        for c in (q.get("gold_chain") or []):
            marks = (f"入证据{'✓' if c['ev_id'] else '✗'}→抽原子{'✓' if c['n_extracted'] else '✗'}({c['n_extracted']})"
                     f"→检索top10{'✓' if c['in_hits'] else '✗'}→材料{'✓' if c['in_material'] else '✗'}"
                     f"→R5引用{'✓' if c['cited'] else '✗'}")
            lines.append(f"  - `{_t(c['dia'])}` {marks}")
        rv = q.get("review") or {}
        lines.append(f"- 核判: {_t(rv.get('verdict'))}{'(重答)' if rv.get('retried') else ''} · "
                     f"指正: {_t((rv.get('critique') or '')[:120])}")
        lines.append(f"- 记忆: {'; '.join(_t(m2['text'])[:80] for m2 in q.get('memories', [])[:3]) or '(无)'}")
        lines.append(f"- 答案: {_t((q.get('answer') or '')[:150])} · judge: {_t(q.get('judge_raw'))}"
                     f" · R5直判: {'✓' if q.get('judge_r5') else '✗'} {_t(q.get('judge_r5_raw'))}")
        lines.append("")
    lines.append("> 断点=金标证据链最先断的环节;细节见同目录 trace JSON 与 HTML 报告(每题链路可逐环展开)。")
    p = Path(out_path)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


if __name__ == "__main__":
    # 从任意 trace.json 重生成报告(评测跑完后代码更新了也不必重跑):
    #   python -m scripts.bench.report data/bench/conv26.trace.json [输出前缀]
    import sys
    t = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1]).with_suffix("")
    print(render(t, Path(str(prefix) + ".report.html")))
    print(render_md(t, Path(str(prefix) + ".report.md")))
