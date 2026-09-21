"""Benchmark trace -> a self-contained, full-pipeline HTML report.

Design stance: zero dependencies (no CDN, no JS framework, so it opens
offline), and everything can be drilled into (each stage shows its conclusion
by default, with <details> expanding to the detail and the raw LLM output). The
report is the audit: the boundary decision for every ingested turn, the
topic/episode/atoms of every closed cell, and the scope, dual-path ranking, R5
draft, review and score of every question are all visible.

The trace structure is defined in run_locomo.py: ingest (exchanges +
closed_cells) / store (cells + atoms + evidence) / qa (rewrite, hits, ranked,
review, r5, memories, gold_chain) / summary.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

# category -> question-type name, matching Mem0's official evaluation code
# (memory-benchmarks/locomo/prompts.py) and the original locomo repository's
# task_eval/evaluation.py.
_CAT_NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop",
             5: "adversarial (no gold answer)"}

_STYLE = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1c1e21}
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


# -- Ingest section: the boundary decision per turn, plus every closed cell --

def _boundary_mark(ex: dict) -> str:
    if ex.get("forced_close"):
        return '<span class="tag r">closed by safety valve</span>'
    b = ex.get("boundary")
    if b is None:
        return '<span class="tag">first line of a segment</span>'
    if b.get("should_end"):
        return f'<span class="tag y">split (conf={b.get("confidence")})</span>'
    return f'<span class="tag g">continues</span>'


def _cell_html(cb: dict) -> str:
    atoms = "".join(
        f'<div class="small">[{_e(a.get("holder"))}] {_e(a.get("text"))} '
        f'<span class="dim">{_e(a.get("type"))}/{_e(a.get("kind"))} {_e(",".join(a.get("domains") or []))} '
        f'{_e((a.get("occurrence") or "")[:10])} · evidence x{len(a.get("evidence") or [])}</span></div>'
        for a in cb.get("atoms") or [])
    gen = cb.get("gen") or {}
    gen_html = "".join(
        f'<details><summary>W2 {k} raw output</summary><pre>{_e(v.get("raw"))}</pre></details>'
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
  <div><b>turn {i}</b> <span class="tag">{_e(ex.get("holder"))}</span>
   <span class="dim">{_e(ex.get("dia"))}</span> {_boundary_mark(ex)}{hint}</div>
  <div style="margin:4px 0">{_e(ex.get("text"))}</div>
</div>""")
        cells_html = "".join(_cell_html(cb) for cb in s.get("closed_cells", []))
        out.append(f"""
<h2>1.{s["session"]} - Session {s["session"]} <span class="dim">@{_e(s["dt"][:16])} - {len(s.get("exchanges", []))} turns - {s.get("secs")}s</span></h2>
{''.join(ex_rows)}
<div class="small">{len(s.get("closed_cells", []))} cells closed in this session ({s.get("cells_total")} cumulative), {s.get("atoms_total")} atoms in total</div>
{cells_html}""")
    return "".join(out) or ('<div class="dim">(--skip-ingest reused the store, so nothing was '
                            'loaded this run; see the previous full run for ingest detail)</div>')


# -- The whole store --

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
<h2>2. The memory store (final state after the benchmark)</h2>
<div class="card"><b>Cell layer: {len(cells)} segments</b> (episode narrative with atoms attached)
<table><tr><th>starts</th><th>topic</th><th>domains</th><th>episode (first 160)</th><th>atoms</th></tr>{cell_rows}</table></div>
<div class="card"><b>Atom layer: {len(atoms)} atoms</b> (the retrieval unit, pointing at a cell and its evidence)
<table><tr><th>#</th><th>type/kind</th><th>holder</th><th>proposition</th><th>domains</th><th>occurred</th><th>evidence</th></tr>{atom_rows}</table></div>
<div class="card"><b>Evidence layer: {len(evs)} records</b> (verbatim text, the immutable source of truth)
<table><tr><th>time</th><th>speaker</th><th>verbatim</th></tr>{ev_rows}</table></div>"""


# -- QA section --

def _error_kind(q: dict) -> str:
    """A first-pass attribution for a wrong answer (heuristic; break_point and
    human review are the final word). run_all's aggregation uses it too."""
    ans = (q.get("answer") or "").strip()
    if ans.lower().startswith("i don't have"):
        verdict = (q.get("review") or {}).get("verdict")
        if verdict == "insufficient_material":
            return ("refusal - review found insufficient material (check extraction: was the "
                    "information extracted at all? then check retrieval)")
        return ("refusal - review returned a defect and it still refused (check whether R5 "
                "answering or the answerer is too conservative)")
    return "answer disagrees with the gold (check ranking -> R5 materials -> answerer -> judge)"


def _chain_html(q: dict) -> str:
    """Visualize the gold evidence chain: how far each gold verbatim line got at
    each stage. The break point is where the error is."""
    chains = q.get("gold_chain") or []
    if not chains:
        return ""
    rows = []
    for c in chains:
        def mk(ok):
            return '<span class="ok">✓</span>' if ok else '<span class="bad">✗</span>'
        atoms = f'<div class="small">extracted as: {"; ".join(_e(a) for a in c["extracted_atoms"])}</div>' \
            if c.get("extracted_atoms") else ""
        rows.append(f"""<div class="step"><code>{_e(c['dia'])}</code>
<div class="small">pipeline: {mk(bool(c['ev_id']))} stored as evidence -> {mk(c['n_extracted']>0)} extracted into atoms ({c['n_extracted']})
 -> {mk(c['in_hits'])} retrieval top 10 -> {mk(c['in_material'])} answering materials -> {mk(c['cited'])} cited by R5</div>{atoms}</div>""")
    bp = q.get("break_point")
    bp_html = f'<div class="tag r">break point: {_e(bp)}</div>' if bp else ""
    return f'<div style="margin:4px 0">{bp_html}{"".join(rows)}</div>'


def _hits_table(hits: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{h.get('rank')}</td><td>{_e(h.get('topic'))}</td>"
        f"<td>{h.get('rrf')}</td><td>{h.get('best_sim')}</td>"
        f"<td>{'woven chain' if h.get('covers') else 'plain cell'}</td>"
        f"<td>{_e(h.get('rerank_score'))}</td><td>{len(h.get('atoms') or [])}</td></tr>"
        for h in hits)
    return (f'<table><tr><th>#</th><th>topic</th><th>RRF</th><th>MaxSim</th>'
            f'<th>unit</th><th>rerank score</th><th>atoms</th></tr>{rows}</table>')


def _errors_html(trace: dict) -> str:
    wrong = [q for q in trace.get("qa", []) if not q.get("judge")]
    if not wrong:
        return '<h2>5. Error analysis</h2><div class="card">no wrong answers</div>'
    out = ['<h2>5. Error analysis</h2>']
    kinds: dict[str, int] = {}
    for q in wrong:
        k = q.get("break_point") or _error_kind(q)
        kinds[k] = kinds.get(k, 0) + 1
    stat = " · ".join(f"{k} ×{v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1]))
    out.append(f'<div class="card"><b>{len(wrong)} questions scored wrong; break-point '
               f'distribution:</b><br>{_e(stat)}'
               '<div class="small">The break point is the first stage at which the gold evidence '
               'chain broke, i.e. where the error is. Expand a question to see every stage.</div></div>')
    for i, q in enumerate(wrong, 1):
        mems = "<br>".join(f"- {_e(m['text'])}" for m in q.get("memories", [])[:4]) or "(none)"
        missing = (q.get("review") or {}).get("critique") or ""
        out.append(f"""
<div class="q badq">
  <div><b>E{i} [{_CAT_NAME.get(q["category"], q["category"])}]</b> {_e(q["question"])}
   <span class="tag r">break point: {_e(q.get("break_point") or _error_kind(q))}</span></div>
  <div>gold: <b>{_e(q["gold"])}</b> | evidence: {_j(q["evidence"])}</div>
  {_chain_html(q)}
  <div class="small">review critique: {_e(missing[:200])}</div>
  <div class="small">memories relied on:<br>{mems}</div>
  <div style="margin-top:4px">answer: {_e((q.get("answer") or "")[:200])} - judge: {_e(q.get("judge_raw"))}</div>
</div>""")
    return "".join(out)


def _mem_html(m: dict) -> str:
    """One memory that was relied on, plus the verbatim text supporting it (the
    same Q/A convention as evidence_entries, so the report can be cross-checked)."""
    ev = m.get("evidence") or []
    ev_html = "".join(
        f'<div class="small">[{_e(e.get("at"))}] {_e(e.get("holder"))}: {_e(e.get("content"))}</div>'
        for e in ev) or '<div class="small dim">(no attached evidence)</div>'
    return f'<div class="step"><b>memory:</b> {_e(m.get("text"))}{ev_html}</div>'


def _qa_html(trace: dict) -> str:
    out = []
    for i, q in enumerate(trace.get("qa", []), 1):
        cls = "" if q["judge"] else " badq"
        mark = '<span class="ok">correct</span>' if q["judge"] else '<span class="bad">wrong</span>'
        mark5 = ('<span class="ok">R5 correct</span>' if q.get("judge_r5")
                 else '<span class="bad">R5 wrong</span>')
        rv = q.get("review") or {}
        r5 = q.get("r5") or {}
        rw = q.get("rewrite") or {}
        vcls = {"ok": "g", "answer_defect": "y", "insufficient_material": "r"}.get(
            rv.get("verdict"), "")
        flow = ['<span class="n">R0 scope</span><span class="ar">-&gt;</span>'
                '<span class="n">R1 dual path MaxSim+RRF</span><span class="ar">-&gt;</span>'
                '<span class="n">R2 rerank</span><span class="ar">-&gt;</span>'
                '<span class="n">R5 draft</span><span class="ar">-&gt;</span>'
                f'<span class="n">R3\' review {_e(rv.get("verdict"))}'
                f'{" (re-answered)" if rv.get("retried") else ""}</span><span class="ar">-&gt;</span>'
                '<span class="n">answerer</span><span class="ar">-&gt;</span>'
                '<span class="n">judge, both conventions</span>']
        mem_html = "".join(_mem_html(m) for m in q.get("memories") or []) \
            or '<div class="dim">(no memories hit)</div>'
        out.append(f"""
<div class="q{cls}">
  <div><b>Q{i} [{_CAT_NAME.get(q["category"], q["category"])}]</b> {mark} - {mark5}
   <span class="tag {vcls}">review {_e(rv.get("verdict"))}</span><span class="tag">{q["secs"]}s</span></div>
  <div style="margin:6px 0"><b>question:</b> {_e(q["question"])}</div>
  <div><b>gold:</b> {_e(q["gold"])} <span class="dim">| evidence: {_j(q["evidence"])}</span></div>
  <div class="flow">{ ''.join(flow) }</div>
  <details><summary>expand: R0 retrieval scope (what retrieval actually received)</summary>
    <div class="small">resolved: {_e(rw.get("resolved"))}<br>
    subject: {_e(rw.get("subject"))} | expansions: {_j(rw.get("expansions"))} |
    time window: {_j(rw.get("time_window"))} | domains: {_j(rw.get("domains"))}</div>
  </details>
  <details><summary>expand: R1 fused order (top 10) and R2 reranked order (top 10)</summary>
    {_hits_table(q.get("hits") or [])}
    <div class="small" style="margin:6px 0">R2 reranked order (an empty rerank_score means the order was kept):</div>
    {_hits_table(q.get("ranked") or [])}
  </details>
  <details><summary>expand: R3' review and the R5 answer</summary>
    <div class="small">verdict={_e(rv.get("verdict"))} retried={_e(rv.get("retried"))}
     critique={_e(rv.get("critique"))}</div>
    <div class="small">R5 answer (the answerer's brief): {_e((r5.get("answer") or "")[:400])}</div>
    <div class="small">cells cited by R5: {_j(r5.get("cited_cells"))}</div>
  </details>
  <details open><summary>memories hit (the answerer's materials, top 20 reranked cells) and their supporting text</summary>{mem_html}</details>
  <div style="margin-top:6px"><b>mode A answer:</b> {_e(q["answer"])}</div>
  <div><b>judge (Mem0 convention):</b> {_e(q["judge_raw"])}</div>
  <div><b>judge_r5 (product convention):</b> {_e(q.get("judge_r5_raw"))}</div>
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
<h2>4. Summary</h2>
<div class="card"><b>Mem0 convention (answerer): {s.get("score")}/{s.get("n_questions")}
({100*s.get("score", 0)/n:.0f}%) - product convention (R5 judged directly): {s.get("score_r5", 0)}/
{s.get("n_questions")} ({100*s.get("score_r5", 0)/n:.0f}%)</b>
<table><tr><th>category</th><th>correct/total (Mem0 convention)</th><th>accuracy</th></tr>{rows}</table>
<div class="small" style="margin-top:8px">review verdicts: ok {vs.get("ok", 0)} -
answer_defect {vs.get("answer_defect", 0)} ({s.get("retried", 0)} re-answers) -
insufficient_material {vs.get("insufficient_material", 0)} - escalated to the deep path on
{s.get("escalated", 0)} questions</div>
<div class="small">Note: the judge is a single model making a single decision and can misjudge
semantic equivalence, so expand the pipeline before drawing conclusions. When comparing against a
baseline, confirm the sessions and question selection match, and when comparing externally, state
the difference between the two scoring conventions.</div></div>"""


def render(trace: dict, out_path: str | Path) -> Path:
    m = trace.get("meta", {})
    body = f"""
<div class="wrap">
<h1>LoCoMo x personos - full-pipeline benchmark report</h1>
<div class="meta">conversation {_e(m.get("conv"))} - first {_e(m.get("n_sessions"))} sessions - mode {_e(m.get("mode"))} -
LLM / judge: {_e(m.get("llm"))} / {_e(m.get("judge"))} - answerer: {_e(m.get("answerer"))} -
git {_e(m.get("git"))}</div>
{_summary_html(trace)}
<h2>1. Ingest pipeline (W0 evidence -&gt; W1 boundaries -&gt; W2 cell building)</h2>
{_ingest_html(trace)}
{_store_html(trace)}
<h2>3. QA benchmark (per question: R0 -&gt; R1 -&gt; R2 -&gt; R5 draft -&gt; R3' review -&gt; answerer -&gt; judge, both conventions)</h2>
{_qa_html(trace)}
{_errors_html(trace)}
</div>"""
    doc = f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">" \
          f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">" \
          f"<title>LoCoMo x personos benchmark report - {_e(m.get('conv'))}</title><style>{_STYLE}</style></head>" \
          f"<body>{body}</body></html>"
    p = Path(out_path)
    p.write_text(doc, encoding="utf-8")
    return p


def _t(s) -> str:
    """Plain text for markdown: markdown does not want HTML escaping, so only
    the table pipe character is guarded."""
    return str(s if s is not None else "").replace("|", r"\|")


def render_md(trace: dict, out_path: str | Path) -> Path:
    """The final results report in markdown, friendly to archiving and git:
    totals under both conventions, a per-category breakdown, review verdicts and
    an error analysis."""
    m = trace.get("meta", {})
    s = trace.get("summary", {})
    qa = trace.get("qa", [])
    st = trace.get("store", {})
    lines = [f"# LoCoMo x personos final benchmark report - {_t(m.get('conv'))}", ""]
    lines.append(f"- run: {_t(m.get('run_at', ''))} - mode {_t(m.get('mode'))} - "
                 f"LLM / judge {_t(m.get('llm'))} / {_t(m.get('judge'))} - git `{_t(m.get('git', ''))}`")
    lines.append(f"- answerer: {_t(m.get('answerer', ''))}")
    lines.append(f"- scope: the first {m.get('n_sessions')} sessions - {s.get('n_questions')} "
                 f"valid questions (cat5 adversarial questions excluded)")
    lines.append(f"- final store state: {len(st.get('cells', []))} cells / "
                 f"{len(st.get('atoms', []))} atoms / {len(st.get('evidence', []))} evidence records")
    n = max(1, len(qa))
    lines.append(f"- average {sum(q.get('secs', 0) for q in qa)/n:.0f}s per question")
    lines.append("")
    lines.append(f"## Total: **Mem0 convention {s.get('score')}/{s.get('n_questions')} "
                 f"({100*s.get('score', 0)/n:.1f}%) - product convention (R5 judged directly) "
                 f"{s.get('score_r5', 0)}/{s.get('n_questions')} "
                 f"({100*s.get('score_r5', 0)/n:.1f}%)**")
    lines.append("")
    lines.append("| category | correct/total (Mem0 convention) | accuracy |")
    lines.append("|---|---|---|")
    for k, v in sorted(s.get("by_category", {}).items()):
        lines.append(f"| cat{k} {_CAT_NAME.get(int(k), '')} | {v[0]}/{v[1]} | {100*v[0]/max(1,v[1]):.0f}% |")
    vs = s.get("verdicts", {})
    lines.append("")
    lines.append(f"review verdicts: ok {vs.get('ok', 0)} - answer_defect {vs.get('answer_defect', 0)}"
                 f" ({s.get('retried', 0)} re-answers) - insufficient_material "
                 f"{vs.get('insufficient_material', 0)} - escalated to the deep path on "
                 f"{s.get('escalated', 0)} questions")
    lines.append("")
    lines.append("## Error analysis")
    wrong = [q for q in qa if not q.get("judge")]
    if not wrong:
        lines.append("No wrong answers.")
    for i, q in enumerate(wrong, 1):
        lines.append(f"### E{i} [{_CAT_NAME.get(q['category'], q['category'])}] {_t(q['question'])}")
        lines.append(f"- gold: **{_t(q['gold'])}** - evidence: {', '.join(q.get('evidence', []))}")
        lines.append(f"- break point: **{_t(q.get('break_point') or _error_kind(q))}**")
        for c in (q.get("gold_chain") or []):
            marks = (f"evidence {'y' if c['ev_id'] else 'n'} -> atoms "
                     f"{'y' if c['n_extracted'] else 'n'} ({c['n_extracted']})"
                     f" -> retrieval top10 {'y' if c['in_hits'] else 'n'}"
                     f" -> materials {'y' if c['in_material'] else 'n'}"
                     f" -> cited by R5 {'y' if c['cited'] else 'n'}")
            lines.append(f"  - `{_t(c['dia'])}` {marks}")
        rv = q.get("review") or {}
        lines.append(f"- review: {_t(rv.get('verdict'))}{' (re-answered)' if rv.get('retried') else ''} - "
                     f"critique: {_t((rv.get('critique') or '')[:120])}")
        lines.append(f"- memories: {'; '.join(_t(m2['text'])[:80] for m2 in q.get('memories', [])[:3]) or '(none)'}")
        lines.append(f"- answer: {_t((q.get('answer') or '')[:150])} - judge: {_t(q.get('judge_raw'))}"
                     f" - R5 judged directly: {'correct' if q.get('judge_r5') else 'wrong'} "
                     f"{_t(q.get('judge_r5_raw'))}")
        lines.append("")
    lines.append("> The break point is the first stage at which the gold evidence chain broke; "
                 "for detail see the trace JSON and the HTML report in the same directory, where "
                 "each question's pipeline can be expanded stage by stage.")
    p = Path(out_path)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


if __name__ == "__main__":
    # Regenerate a report from any trace.json, so a code change after a
    # benchmark run does not force the run to be repeated:
    #   python -m scripts.bench.report data/bench/conv26.trace.json [output prefix]
    import sys
    t = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1]).with_suffix("")
    print(render(t, Path(str(prefix) + ".report.html")))
    print(render_md(t, Path(str(prefix) + ".report.md")))
