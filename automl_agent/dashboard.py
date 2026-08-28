from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .errors import AgentError
from .reporting import list_runs, load_run, markdown_report, run_detail


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Autonomous ML Research Agent</title>
<style>
:root{color-scheme:dark;--bg:#0a0e17;--panel:#121927;--line:#273247;--text:#e8edf6;--muted:#94a3b8;--cyan:#22d3ee;--green:#34d399;--red:#fb7185;--amber:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#11213c 0,var(--bg) 38%);font:14px Inter,ui-sans-serif,system-ui;color:var(--text)}
header{padding:28px 32px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:18px;align-items:center}h1{font-size:23px;margin:0}header p{margin:5px 0 0;color:var(--muted)}
main{padding:24px 32px;max-width:1500px;margin:auto}.toolbar{display:flex;gap:12px;align-items:center;margin-bottom:18px}select,button{background:var(--panel);border:1px solid var(--line);color:var(--text);padding:10px 13px;border-radius:9px}button{cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:12px}.card,.panel{background:rgba(18,25,39,.94);border:1px solid var(--line);border-radius:12px}.card{padding:16px}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}.value{font-size:21px;font-weight:700;margin-top:7px}.positive{color:var(--green)}.negative{color:var(--red)}
.columns{display:grid;grid-template-columns:1fr 1.6fr;gap:16px;margin-top:16px}.panel{padding:18px;overflow:auto}h2{font-size:15px;margin:0 0 14px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-size:12px}.chip{padding:4px 8px;border-radius:99px;background:#253149;font-size:12px}.chip.promoted,.chip.completed{background:#123c34;color:#6ee7b7}.chip.failed,.chip.rejected,.chip.timed_out{background:#47202b;color:#fda4af}
.events{max-height:310px;overflow:auto}.event{padding:9px 0;border-bottom:1px solid var(--line)}.event small{color:var(--muted);display:block;margin-top:3px}.notice{color:var(--amber)}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}.columns{grid-template-columns:1fr}header,main{padding-left:16px;padding-right:16px}}
</style></head><body>
<header><div><h1>Autonomous ML Research Agent</h1><p>Evidence → hypothesis → experiment → promotion → convergence</p></div><div id="profile" class="chip">loading</div></header>
<main><div class="toolbar"><select id="runs"></select><button id="refresh">Refresh</button><span id="message" class="notice"></span></div>
<section class="grid" id="cards"></section>
<section class="columns"><div class="panel"><h2>Metric movement</h2><table><thead><tr><th>Metric</th><th>Baseline</th><th>Final</th><th>Δ</th></tr></thead><tbody id="metrics"></tbody></table></div>
<div class="panel"><h2>Experiment trajectory</h2><table><thead><tr><th>#</th><th>Experiment</th><th>Family</th><th>Status</th><th>Score</th></tr></thead><tbody id="experiments"></tbody></table></div></section>
<section class="columns"><div class="panel"><h2>Resource usage</h2><div id="resources"></div></div><div class="panel"><h2>Recent events</h2><div id="events" class="events"></div></div></section></main>
<script>
const fmt=(v,n=4)=>v==null?'—':typeof v==='number'?v.toFixed(n):v; const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path){const r=await fetch(path);if(!r.ok)throw new Error(await r.text());return r.json()}
async function loadRuns(){const rows=await api('/api/runs');const sel=document.querySelector('#runs');const old=sel.value;sel.innerHTML=rows.map(r=>`<option value="${esc(r.run_id)}">${esc(r.run_id)} · ${esc(r.status)}</option>`).join('');if(rows.some(r=>r.run_id===old))sel.value=old;if(rows.length)await loadRun(sel.value);else document.querySelector('#message').textContent='No run artifacts found.'}
async function loadRun(id){const d=await api('/api/runs/'+encodeURIComponent(id));document.querySelector('#profile').textContent=d.profile+(d.profile==='competition'?'':' · non-competition');const delta=d.selection_delta;const cards=[['Status',d.status],['Baseline',fmt(d.baseline_score)],['Final',fmt(d.final_score)],['Selection Δ',fmt(delta),delta>=0?'positive':'negative'],['Iterations',d.iterations_used],['Stop',d.stop_reason]];document.querySelector('#cards').innerHTML=cards.map(c=>`<div class="card"><div class="label">${esc(c[0])}</div><div class="value ${c[2]||''}">${esc(c[1])}</div></div>`).join('');document.querySelector('#metrics').innerHTML=d.metrics.map(m=>`<tr><td>${esc(m.name)}</td><td>${fmt(m.baseline,6)}</td><td>${fmt(m.final,6)}</td><td class="${m.delta>=0?'positive':'negative'}">${m.delta>=0?'+':''}${fmt(m.delta,6)}</td></tr>`).join('');document.querySelector('#experiments').innerHTML=d.experiments.map(x=>`<tr><td>${x.iteration}</td><td>${esc(x.experiment_id)}</td><td>${esc(x.family)}</td><td><span class="chip ${esc(x.status)}">${esc(x.status)}</span></td><td>${fmt(x.selection_score,6)}</td></tr>`).join('');const r=d.resources||{};document.querySelector('#resources').innerHTML=`<p><span class="label">Command seconds</span><br><b>${fmt(r.command_seconds,2)}</b></p><p><span class="label">GPU hours</span><br><b>${fmt(r.gpu_hours,3)}</b></p><p><span class="label">LLM tokens</span><br><b>${fmt(r.llm_tokens,0)}</b></p><p><span class="label">Manual interventions</span><br><b>${d.manual_interventions}</b></p>`;document.querySelector('#events').innerHTML=[...d.events].reverse().slice(0,20).map(e=>`<div class="event"><b>${esc(e.event)}</b><small>${esc(e.at)}</small></div>`).join('')}
document.querySelector('#runs').addEventListener('change',e=>loadRun(e.target.value).catch(show));document.querySelector('#refresh').addEventListener('click',()=>loadRuns().catch(show));function show(e){document.querySelector('#message').textContent=e.message}loadRuns().catch(show);setInterval(()=>{const id=document.querySelector('#runs').value;if(id)loadRun(id).catch(()=>{})},5000);
</script></body></html>"""


def handler_for(run_root: Path) -> type[BaseHTTPRequestHandler]:
    root = run_root.resolve()

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "AutoMLDashboard/0.1"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(value, sort_keys=True).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/":
                    self._send(HTTPStatus.OK, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if path == "/healthz":
                    self._json({"status": "ok"})
                    return
                if path == "/api/runs":
                    self._json(list_runs(root))
                    return
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                    run_id = parts[2]
                    state, events = load_run(root, run_id)
                    if len(parts) == 3:
                        self._json(run_detail(state, events))
                        return
                    if len(parts) == 4 and parts[3] == "report":
                        self._send(
                            HTTPStatus.OK,
                            markdown_report(state, events).encode("utf-8"),
                            "text/markdown; charset=utf-8",
                        )
                        return
                self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            except AgentError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: object) -> None:
            print(f"dashboard {self.address_string()} {format % args}")

    return DashboardHandler


def make_server(run_root: Path, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_for(run_root))


def serve(run_root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = make_server(run_root, host, port)
    address, actual_port = server.server_address[:2]
    print(f"Dashboard: http://{address}:{actual_port} (run root: {run_root.resolve()})")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

