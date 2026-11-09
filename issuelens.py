#!/usr/bin/env python3
"""
IssueLens - zero-dependency AI issue triage for open-source maintainers.

Usage:
    python issuelens.py demo                         # offline demo, nothing to set up
    python issuelens.py analyze --repo owner/name    # triage a live GitHub repo
    python issuelens.py analyze --file issues.json   # triage an exported JSON file
    python issuelens.py check "Title" "Body text"    # triage one issue
    python issuelens.py serve                        # local web dashboard
    python issuelens.py selftest                     # verify everything works

Only the Python standard library is used (Python 3.8+).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import issuelens_brain as brain
except ImportError:  # pragma: no cover
    sys.stderr.write("issuelens_brain.py must sit next to issuelens.py\n")
    raise SystemExit(2)

VERSION = brain.VERSION
LOGO = r"""
  ___                      _
 |_ _|___ ___ _   _  ___  | |    ___ _ __  ___
  | |/ __/ __| | | |/ _ \ | |   / _ \ '_ \/ __|
  | |\__ \__ \ |_| |  __/ | |__|  __/ | | \__ \
 |___|___/___/\__,_|\___| |_____\___|_| |_|___/
"""


class IssueLensError(Exception):
    """A user-facing error (bad input, network problem...)."""


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #

def parse_repo(value: str) -> str:
    v = (value or "").strip()
    v = re.sub(r"^https?://(www\.)?github\.com/", "", v).strip("/")
    v = re.sub(r"\.git$", "", v)
    v = "/".join(v.split("/")[:2])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", v):
        raise IssueLensError("Repository must look like 'owner/name' (got %r)." % value)
    return v


def fetch_github_issues(repo: str, token: str = None, limit: int = 200, state: str = "open") -> list:
    repo = parse_repo(repo)
    limit = max(1, min(int(limit), 1000))
    token = token or os.environ.get("GITHUB_TOKEN")
    out, page = [], 1
    while len(out) < limit and page <= 20:
        url = ("https://api.github.com/repos/%s/issues?state=%s&per_page=100&page=%d"
               "&sort=created&direction=desc" % (repo, state, page))
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "issuelens/" + VERSION}
        if token:
            headers["Authorization"] = "Bearer " + token
        try:
            with urlopen(Request(url, headers=headers), timeout=25) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 404:
                raise IssueLensError("Repository '%s' not found (or it is private - pass --token)." % repo)
            if e.code in (401, 403):
                raise IssueLensError("GitHub refused the request (%d). You may be rate-limited: set "
                                     "GITHUB_TOKEN or pass --token." % e.code)
            raise IssueLensError("GitHub API error %d." % e.code)
        except URLError as e:
            raise IssueLensError("Cannot reach GitHub (%s). Check your connection, or use --file." % e.reason)
        except (ValueError, UnicodeDecodeError):
            raise IssueLensError("GitHub returned an unreadable response.")
        if not isinstance(payload, list) or not payload:
            break
        out.extend(p for p in payload if isinstance(p, dict) and "pull_request" not in p)
        if len(payload) < 100:
            break
        page += 1
    return out[:limit]


def run_analysis(raw_issues, learn: bool = False, threshold: float = 0.45):
    issues = brain.coerce_issues(raw_issues)
    if not issues:
        raise IssueLensError("No issues found to analyse.")
    triager = brain.Triager()
    if learn:
        triager.learn(issues)
    return triager.analyze(issues, dup_threshold=threshold)


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

class Style:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code, s):
        return "\033[%sm%s\033[0m" % (code, s) if self.on else s

    def bold(self, s): return self._w("1", s)
    def dim(self, s): return self._w("2", s)
    def red(self, s): return self._w("31", s)
    def green(self, s): return self._w("32", s)
    def yellow(self, s): return self._w("33", s)
    def blue(self, s): return self._w("34", s)
    def magenta(self, s): return self._w("35", s)
    def cyan(self, s): return self._w("36", s)

    def priority(self, p):
        return {"P0": self.red, "P1": self.yellow, "P2": self.blue}.get(p, self.dim)(p)


def _trunc(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _sorted(results, sort):
    if sort == "number":
        return sorted(results, key=lambda r: r["number"])
    return sorted(results, key=lambda r: (-r["priority_score"], r["number"]))


def _flags(r):
    f = []
    if r["spam"]: f.append("SPAM")
    if r["noise"]: f.append("NOISE")
    if r["duplicate_of"]: f.append("dup of #%d (%d%%)" % (r["duplicate_of"]["number"],
                                                         round(r["duplicate_of"]["similarity"] * 100)))
    if r["needs_info"]: f.append("needs info")
    if r["good_first_issue"]: f.append("good first issue")
    return f


def render_text(results, summary, color=True, verbose=False, sort="priority", top=None) -> str:
    c = Style(color)
    L = []
    s = summary
    L.append(c.bold(c.cyan("IssueLens %s" % VERSION)) + " - triaged %d issues" % s["total"])
    L.append(c.dim("-" * 72))
    L.append("Priority : " + "  ".join("%s %d" % (c.priority(p), n) for p, n in s["by_priority"].items()))
    L.append("Types    : " + "  ".join("%s %d" % (k, v) for k, v in s["by_label"].items()))
    L.append("Findings : %d duplicates in %d clusters | %d spam | %d noise | %d need info | %d good-first-issues"
             % (s["duplicates"], s["duplicate_clusters"], s["spam"], s["noise"], s["needs_info"],
                s["good_first_issues"]))
    if s["learned_examples"]:
        L.append("Learning : fine-tuned on %d of this repo's labelled issues" % s["learned_examples"])
    L.append(c.green("Saved    : ~%d minutes of manual triage (est. 5 min/issue)" % s["minutes_saved"]))
    L.append(c.dim("-" * 72))
    rows = _sorted(results, sort)
    if top:
        rows = rows[: int(top)]
    for r in rows:
        labs = ", ".join("%s %d%%" % (l["name"], round(l["confidence"] * 100)) for l in r["labels"])
        L.append("%s %3d  #%-5d %s" % (c.priority(r["priority"]), r["priority_score"], r["number"],
                                       c.bold(_trunc(r["title"], 58))))
        meta = "labels: %s | quality %d/100" % (labs, r["quality"])
        flags = _flags(r)
        if flags:
            meta += " | " + c.magenta(", ".join(flags))
        L.append("            " + c.dim(meta))
        if verbose:
            if r["reasons"]:
                L.append("            why: " + "; ".join(r["reasons"]))
            if r["missing"]:
                L.append("            missing: " + "; ".join(r["missing"]))
            L.append("            suggested labels: " + ", ".join(r["suggested_labels"]))
            L.append(c.dim("            --- draft reply ---"))
            for line in r["reply"].splitlines():
                L.append("            " + line)
            L.append("")
    return "\n".join(L)


def render_json(results, summary) -> str:
    return json.dumps({"tool": "issuelens", "version": VERSION, "summary": summary, "results": results},
                      indent=2, ensure_ascii=False)


def render_markdown(results, summary, sort="priority") -> str:
    s = summary
    L = ["# IssueLens triage report", "",
         "**%d issues** | %d duplicates | %d spam | %d need info | ~%d minutes saved" %
         (s["total"], s["duplicates"], s["spam"], s["needs_info"], s["minutes_saved"]), "",
         "| Priority | Count |", "|---|---|"]
    L += ["| %s | %d |" % (p, n) for p, n in s["by_priority"].items()]
    L += ["", "## Issues", ""]
    for r in _sorted(results, sort):
        labs = ", ".join("`%s`" % l["name"] for l in r["labels"])
        flags = _flags(r)
        L.append("- [ ] **%s** (%d) #%d - %s - %s%s" % (
            r["priority"], r["priority_score"], r["number"], r["title"].replace("|", "\\|"), labs,
            (" - _" + ", ".join(flags) + "_") if flags else ""))
        if r["missing"]:
            L.append("  - missing: " + "; ".join(r["missing"]))
    return "\n".join(L) + "\n"


HTML_STYLE = """
:root{--bg:#f6f8fb;--card:#fff;--fg:#1b2430;--mut:#667085;--bd:#e3e8ef;--acc:#4f46e5}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--mut:#8b949e;--bd:#30363d;--acc:#8b8bff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:24px 16px 60px}h1{margin:0 0 4px;font-size:26px}
.sub{color:var(--mut);margin-bottom:20px}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:22px}
.stat{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:12px}.stat b{display:block;font-size:22px}
.stat span{color:var(--mut);font-size:12px}.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px 16px;margin-bottom:10px}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.title{font-weight:600;flex:1;min-width:200px}
.pill{border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600;border:1px solid var(--bd)}
.P0{background:#fee2e2;color:#b91c1c}.P1{background:#ffedd5;color:#c2410c}.P2{background:#dbeafe;color:#1d4ed8}.P3{background:#f1f5f9;color:#475569}
.tag{background:color-mix(in srgb,var(--acc) 14%,transparent);color:var(--acc)}.warn{background:#fef9c3;color:#854d0e}
.meta{color:var(--mut);font-size:13px;margin-top:6px}details{margin-top:8px}summary{cursor:pointer;color:var(--acc);font-size:13px}
pre{white-space:pre-wrap;background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:10px;font-size:13px}
"""


def render_html(results, summary, sort="priority", title="IssueLens report") -> str:
    e = html.escape
    s = summary
    stats = [(s["total"], "issues"), (s["by_priority"]["P0"], "P0 critical"), (s["by_priority"]["P1"], "P1 high"),
             (s["duplicates"], "duplicates"), (s["spam"], "spam"), (s["needs_info"], "need info"),
             (s["good_first_issues"], "good first issues"), ("~%dm" % s["minutes_saved"], "time saved")]
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             "<meta name='viewport' content='width=device-width,initial-scale=1'>"
             "<title>%s</title><style>%s</style></head><body><div class='wrap'>" % (e(title), HTML_STYLE),
             "<h1>\U0001F50E %s</h1><div class='sub'>Generated by IssueLens %s - 100%% local, zero dependencies"
             "</div><div class='stats'>" % (e(title), VERSION)]
    parts += ["<div class='stat'><b>%s</b><span>%s</span></div>" % (e(str(v)), e(k)) for v, k in stats]
    parts.append("</div>")
    for r in _sorted(results, sort):
        pills = "".join("<span class='pill tag'>%s %d%%</span>" % (e(l["name"]), round(l["confidence"] * 100))
                        for l in r["labels"])
        pills += "".join("<span class='pill warn'>%s</span>" % e(f) for f in _flags(r))
        link = ("<a href='%s'>#%d</a>" % (e(r["url"], quote=True), r["number"])
                if r["url"].startswith("http") else "#%d" % r["number"])
        parts.append("<div class='card'><div class='top'><span class='pill %s'>%s &middot; %d</span>"
                     "<span class='title'>%s %s</span></div><div class='meta'>%s quality %d/100%s</div>"
                     % (r["priority"], r["priority"], r["priority_score"], link, e(r["title"]), pills,
                        r["quality"], (" &middot; " + e("; ".join(r["reasons"]))) if r["reasons"] else ""))
        if r["missing"]:
            parts.append("<div class='meta'>Missing: %s</div>" % e("; ".join(r["missing"])))
        parts.append("<details><summary>Draft reply</summary><pre>%s</pre></details></div>" % e(r["reply"]))
    parts.append("</div></body></html>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Web dashboard (local only)
# --------------------------------------------------------------------------- #

UI_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>IssueLens</title><style>
""" + HTML_STYLE + r"""
input,textarea,button{font:inherit;border-radius:8px;border:1px solid var(--bd);padding:8px 10px;background:var(--card);color:var(--fg)}
textarea{width:100%;min-height:110px}button{cursor:pointer;background:var(--acc);color:#fff;border:0;font-weight:600}
button.alt{background:var(--card);color:var(--fg);border:1px solid var(--bd)}.row{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.row input{flex:1;min-width:150px}.err{color:#b91c1c;margin:8px 0}
</style></head><body><div class="wrap">
<h1>&#128270; IssueLens</h1><div class="sub">Zero-dependency AI issue triage. Runs 100% on your machine.</div>
<div class="card"><b>Triage a GitHub repo</b>
<div class="row"><input id="repo" placeholder="owner/name  (e.g. psf/requests)"><input id="token" type="password" placeholder="GitHub token (optional)">
<input id="limit" type="number" value="100" min="1" max="1000" style="max-width:90px"></div>
<label><input type="checkbox" id="learn"> Learn from this repo's labels</label>
<div class="row"><button onclick="go({repo:v('repo'),token:v('token'),limit:+v('limit'),learn:c('learn')})">Analyse repo</button>
<button class="alt" onclick="go({demo:true})">Run offline demo</button></div></div>
<div class="card"><b>Or paste issues JSON / a single issue</b>
<textarea id="json" placeholder='[{"title":"App crashes on start","body":"..."}]'></textarea>
<div class="row"><button onclick="goJson()">Analyse JSON</button></div></div>
<div id="err" class="err"></div><div id="out"></div></div>
<script>
const v=id=>document.getElementById(id).value.trim(), c=id=>document.getElementById(id).checked;
function el(t,cls,txt){const x=document.createElement(t);if(cls)x.className=cls;if(txt!==undefined)x.textContent=txt;return x}
function goJson(){let d;try{d=JSON.parse(v('json'))}catch(e){return show('Invalid JSON: '+e.message)}go({issues:d,learn:c('learn')})}
function show(m){document.getElementById('err').textContent=m||''}
async function go(payload){show('Analysing...');document.getElementById('out').replaceChildren();
 try{const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 const d=await r.json();if(!r.ok)throw new Error(d.error||'Request failed');show('');render(d)}catch(e){show(e.message)}}
function render(d){const out=document.getElementById('out'),s=d.summary;out.replaceChildren();
 const st=el('div','stats');[[s.total,'issues'],[s.by_priority.P0,'P0 critical'],[s.by_priority.P1,'P1 high'],[s.duplicates,'duplicates'],
 [s.spam,'spam'],[s.needs_info,'need info'],[s.good_first_issues,'good first issues'],['~'+s.minutes_saved+'m','time saved']].forEach(([n,k])=>{
 const b=el('div','stat');b.append(el('b','',n),el('span','',k));st.append(b)});out.append(st);
 d.results.sort((a,b)=>b.priority_score-a.priority_score).forEach(r=>{const cd=el('div','card'),top=el('div','top');
 top.append(el('span','pill '+r.priority,r.priority+' · '+r.priority_score),el('span','title','#'+r.number+' '+r.title));cd.append(top);
 const m=el('div','meta');r.labels.forEach(l=>m.append(el('span','pill tag',l.name+' '+Math.round(l.confidence*100)+'%'),' '));
 if(r.duplicate_of)m.append(el('span','pill warn','dup of #'+r.duplicate_of.number+' ('+Math.round(r.duplicate_of.similarity*100)+'%)'),' ');
 if(r.spam)m.append(el('span','pill warn','SPAM'),' ');if(r.needs_info)m.append(el('span','pill warn','needs info'),' ');
 if(r.good_first_issue)m.append(el('span','pill warn','good first issue'),' ');m.append('quality '+r.quality+'/100');cd.append(m);
 if(r.reasons.length)cd.append(el('div','meta','Why: '+r.reasons.join('; ')));
 if(r.missing.length)cd.append(el('div','meta','Missing: '+r.missing.join('; ')));
 const dt=el('details'),sm=el('summary','','Draft reply');dt.append(sm,el('pre','',r.reply));cd.append(dt);out.append(cd)})}
</script></body></html>"""

MAX_BODY = 8 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "IssueLens/" + VERSION

    def log_message(self, *args):  # keep the console quiet
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, UI_HTML, "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True, "version": VERSION}))
        else:
            self._send(404, json.dumps({"error": "Not found"}))

    def do_POST(self):
        if self.path != "/api/analyze":
            return self._send(404, json.dumps({"error": "Not found"}))
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                raise IssueLensError("Request body missing or too large.")
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(req, dict):
                raise IssueLensError("Request must be a JSON object.")
            if req.get("demo"):
                raw = brain.DEMO_ISSUES
            elif req.get("repo"):
                raw = fetch_github_issues(req["repo"], req.get("token") or None, req.get("limit") or 100)
            elif "issues" in req:
                raw = req["issues"]
            else:
                raise IssueLensError("Provide 'repo', 'issues' or 'demo'.")
            results, summary = run_analysis(raw, learn=bool(req.get("learn")))
            self._send(200, json.dumps({"summary": summary, "results": results}))
        except IssueLensError as e:
            self._send(400, json.dumps({"error": str(e)}))
        except (ValueError, UnicodeDecodeError) as e:
            self._send(400, json.dumps({"error": "Invalid input: %s" % e}))
        except Exception as e:  # never crash the server
            self._send(500, json.dumps({"error": "Internal error: %s" % e}))


def serve(port: int = 8765, open_browser: bool = True):
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        raise IssueLensError("Cannot start server on port %d (%s). Try --port 9000." % (port, e))
    url = "http://127.0.0.1:%d" % httpd.server_address[1]
    print("IssueLens dashboard running at %s  (Ctrl+C to stop)" % url)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def selftest() -> int:
    passed, failed = 0, []

    def check(name, cond):
        nonlocal passed
        if cond:
            passed += 1
            print("  [ok]   " + name)
        else:
            failed.append(name)
            print("  [FAIL] " + name)

    print("IssueLens self-test")
    check("tokenizer stems and drops stop words", brain.tokenize("The app crashes!") == ["app", "crash"])
    check("contractions keep negation", "not" in brain.tokenize("it doesn't work"))
    results, summary = run_analysis(brain.DEMO_ISSUES)
    by = {r["number"]: r for r in results}
    check("analyses all 16 demo issues", summary["total"] == 16)
    check("classifies bug", by[101]["primary"] == "bug")
    check("classifies feature", by[103]["primary"] == "feature")
    check("classifies question", by[104]["primary"] == "question")
    check("classifies security", by[105]["primary"] == "security")
    check("classifies documentation", by[106]["primary"] == "documentation")
    check("classifies performance", by[107]["primary"] == "performance")
    check("classifies build", by[108]["primary"] == "build")
    check("detects crash duplicate", (by[102]["duplicate_of"] or {}).get("number") == 101)
    check("detects feature duplicate", (by[112]["duplicate_of"] or {}).get("number") == 103)
    check("detects spam", by[109]["spam"] and not by[101]["spam"])
    check("detects +1 noise", by[116]["noise"])
    check("flags low-quality bug", by[110]["quality"] < 70 and by[101]["quality"] >= 90)
    check("security ranks above docs", by[105]["priority_score"] > by[106]["priority_score"])
    check("urgent regression is P0", by[111]["priority"] == "P0")
    check("good-first-issue detected", by[106]["good_first_issue"])
    check("frustration detected", by[111]["tone"] == "frustrated")
    trained, _ = run_analysis(brain.DEMO_ISSUES, learn=True)
    check("learn mode runs", len(trained) == 16)
    check("json output parses", json.loads(render_json(results, summary))["summary"]["total"] == 16)
    check("text report renders", "IssueLens" in render_text(results, summary, color=False, verbose=True))
    check("markdown report renders", "# IssueLens triage report" in render_markdown(results, summary))
    xss, _ = run_analysis([{"title": "<script>alert(1)</script>", "body": "x" * 50}])
    check("html report escapes input", "<script>alert" not in render_html(xss, _))
    check("GitHub API shape accepted", len(brain.coerce_issues(
        [{"number": 7, "title": "t", "body": None, "user": {"login": "a"}, "labels": [{"name": "bug"}],
          "reactions": {"total_count": 2}}, {"number": 8, "title": "pr", "pull_request": {}}])) == 1)
    check("repo parsing", parse_repo("https://github.com/psf/requests.git") == "psf/requests")
    try:
        parse_repo("nonsense")
        check("bad repo rejected", False)
    except IssueLensError:
        check("bad repo rejected", True)
    # live local server round-trip
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % httpd.server_address[1]
    try:
        with urlopen(base + "/", timeout=10) as r:
            check("web UI served", b"IssueLens" in r.read())
        req = Request(base + "/api/analyze", data=json.dumps({"demo": True}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as r:
            check("web API analyses", json.loads(r.read())["summary"]["total"] == 16)
        try:
            urlopen(Request(base + "/api/analyze", data=b"{bad", headers={"Content-Type": "application/json"}))
            check("web API rejects bad JSON", False)
        except HTTPError as e:
            check("web API rejects bad JSON", e.code == 400)
    finally:
        httpd.shutdown()
        httpd.server_close()
    print("\n%d passed, %d failed" % (passed, len(failed)))
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _emit(args, results, summary):
    fmt = args.format
    if fmt == "json":
        text = render_json(results, summary)
    elif fmt == "md":
        text = render_markdown(results, summary, args.sort)
    elif fmt == "html":
        out = args.out or "issuelens-report.html"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render_html(results, summary, args.sort))
        print("HTML report written to %s" % os.path.abspath(out))
        return
    else:
        color = sys.stdout.isatty() and not args.out and not os.environ.get("NO_COLOR")
        text = render_text(results, summary, color, args.verbose, args.sort, args.top)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print("Report written to %s" % os.path.abspath(args.out))
    else:
        print(text)


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--format", choices=["text", "json", "md", "html"], default="text")
    common.add_argument("--out", help="write the report to this file")
    common.add_argument("--sort", choices=["priority", "number"], default="priority")
    common.add_argument("--top", type=int, help="only show the top N issues (text format)")
    common.add_argument("-v", "--verbose", action="store_true", help="show reasons and draft replies")
    common.add_argument("--learn", action="store_true", help="fine-tune on the repo's existing labels")
    common.add_argument("--threshold", type=float, default=0.45, help="duplicate sensitivity 0-1 (default 0.45)")
    p = argparse.ArgumentParser(prog="issuelens", description="Zero-dependency AI issue triage.")
    p.add_argument("--version", action="version", version="IssueLens " + VERSION)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("demo", parents=[common], help="run on built-in sample issues (offline)")
    a = sub.add_parser("analyze", parents=[common], help="triage a repo or a JSON file")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--repo", help="owner/name or GitHub URL")
    g.add_argument("--file", help="JSON file of issues ('-' for stdin)")
    a.add_argument("--token", help="GitHub token (or set GITHUB_TOKEN)")
    a.add_argument("--limit", type=int, default=200, help="max issues to fetch (default 200)")
    a.add_argument("--state", choices=["open", "closed", "all"], default="open")
    c = sub.add_parser("check", parents=[common], help="triage a single issue")
    c.add_argument("title")
    c.add_argument("body", nargs="?", default="")
    s = sub.add_parser("serve", help="start the local web dashboard")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-open", action="store_true", help="do not open the browser")
    sub.add_parser("selftest", help="verify the installation")
    return p


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if os.name == "nt":
        os.system("")  # enable ANSI colours on Windows 10+
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        print(LOGO)
        parser.print_help()
        return 0
    try:
        if args.cmd == "selftest":
            return selftest()
        if args.cmd == "serve":
            serve(args.port, not args.no_open)
            return 0
        if args.cmd == "demo":
            raw = brain.DEMO_ISSUES
        elif args.cmd == "check":
            raw = [{"number": 1, "title": args.title, "body": args.body}]
            args.verbose = True
        elif args.file:
            try:
                fh = sys.stdin if args.file == "-" else open(args.file, "r", encoding="utf-8")
                with fh:
                    raw = json.load(fh)
            except OSError as e:
                raise IssueLensError("Cannot read %s: %s" % (args.file, e.strerror or e))
            except ValueError as e:
                raise IssueLensError("%s is not valid JSON: %s" % (args.file, e))
        else:
            raw = fetch_github_issues(args.repo, args.token, args.limit, args.state)
        results, summary = run_analysis(raw, args.learn, args.threshold)
        _emit(args, results, summary)
        return 0
    except IssueLensError as e:
        sys.stderr.write("error: %s\n" % e)
        return 1
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
