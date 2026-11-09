<div align="center">

```
  ___                      _
 |_ _|___ ___ _   _  ___  | |    ___ _ __  ___
  | |/ __/ __| | | |/ _ \ | |   / _ \ '_ \/ __|
  | |\__ \__ \ |_| |  __/ | |__|  __/ | | \__ \
 |___|___/___/\__,_|\___| |_____\___|_| |_|___/
```

# 🔎 IssueLens

### The zero-dependency AI that triages your GitHub issues in seconds

*Labels · Duplicates · Priority · Spam · Missing info · Draft replies — 100% offline, nothing to install.*

![Python](https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Offline](https://img.shields.io/badge/works-offline-8A2BE2)
![Tests](https://img.shields.io/badge/self--test-30%2F30-success)

[**Quick start**](#-quick-start) · [**Features**](#-features) · [**How it works**](#-how-it-works) · [**Web dashboard**](#-web-dashboard) · [**FAQ**](#-faq)

</div>

---

## 😩 The problem

Maintainers of open-source projects drown in issues. Every day they must:

- 🏷️ label dozens of issues by hand
- 🔁 notice that today's bug is a duplicate of one filed last month
- 🕳️ chase reporters for missing reproduction steps
- 🗑️ sift out spam and `+1` comments
- 🚨 spot the one security report hiding in the pile

This is one of the biggest causes of **maintainer burnout**, and existing "AI triage" tools need API keys, paid cloud services, or a pile of dependencies.

## ✨ The solution

**IssueLens** is **two Python files** and **no dependencies**. Download, run, done. The AI is a self-contained NLP engine (Naive Bayes + TF-IDF) embedded in the code — no API keys, no model downloads, no data leaving your machine.

```bash
python issuelens.py demo
```

## 🚀 Quick start

You only need **Python 3.8+**. No `pip install`. No virtualenv. No config.

```bash
git clone https://github.com/theinsidecoder/issuelens.git
cd issuelens

python issuelens.py selftest                 # verify everything works (30 checks)
python issuelens.py demo                     # offline demo
python issuelens.py analyze --repo owner/name
python issuelens.py serve                    # local web dashboard
```

> Prefer not to clone? Download just `issuelens.py` and `issuelens_brain.py` into one folder.

## 🧰 Features

| | Feature | What you get |
|---|---|---|
| 🏷️ | **Smart labelling** | `bug`, `feature`, `question`, `documentation`, `security`, `performance`, `build` — with confidence scores, multi-label aware |
| 🔁 | **Duplicate detection** | TF-IDF similarity + overlap scoring finds paraphrased duplicates and groups them into clusters |
| 🚨 | **Priority scoring** | P0–P3 with human-readable reasons (crash language, regressions, security, reactions, comments, urgency) |
| 🕳️ | **Missing-info detector** | Flags absent repro steps, versions, environment, logs, or feature motivation |
| 🗑️ | **Spam & noise filter** | Catches link-spam, shouting, `+1` / `bump` comments |
| 😤 | **Tone detection** | Spots frustrated reporters so replies can be kinder |
| 🌱 | **Good-first-issue finder** | Surfaces typo / broken-link style tasks for new contributors |
| ✍️ | **Draft replies** | Ready-to-edit responses for every issue |
| 🧠 | **Learns your repo** | `--learn` fine-tunes the model on your existing labels |
| 📊 | **4 report formats** | Terminal, JSON, Markdown, self-contained HTML |
| 🌐 | **Web dashboard** | Local-only UI, no external assets |
| 🔒 | **Read-only & private** | Never writes to GitHub. Nothing is sent anywhere except the GitHub API fetch you ask for |

## 🖥️ Example output

Real output of `python issuelens.py demo --top 6`:

```text
IssueLens 1.0.0 - triaged 16 issues
------------------------------------------------------------------------
Priority : P0 2  P1 3  P2 3  P3 8
Types    : bug 5  question 3  feature 2  security 2  documentation 2  performance 1  build 1
Findings : 2 duplicates in 2 clusters | 1 spam | 1 noise | 2 need info | 2 good-first-issues
Saved    : ~80 minutes of manual triage (est. 5 min/issue)
------------------------------------------------------------------------
P0 100  #111   Terrible regression: export is totally broken in producti…
            labels: bug 97% | quality 45/100 | needs info
P0  83  #105   SQL injection in the search endpoint
            labels: security 100% | quality 100/100
P1  66  #101   App crashes with segfault when opening large file
            labels: bug 100% | quality 100/100
P1  65  #114   API token printed in debug logs
            labels: security 99% | quality 100/100
P1  50  #102   Crash on opening big file
            labels: bug 100% | quality 35/100 | dup of #101 (65%), needs info
P2  34  #107   Memory leak: RAM grows until the process is killed
            labels: performance 100% | quality 70/100
```

## 📖 Usage

```bash
# Triage a live repo (set GITHUB_TOKEN to avoid rate limits / read private repos)
python issuelens.py analyze --repo pallets/flask --limit 300 --learn

# Triage an exported JSON file (or pipe from stdin with --file -)
python issuelens.py analyze --file issues.json --format md --out triage.md

# Beautiful shareable HTML report
python issuelens.py analyze --repo owner/name --format html --out report.html

# Check one issue before you even file it
python issuelens.py check "App crashes on start" "It crashes since the upgrade"

# Show reasons, missing info and draft replies
python issuelens.py demo -v
```

| Option | Description |
|---|---|
| `--repo owner/name` | GitHub repository (URL also accepted) |
| `--file path` | JSON file with issues (`-` = stdin). Accepts GitHub API format or plain `{"title","body"}` |
| `--token` | GitHub token (or env `GITHUB_TOKEN`) |
| `--limit N` | Max issues to fetch (default 200, max 1000) |
| `--state` | `open`, `closed`, or `all` |
| `--learn` | Fine-tune on the repo's existing labels |
| `--threshold` | Duplicate sensitivity `0–1` (default `0.45`, lower = more matches) |
| `--format` | `text`, `json`, `md`, `html` |
| `--sort` / `--top` | Order by `priority` or `number`; limit rows |
| `-v` | Verbose: reasons, missing info, draft replies |

## 🌐 Web dashboard

```bash
python issuelens.py serve            # http://127.0.0.1:8765
python issuelens.py serve --port 9000 --no-open
```

Analyse a repo, paste JSON, or run the demo — in a clean light/dark UI. The server binds to `127.0.0.1` only.

## 🧠 How it works

```
 issues ─► tokenizer + stemmer ─┬─► Naive Bayes classifier ──► labels + confidence
 (GitHub                        │     (pre-trained on embedded seed corpus,
  or JSON)                      │      optionally fine-tuned with --learn)
                                ├─► TF-IDF + overlap similarity ──► duplicates & clusters
                                ├─► pattern + lexicon rules ──► spam · tone · quality · urgency
                                └─► transparent scoring ──► P0–P3 priority + reasons + reply
```

- **No black box.** Every priority score comes with its reasons; every rule is readable in `issuelens_brain.py`.
- **No network for the AI.** The only network call is the optional GitHub fetch.
- **Deterministic.** Same input, same output.

## 📁 Project layout

```
issuelens/
├── issuelens.py          # CLI, GitHub fetcher, reports, web dashboard, self-test
├── issuelens_brain.py    # AI engine + embedded knowledge & demo data
├── README.md
├── LICENSE               # MIT
└── .gitignore
```

## ⚠️ Honest limitations

- It is a **fast statistical assistant, not a replacement for judgement**. Always review suggestions.
- The built-in model knows 7 common categories; use `--learn` so it adapts to your project's vocabulary.
- Language support is English-first.
- Duplicate detection is lexical (word-based), so heavily reworded duplicates may be missed — tune `--threshold`.
- Read-only by design: it reports and drafts, it does not label or comment for you.

## ❓ FAQ

**Does it need an API key or internet?** No. Only fetching from GitHub needs internet; `--file`, `demo`, `check` and `serve` work fully offline.

**Does it upload my issues anywhere?** No.

**Which Python versions?** 3.8 and newer, on Windows, macOS and Linux.

**How do I know it works on my machine?** Run `python issuelens.py selftest`.

**Can I use it as a library?**
```python
from issuelens_brain import Triager, coerce_issues
results, summary = Triager().analyze(coerce_issues(my_issues))
```

## 🤝 Contributing

Ideas welcome: more seed examples, more languages, new heuristics. Keep the golden rule — **zero dependencies**. Please run `python issuelens.py selftest` before opening a PR.

## 👤 Author

Created and maintained by **[theinsidecoder](https://github.com/theinsidecoder)**.

## 📜 License

[MIT](LICENSE) — free for personal and commercial use.

<div align="center">

**If IssueLens saves you time, give it a ⭐ — it helps other maintainers find it.**

</div>
