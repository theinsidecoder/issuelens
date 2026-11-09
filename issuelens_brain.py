#!/usr/bin/env python3
"""
issuelens_brain.py - the AI engine of IssueLens.

Everything here is pure Python standard library. There is NOTHING to install and
NOTHING is downloaded: the "model" is a Naive Bayes text classifier pre-trained
on the seed corpus embedded below, combined with a TF-IDF similarity engine and
transparent scoring heuristics. It can also learn from a repository's own
labelled issues, so it adapts to every project.

Public API (used by issuelens.py, but fully usable on its own):

    from issuelens_brain import Triager, coerce_issues
    triager = Triager()
    results, summary = triager.analyze(coerce_issues(list_of_issue_dicts))
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Embedded knowledge: stop words, seed training corpus, lexicons
# --------------------------------------------------------------------------- #

STOPWORDS = frozenset(
    """a an the and or but if then else when while of to in on at by for with from as is
    are was were be been being it its this that these those i me my we our you your he she
    they them their do does did done have has had will would can could should may might just
    also very so than too into over about out up down off again there here what which who
    whom how why all any both each few more most other some such only own same s t don
    doesn didn isn aren wasn weren won wouldn couldn shouldn im ive thanks thank please hi
    hello hey get got one like use used using""".split()
)

# label -> example sentences. Short on purpose: the model is a strong prior that is
# refined by `Triager.learn()` with real, labelled issues from the target repository.
SEED_CORPUS = {
    "bug": [
        "app crashes with segmentation fault when I open a large file",
        "unexpected exception traceback error thrown on startup",
        "this function returns the wrong result, expected 5 but got 4",
        "regression after upgrading, it used to work in the previous version",
        "button does not work, nothing happens when clicked",
        "null pointer exception causes the process to crash",
        "steps to reproduce: run the command and observe the error message",
        "broken behavior, incorrect output, fails with an error",
        "program hangs and freezes and never finishes",
        "unhandled error cannot read property of undefined",
        "data is corrupted and lost after saving the file",
        "test fails intermittently, flaky failure on windows",
        "wrong value displayed, calculation is incorrect",
    ],
    "feature": [
        "it would be great to add support for a new option",
        "feature request: please add the ability to export to csv",
        "proposal: implement a plugin system for extensions",
        "can you add an api to configure custom behavior",
        "I would like to have dark mode support",
        "suggestion: allow users to customize the output format",
        "enhancement: support for a new language and integration",
        "nice to have: an option flag to enable this capability",
        "please consider adding a command to list all items",
        "wishlist: add an ability to import settings",
        "add support for yaml configuration files",
    ],
    "question": [
        "how do I configure this to work with my setup",
        "is it possible to use this with docker, any example",
        "question: what is the difference between these two options",
        "can someone explain how this works, I am confused",
        "help needed, how can I get started",
        "what is the recommended way to do this",
        "does it support multiple users, or is there a way to",
        "not sure whether this is the correct usage, any advice",
        "where can I find information about this setting",
        "how to use this library with another framework",
    ],
    "documentation": [
        "the documentation is outdated and the readme has a typo",
        "docs are missing an example for this function",
        "tutorial link is broken in the guide",
        "please improve the documentation and add a changelog entry",
        "typo in readme, wrong spelling on the docs page",
        "api reference is incomplete, missing docstring description",
        "contributing guide is unclear and needs clarification",
        "docs example code does not match the current api",
        "broken link in the readme",
    ],
    "security": [
        "security vulnerability allows remote code execution",
        "cve reported, exploit possible via injection",
        "xss cross site scripting in the input field",
        "sensitive password token leaked in logs, credentials exposed",
        "authentication bypass lets an attacker gain unauthorized access",
        "sql injection vulnerability in query parameter",
        "insecure default permission, privilege escalation risk",
        "dependency has a known vulnerability, please update to the patched version",
        "api key secret exposed publicly",
    ],
    "performance": [
        "very slow performance when processing many items",
        "high memory usage and memory leak over time",
        "cpu usage spikes to 100 percent, latency is too high",
        "optimize the startup time, it takes minutes to load",
        "benchmark shows regression in speed, throughput dropped",
        "excessive allocation makes garbage collection slow",
        "queries take too long, need caching to speed up",
        "ram consumption grows until the system runs out of memory",
    ],
    "build": [
        "build fails on ci pipeline with a compile error",
        "cannot install package, dependency conflict during install",
        "cmake configure fails, linker error undefined reference",
        "github actions workflow failing, docker image build broken",
        "packaging issue, wheel does not build on macos",
        "npm install error, lockfile conflict with node version",
        "release pipeline broken, tests fail in continuous integration",
        "compiler warning treated as error with newer toolchain",
        "pip install fails with error while building wheel",
    ],
}

NEGATIVE_WORDS = frozenset(
    """ridiculous terrible awful useless unacceptable frustrating frustrated annoying
    garbage worst hate disappointing disappointed waste horrible stupid unusable joke
    sucks pathetic broken nightmare""".split()
)
POSITIVE_WORDS = frozenset(
    """thanks great love awesome appreciate wonderful amazing excellent fantastic helpful
    nice""".split()
)

LABEL_RULES = [  # (label, regex, probability boost) - sharpen the statistical model
    ("security", r"\b(cve-\d{4}-\d+|vulnerab\w*|exploit\w*|xss|csrf|rce|sql injection|"
                 r"privilege escalation|security advisory|credentials? (leak|expos)\w*)\b", 0.60),
    ("bug", r"\b(traceback|stack ?trace|segfault|segmentation fault|exception|crash\w*|panic|"
            r"not work\w*|fails?|failed|error)\b", 0.25),
    ("feature", r"\b(feature request|enhancement|proposal|would be (nice|great)|wishlist|"
               r"nice to have)\b", 0.35),
    ("question", r"\b(how (do|can|to|should)|is it possible|any way to|what is the)\b", 0.20),
    ("documentation", r"\b(typo|readme|docs?|documentation|docstring|tutorial|spelling)\b", 0.20),
    ("performance", r"\b(slow|memory leak|latency|performance|speed up|optimi[sz]\w*)\b", 0.25),
    ("build", r"\b(ci|build fails?|can not install|install(ation)? (error|fail\w*)|compile error|"
             r"linker|cmake|dockerfile)\b", 0.25),
]
LABEL_RULES = [(l, re.compile(r, re.I), b) for l, r, b in LABEL_RULES]

CRASH_RE = re.compile(r"\b(crash\w*|segfault|segmentation fault|data loss|corrupt\w*|panic|deadlock|"
                      r"hang\w*|freez\w*|lost data|overwr\w+)\b")
REGRESSION_RE = re.compile(r"\b(regression|used to work|worked (before|fine|in)|"
                           r"broke(n)? (in|after|since)|after (the )?(upgrad|updat)\w*)\b")
URGENT_RE = re.compile(r"\b(urgent\w*|asap|blocker|blocking|critical|production|outage|immediately|"
                       r"emergency|showstopper|can not (use|deploy|run))\b")
SPAM_RE = re.compile(r"\b(casino|viagra|cheap|buy now|free money|crypto|bitcoin|seo|click here|"
                     r"earn \$?\d+|whatsapp|telegram|promo\w*|discount|giveaway|porn|escort|loan|"
                     r"forex|backlinks?|essay writing|guaranteed)\b")
NOISE_RE = re.compile(r"^(\+1|\+\+|bump|me too|same( here)?|any updates?|up|ping)[\s!.?]*$")
GFI_RE = re.compile(r"\b(typo|spelling|grammar|broken link|wording|trivial|easy fix|small fix|"
                    r"good first issue|beginner|minor)\b")
ENV_RE = re.compile(r"\b(windows|linux|macos|mac os|osx|ubuntu|debian|fedora|android|ios|python|"
                    r"node|java|go|rust|browser|chrome|firefox|safari|docker|arch|x86|arm64)\b")

IGNORED_LABEL_HINTS = ("good first", "wontfix", "won't", "duplicate", "invalid", "stale", "help wanted",
                       "hacktoberfest", "priority", "triage", "needs", "status", "blocked", "wip",
                       "p0", "p1", "p2", "p3", "approved", "released")
LABEL_MAP = [("secur", "security"), ("vuln", "security"), ("perf", "performance"), ("slow", "performance"),
             ("memory", "performance"), ("doc", "documentation"), ("bug", "bug"), ("defect", "bug"),
             ("crash", "bug"), ("regress", "bug"), ("error", "bug"), ("feature", "feature"),
             ("enhanc", "feature"), ("request", "feature"), ("proposal", "feature"),
             ("idea", "feature"), ("question", "question"), ("support", "question"),
             ("howto", "question"), ("discussion", "question"), ("build", "build"), ("ci", "build"),
             ("install", "build"), ("packag", "build"), ("compil", "build")]

# --------------------------------------------------------------------------- #
# Text processing
# --------------------------------------------------------------------------- #

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_URL = re.compile(r"https?://\S+")
_WORD = re.compile(r"[a-z0-9_]+")


def normalize(text: str) -> str:
    """Lower-case and expand contractions so 'doesn't' keeps its negative meaning."""
    t = (text or "").lower().replace("\u2019", "'")
    t = t.replace("won't", "will not").replace("can't", "can not").replace("cannot", "can not")
    return re.sub(r"n't\b", " not", t)


def strip_code(text: str) -> str:
    return _CODE_BLOCK.sub(" ", text or "")


def stem(word: str) -> str:
    """A tiny, fast suffix stripper (good enough for matching 'crashes' ~ 'crashing')."""
    if len(word) <= 3:
        return word
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""),
                         ("ly", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                continue
            return word[: len(word) - len(suffix)] + repl
    return word


def tokenize(text: str, bigrams: bool = False, drop_code: bool = False) -> list:
    t = strip_code(text) if drop_code else (text or "")
    t = _URL.sub(" ", normalize(t))
    words = []
    for w in _WORD.findall(t):
        if len(w) < 2 or w in STOPWORDS or w.isdigit():
            continue
        words.append(stem(w))
    if bigrams:
        words = words + [a + "_" + b for a, b in zip(words, words[1:])]
    return words


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class NaiveBayes:
    """Multinomial Naive Bayes with Laplace smoothing and length-aware calibration."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.class_docs = Counter()
        self.word_counts = defaultdict(Counter)
        self.class_tokens = Counter()
        self.vocab = set()
        self.total_docs = 0.0

    def train(self, tokens: list, label: str, weight: float = 1.0) -> None:
        self.class_docs[label] += weight
        self.total_docs += weight
        for tok in tokens:
            self.word_counts[label][tok] += weight
            self.class_tokens[label] += weight
            self.vocab.add(tok)

    def predict(self, tokens: list) -> list:
        """Return [(label, probability)] sorted by probability, summing to 1."""
        classes = list(self.class_docs)
        if not classes:
            return []
        known = [t for t in tokens if t in self.vocab]
        v = max(len(self.vocab), 1)
        scale = 0.6 / max(1.0, math.sqrt(len(known) / 6.0))
        logs = {}
        for c in classes:
            lp = math.log((self.class_docs[c] + 1.0) / (self.total_docs + len(classes)))
            denom = self.class_tokens[c] + self.alpha * v
            wc = self.word_counts[c]
            for tok in known:
                lp += math.log((wc.get(tok, 0.0) + self.alpha) / denom) * scale
            logs[c] = lp
        peak = max(logs.values())
        exps = {c: math.exp(lp - peak) for c, lp in logs.items()}
        z = sum(exps.values())
        return sorted(((c, e / z) for c, e in exps.items()), key=lambda x: -x[1])


class SimilarityIndex:
    """TF-IDF cosine similarity with an inverted index (fast for thousands of issues)."""

    def __init__(self, docs: list):
        self.n = len(docs)
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log((1 + self.n) / (1 + c)) + 1.0 for t, c in df.items()}
        # Tokens present in most issues are template boilerplate: ignore them.
        self.boiler = {t for t, c in df.items() if self.n >= 8 and c / self.n > 0.4}
        self.vecs = [self._vec(d) for d in docs]
        # unigram sets (+ their IDF mass) used for the overlap coefficient
        self.sets = [set(t for t in d if "_" not in t and t not in self.boiler) for d in docs]
        self.mass = [sum(self.idf[t] for t in st) for st in self.sets]
        self.postings = defaultdict(list)
        for i, vec in enumerate(self.vecs):
            for t, w in vec.items():
                self.postings[t].append((i, w))

    def _vec(self, tokens: list) -> dict:
        vec = {}
        for t, c in Counter(tokens).items():
            if t in self.boiler or t not in self.idf:
                continue
            vec[t] = (1.0 + math.log(c)) * self.idf[t]
        norm = math.sqrt(sum(x * x for x in vec.values())) or 1.0
        return {t: x / norm for t, x in vec.items()}

    def pairs(self, threshold: float) -> list:
        out = []
        for i, vec in enumerate(self.vecs):
            acc = defaultdict(float)
            for t, w in vec.items():
                for j, wj in self.postings[t]:
                    if j > i:
                        acc[j] += w * wj
            for j, cos in acc.items():
                if cos < 0.12:
                    continue
                sim = cos
                if min(len(self.sets[i]), len(self.sets[j])) >= 4:
                    shared = sum(self.idf[t] for t in self.sets[i] & self.sets[j])
                    overlap = shared / (min(self.mass[i], self.mass[j]) or 1.0)
                    sim = 0.5 * cos + 0.5 * overlap  # robust when one issue is much shorter
                if sim >= threshold:
                    out.append((i, j, sim))
        return out


# --------------------------------------------------------------------------- #
# Input normalisation
# --------------------------------------------------------------------------- #

def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_issues(data) -> list:
    """Accept GitHub API JSON or simple {"title","body"} dicts; return clean, sorted issues."""
    if isinstance(data, dict):
        for key in ("issues", "items", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of issues (or an object containing an 'issues' list).")
    out, used = [], set()
    counter = 1
    for raw in data:
        if not isinstance(raw, dict) or "pull_request" in raw:
            continue
        title = str(raw.get("title") or "").strip()
        body = str(raw.get("body") or "")
        if not title and not body.strip():
            continue
        num = raw.get("number")
        if not isinstance(num, int) or num in used:
            while counter in used:
                counter += 1
            num = counter
        used.add(num)
        labels = []
        for lab in raw.get("labels") or []:
            name = lab.get("name") if isinstance(lab, dict) else lab
            if name:
                labels.append(str(name))
        user = raw.get("user")
        author = user.get("login") if isinstance(user, dict) else (raw.get("author") or "")
        react = raw.get("reactions")
        reactions = _int(react.get("total_count")) if isinstance(react, dict) else _int(react)
        out.append({
            "number": num, "title": title or "(no title)", "body": body, "author": str(author or ""),
            "labels": labels, "comments": _int(raw.get("comments")), "reactions": reactions,
            "created_at": str(raw.get("created_at") or ""),
            "url": str(raw.get("html_url") or raw.get("url") or ""),
        })
    out.sort(key=lambda i: i["number"])
    return out


def canonical_label(name: str):
    """Map a repository label such as 'type: bug' to a canonical class (or a custom class)."""
    n = (name or "").lower().strip()
    if not n or any(h in n for h in IGNORED_LABEL_HINTS):
        return None
    for needle, canon in LABEL_MAP:
        if needle in n:
            return canon
    clean = re.sub(r"[^a-z0-9]+", "-", n).strip("-")
    return clean[:30] or None


# --------------------------------------------------------------------------- #
# The Triager
# --------------------------------------------------------------------------- #

class Triager:
    def __init__(self):
        self.nb = NaiveBayes()
        self.learned = 0
        for label, examples in SEED_CORPUS.items():
            for sentence in examples:
                self.nb.train(tokenize(sentence), label, 2.0)

    # -- learning ---------------------------------------------------------- #
    def learn(self, issues: list) -> int:
        """Fine-tune on a repository's own labelled issues. Returns number of examples used."""
        pairs, counts = [], Counter()
        for iss in issues:
            for name in iss.get("labels", []):
                canon = canonical_label(name)
                if canon:
                    pairs.append((iss, canon))
                    counts[canon] += 1
        used = 0
        for iss, canon in pairs:
            if canon not in SEED_CORPUS and counts[canon] < 3:
                continue  # ignore rare custom labels
            toks = tokenize(iss["title"] + " " + iss["title"] + " " + strip_code(iss["body"]))
            self.nb.train(toks, canon, 3.0)
            used += 1
        self.learned += used
        return used

    # -- single signals ---------------------------------------------------- #
    def classify(self, title: str, body: str) -> list:
        text = normalize(title + " \n" + strip_code(body))
        toks = tokenize(title + " " + title + " " + body, drop_code=True)
        probs = dict(self.nb.predict(toks))
        for label, rx, boost in LABEL_RULES:
            if label in probs and rx.search(text):
                probs[label] += boost
        z = sum(probs.values()) or 1.0
        ranked = sorted(((l, p / z) for l, p in probs.items()), key=lambda x: -x[1])
        chosen = [ranked[0]]
        for label, p in ranked[1:3]:
            if p >= 0.28 and p >= 0.5 * ranked[0][1]:
                chosen.append((label, p))
        sec = dict(ranked).get("security", 0.0)
        if sec >= 0.30 and "security" not in [l for l, _ in chosen]:
            chosen.append(("security", sec))
        return [(l, round(p, 3)) for l, p in chosen]

    @staticmethod
    def tone(text: str) -> str:
        words = _WORD.findall(normalize(text))
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        if neg >= 2 or (neg >= 1 and "!!" in text) or "!!!" in text:
            return "frustrated"
        if pos >= 1 and neg == 0:
            return "positive"
        return "neutral"

    @staticmethod
    def spam_score(title: str, body: str):
        text = normalize(title + " " + body)
        score, why = 0, []
        urls = len(_URL.findall(text))
        if urls >= 3:
            score += 40
            why.append("%d links" % urls)
        hits = set(m.group(0) for m in SPAM_RE.finditer(text))
        if hits:
            score += 25 * len(hits)
            why.append("spam words: " + ", ".join(sorted(hits)))
        letters = [c for c in (title + body) if c.isalpha()]
        if len(letters) > 20 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
            score += 20
            why.append("SHOUTING")
        if not body.strip() and len(title.strip()) < 12:
            score += 20
            why.append("empty body")
        return score, why

    @staticmethod
    def assess_quality(title: str, body: str, primary: str):
        low = normalize(body)
        score, missing = 100, []
        if len(body.strip()) < 30:
            score -= 40
            missing.append("A meaningful description")
        if len(title.strip()) < 10:
            score -= 10
            missing.append("A descriptive title")
        checks = []
        if primary in ("bug", "performance", "build"):
            checks = [
                ("Steps to reproduce", r"(steps? to reproduce|reproduc|\brepro\b|\n\s*1[\.\)])", 20),
                ("Expected vs. actual behaviour", r"(expected|should (have|be|return)|instead of)", 15),
                ("Version information", r"(\bv?\d+\.\d+(\.\d+)?\b|\bversion\b)", 10),
                ("Environment (OS / runtime)", ENV_RE.pattern, 10),
                ("Error output or logs", r"(```|traceback|error|exception|stack ?trace)", 10),
            ]
        elif primary == "feature":
            checks = [("Use case / motivation",
                       r"(use case|because|so that|in order to|motivation|problem|currently|"
                       r"would allow|i need|we need)", 25)]
        for name, pattern, cost in checks:
            if not re.search(pattern, low):
                score -= cost
                missing.append(name)
        return max(score, 0), missing

    @staticmethod
    def priority(issue, labels, text, spam, tone):
        if spam:
            return 0, "P3", ["Looks like spam"]
        reasons, s = [], 10.0
        lab = dict(labels)
        if lab.get("security", 0) >= 0.30:
            s += 45
            reasons.append("Possible security impact")
        if CRASH_RE.search(text):
            s += 20
            reasons.append("Crash / hang / data-loss language")
        if REGRESSION_RE.search(text):
            s += 12
            reasons.append("Regression reported")
        if URGENT_RE.search(text):
            s += 15
            reasons.append("Urgency or production impact")
        if issue["comments"]:
            s += min(12, issue["comments"] * 1.5)
            reasons.append("%d comments" % issue["comments"])
        if issue["reactions"]:
            s += min(12, issue["reactions"] * 1.0)
            reasons.append("%d reactions" % issue["reactions"])
        base = {"bug": 18, "security": 10, "performance": 15, "build": 15, "feature": 0,
                "documentation": -5, "question": -8}.get(labels[0][0], 0)
        s += base
        if tone == "frustrated":
            s += 5
            reasons.append("Frustrated reporter")
        s = int(max(0, min(100, round(s))))
        level = "P0" if s >= 70 else "P1" if s >= 45 else "P2" if s >= 25 else "P3"
        return s, level, reasons

    # -- reply drafting ---------------------------------------------------- #
    @staticmethod
    def draft_reply(res: dict) -> str:
        if res["spam"]:
            return "(No reply recommended. Close as spam and consider blocking the account.)"
        if res["noise"]:
            return ("Thanks! To keep the thread focused, please use a \U0001F44D reaction instead of "
                    "a comment, and share any new information you have.")
        who = ("@" + res["author"]) if res["author"] else "there"
        out = ["Hi %s, thanks for taking the time to open this." % who]
        if res["tone"] == "frustrated":
            out.append("Sorry for the trouble - we understand how annoying this is and want to help.")
        primary = res["primary"]
        if primary == "security":
            out.append("This may be security-sensitive. Please avoid sharing exploit details publicly and "
                       "report it privately through the repository's security policy / private advisory.")
        elif primary == "documentation":
            out.append("Good catch. Documentation fixes are very welcome - a pull request would be "
                       "wonderful if you would like to make the change.")
        elif primary == "question":
            out.append("This looks like a usage question. Please check the docs and existing discussions; "
                       "if you already tried something, tell us what and what you saw.")
        elif primary == "feature":
            out.append("Thanks for the suggestion. We'll evaluate it against the project's goals.")
        else:
            out.append("We'll take a look.")
        if res["missing"]:
            out.append("To help us act on this quickly, could you add:\n"
                       + "\n".join("  - " + m for m in res["missing"]))
        if res["duplicate_of"]:
            d = res["duplicate_of"]
            out.append("This looks similar to #%d (%d%% match). Could you check whether it describes the "
                       "same problem? If so, we'll close this one and continue there."
                       % (d["number"], round(d["similarity"] * 100)))
        if res["good_first_issue"]:
            out.append("This looks like a great good-first-issue; contributors are welcome to pick it up!")
        return "\n\n".join(out)

    # -- full pipeline ----------------------------------------------------- #
    def analyze(self, issues: list, dup_threshold: float = 0.45):
        """Analyse issues. Returns (results, summary). `issues` must come from coerce_issues()."""
        full_docs = [tokenize(i["title"] + " " + i["title"] + " " + i["body"], bigrams=True) for i in issues]
        index = SimilarityIndex(full_docs)
        dups = defaultdict(list)     # j -> [(i, sim)] earlier similar issues
        related = defaultdict(list)  # i -> [(j, sim)]
        parent = list(range(len(issues)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j, sim in index.pairs(dup_threshold):
            dups[j].append((i, sim))
            related[i].append((j, sim))
            parent[find(j)] = find(i)

        results = []
        for idx, iss in enumerate(issues):
            title, body = iss["title"], iss["body"]
            text = normalize(title + " \n" + strip_code(body))
            labels = self.classify(title, body)
            primary = labels[0][0]
            spam_pts, spam_why = self.spam_score(title, body)
            spam = spam_pts >= 50
            noise = bool(NOISE_RE.match(normalize((title + " " + body).strip()).strip()))
            tone = self.tone(title + " " + body)
            quality, missing = self.assess_quality(title, body, primary)
            if spam:
                missing = []
            s, level, reasons = self.priority(iss, labels, text, spam or noise, tone)
            if noise:
                reasons = ["Low-signal comment (+1 / bump)"]
            sims = sorted(dups.get(idx, []), key=lambda x: -x[1])
            dup_of = ({"number": issues[sims[0][0]]["number"], "similarity": round(sims[0][1], 3)}
                      if sims and not spam else None)
            sim_list = sorted([(issues[j]["number"], sm) for j, sm in related.get(idx, [])] +
                              [(issues[j]["number"], sm) for j, sm in dups.get(idx, [])],
                              key=lambda x: -x[1])[:3]
            gfi = bool(GFI_RE.search(text)) and primary != "security" and not spam and not noise
            needs_info = bool(missing) and quality < 60 and not spam and not noise
            suggested = [l for l, _ in labels]
            if needs_info:
                suggested.append("needs-info")
            if dup_of:
                suggested.append("duplicate")
            if spam:
                suggested.append("spam")
            if gfi:
                suggested.append("good first issue")
            res = {
                "number": iss["number"], "title": title, "author": iss["author"], "url": iss["url"],
                "existing_labels": iss["labels"],
                "labels": [{"name": l, "confidence": p} for l, p in labels],
                "primary": primary, "priority": level, "priority_score": s, "reasons": reasons,
                "quality": quality, "missing": missing, "needs_info": needs_info,
                "duplicate_of": dup_of,
                "similar": [{"number": n, "similarity": round(sm, 3)} for n, sm in sim_list],
                "spam": spam, "spam_reasons": spam_why if spam else [], "noise": noise,
                "tone": tone, "good_first_issue": gfi, "suggested_labels": suggested,
            }
            res["reply"] = self.draft_reply(res)
            results.append(res)

        clusters = Counter(find(i) for i in range(len(issues)))
        summary = {
            "total": len(results),
            "by_label": dict(Counter(r["primary"] for r in results).most_common()),
            "by_priority": {p: sum(1 for r in results if r["priority"] == p) for p in ("P0", "P1", "P2", "P3")},
            "duplicates": sum(1 for r in results if r["duplicate_of"]),
            "duplicate_clusters": sum(1 for c in clusters.values() if c > 1),
            "spam": sum(1 for r in results if r["spam"]),
            "noise": sum(1 for r in results if r["noise"]),
            "needs_info": sum(1 for r in results if r["needs_info"]),
            "good_first_issues": sum(1 for r in results if r["good_first_issue"]),
            "learned_examples": self.learned,
            "minutes_saved": len(results) * 5,
        }
        return results, summary


# --------------------------------------------------------------------------- #
# Built-in demo data (works fully offline)
# --------------------------------------------------------------------------- #

DEMO_ISSUES = [
    {"number": 101, "title": "App crashes with segfault when opening large file",
     "body": "Steps to reproduce:\n1. Open a file larger than 2GB\n2. Click Open\n\nExpected: file opens.\n"
             "Actual: the app crashes.\n\nVersion 2.4.1 on Ubuntu 22.04.\n```\nSegmentation fault (core dumped)\n```",
     "user": {"login": "alice"}, "comments": 6, "reactions": {"total_count": 9}},
    {"number": 102, "title": "Crash on opening big file",
     "body": "The app crashes with a segmentation fault when I open a large file.", "user": {"login": "bob"},
     "comments": 1},
    {"number": 103, "title": "Feature request: dark mode",
     "body": "It would be great to have a dark theme because I use the app at night and the white UI hurts "
             "my eyes.", "user": {"login": "carol"}, "comments": 4, "reactions": {"total_count": 21}},
    {"number": 104, "title": "How do I run this inside Docker?",
     "body": "I read the README but I'm not sure how to configure the container. Is it possible to use a volume "
             "for the config?", "user": {"login": "dave"}},
    {"number": 105, "title": "SQL injection in the search endpoint",
     "body": "The `q` query parameter is concatenated into the SQL string. Sending a quote lets an attacker "
             "dump the users table. This is critical and affects production.",
     "user": {"login": "eve"}, "comments": 2},
    {"number": 106, "title": "Typo in README installation section",
     "body": "It says 'instal' instead of 'install' in the Quick Start. Trivial fix.", "user": {"login": "frank"}},
    {"number": 107, "title": "Memory leak: RAM grows until the process is killed",
     "body": "After running for 3 hours memory usage keeps growing and never goes down. Version 2.4.1, "
             "Python 3.11 on Linux. Expected stable memory but it reaches 8GB.", "user": {"login": "grace"},
     "comments": 3, "reactions": {"total_count": 4}},
    {"number": 108, "title": "pip install fails on macOS",
     "body": "Running pip install fails with error while building wheel on macOS 14, Python 3.12, version 2.4.1. "
             "Steps to reproduce: create a fresh venv and run pip install.\n```\nerror: command 'clang' failed\n```",
     "user": {"login": "heidi"}},
    {"number": 109, "title": "BUY CHEAP CRYPTO NOW!!!",
     "body": "CLICK HERE https://spam.example/1 https://spam.example/2 https://spam.example/3 free money "
             "casino bonus guaranteed", "user": {"login": "spambot9"}},
    {"number": 110, "title": "it doesn't work", "body": "help", "user": {"login": "ivan"}},
    {"number": 111, "title": "Terrible regression: export is totally broken in production!!",
     "body": "After upgrading to 2.4.1 the export used to work, now it crashes and we lost data. This is "
             "ridiculous and a blocker, our whole team cannot use it. Urgent!", "user": {"login": "judy"},
     "comments": 8, "reactions": {"total_count": 12}},
    {"number": 112, "title": "Please add a dark theme",
     "body": "Could you add dark theme support? It would be great to have a dark mode.",
     "user": {"login": "ken"}, "comments": 1},
    {"number": 113, "title": "What is the recommended way to configure multiple users?",
     "body": "Does it support multiple users? Not sure whether this is the correct usage.",
     "user": {"login": "laura"}},
    {"number": 114, "title": "API token printed in debug logs",
     "body": "When debug mode is on, the password and token are leaked in the log file and credentials are "
             "exposed to anyone who can read it.", "user": {"login": "mallory"}},
    {"number": 115, "title": "Broken link in contributing guide",
     "body": "The tutorial link in the docs contributing guide returns 404. Small fix.",
     "user": {"login": "nina"}},
    {"number": 116, "title": "+1", "body": "", "user": {"login": "oscar"}},
]
