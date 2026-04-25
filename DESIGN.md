# PDF-to-Structured-JSON Translation: Project Spec

> Status: **v1.0** — agent-ready. All tunable parameters live in `config.yaml` (§6) and can be revised without touching this document.

## 1. Overview

Build a system that develops a Python module which translates magazine PDFs (late-19th-century cycling press, scanned, no reliable OCR) into JSON conforming to a fixed schema. Development is driven by Claude Code as an iterative agent, using local Ollama models — with vision models likely doing significant work, given the input format.

Two run targets:

- **Remote**: H100 droplet on DigitalOcean — for real development runs.
- **Local**: Mac (and probably Windows-via-WSL2) — for plumbing smoke tests.

The agent learns from 3 hand-curated `(PDF, JSON)` training pairs and improves against 3 validation pairs graded by a separate judge process. 3 test pairs are reserved for the user's final review. All set sizes are intended to grow over time.

The deliverable is the Python module, plus orchestration code that lets the user kick off a run from their local terminal with one command.

## 2. Goals and non-goals

**Goals**
- `pdf_to_json(pdf_path: str) -> dict` returning schema-conformant JSON.
- Separation of duties: the developer agent cannot read validation or test ground truth.
- Local Ollama only — no cloud LLM APIs.
- Multi-model strategy: agent routes between vision and text models, large and small, as it sees fit.
- Per-page time budget enforced by the module.
- Single GitHub repo as the working surface. Per-round commits with score in message — `git log` becomes the audit trail.
- Single-command launch from the user's local terminal in either mode.
- Local mode reaches end-to-end stack-up in <5 minutes.
- All grading is graded (continuous), not pass/fail. Stop conditions are wall-clock + improvement plateau.
- Round-by-round trend report committed to the repo so the user can see trajectory.

**Non-goals**
- Generic PDF parsing — the module may be tightly fitted to this corpus.
- Web UI.
- Online learning. The released module is frozen.
- 100% accuracy.
- Local mode as a quality test. Plumbing only.

## 3. Schema (starter Pydantic — finalize in `module/pdf_to_json/schema.py`)

The model below is a working starting point with default decisions baked in. The user is expected to refine it in `schema.py`; the agent should treat that file (not this section) as the source of truth.

```python
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel

class Issue(BaseModel):
    date: date                # ISO; example: 1892-06-03
    volume: int
    number: int

class Publisher(BaseModel):
    name: str
    address: str

class Cost(BaseModel):
    issue:      Optional[str] = None
    annual:     Optional[str] = None     # currency-prefixed strings, e.g. "$2.00"
    semiannual: Optional[str] = None

class MagazineMeta(BaseModel):
    editor:    str
    issue:     Issue
    publisher: Publisher
    cost:      Cost

class Article(BaseModel):
    title: str
    text:  list[str]                                  # paragraphs (prose) or lines (verse)
    pages: list[int]                                  # 1-indexed printed pages, in order
    kind:  Literal["prose", "verse"] = "prose"        # explicit, avoids ambiguity in `text`

class Document(BaseModel):
    magazine: MagazineMeta
    articles: list[Article]                           # ORDERED by printed sequence
```

Default decisions baked in (override in `schema.py` if needed):

- `Article.kind` is explicit. Prose → `text` is paragraphs; verse → `text` is lines.
- Currency stays as strings to match the user's example.
- Page numbers are 1-indexed printed pages, not PDF page indices.
- `articles` is ordered by printed sequence; multi-page articles list pages in printed order (`[1, 2]` or `[1, 7]` for a continuation).
- Typographic artifacts in the source (e.g. stray quote marks, OCR-style typos in the user's transcripts) are **preserved verbatim**, not normalized. This matches the example output. If the user wants normalization, change `schema.py` and add a normalization pass.

Example construction:

```python
doc = Document(
    magazine=MagazineMeta(
        editor="L. J. Berger",
        issue=Issue(date=date(1892, 6, 3), volume=5, number=18),
        publisher=Publisher(name="N. H. Van Sicklen",
                            address="57 Plymouth Place, Chicago"),
        cost=Cost(annual="$2.00", semiannual="$1.00"),  # issue=None
    ),
    articles=[
        Article(title="That's So!",
                text=["It is not always the man who rides the swiftest..."],
                pages=[1],
                kind="prose"),
        Article(title="Both Were Pleased.",
                text=['The haughty rider to Heaven gave thanks',
                      'He "was not made like other men."',
                      'And riders who viewed his skinny shanks',
                      'Pleased at the truth, gave thanks again!'],
                pages=[1],
                kind="verse"),
    ],
)
```

## 4. Data partitioning

| Set | Initial count | PDF readable by developer agent | Ground truth readable by developer agent | Used for grading |
|---|---|---|---|---|
| Train | 3 | yes | yes | no — reference only |
| Validation | 3 | yes | **no — judge container only** | yes |
| Test | 3 | no during dev | no | no — final user review |

Code globs the data directories; counts are not hardcoded.

## 5. Architecture

### 5.1 Containers

Three roles, each in its own Docker container. Same compose file works locally and remotely; differences are GPU mounts and model sizes (§9).

1. **Developer Agent container** — runs Claude Code. Mounts: project repo (rw), `data/train/` (ro), `data/val/pdfs/` (ro), `judge/inbox/` (rw), `judge/feedback/` (ro). **No mount** for `data/val/truth/` or anything under `data/test/`.
2. **Judge container** — runs `judge.py` plus its own Ollama instance. Mounts: `data/val/truth/` (ro), `judge/inbox/` (ro), `judge/feedback/` (rw). Exposes only a healthcheck.
3. **Ollama service container(s)** — one or more Ollama servers on a Docker network. Translator and judge use *different* models so their errors are uncorrelated.

```
host (local Mac OR remote H100 droplet)
├── docker-compose.yml             # base
├── docker-compose.remote.yml      # GPU mounts, full models
├── docker-compose.local.yml       # no GPU, small models only
├── ollama-translator
├── ollama-judge          (different model from translator)
├── developer-agent       (Claude Code)
└── judge
```

### 5.2 GitHub repo as state

A single GitHub repo (URL in `config.yaml`) is the working surface and artifact target.

- Agent's container has a deploy key with write access. Cloned at `/workspace`.
- Per round, the agent: edits files → runs validation → commits → pushes. Commit message template:
  `round {n}: aggregate {score:.3f} (Δ{delta:+.3f}); titles {x}, text {y}, order {z}, ...`
- A trend file (§7.2) is committed alongside code, so the history shows both code changes and score evolution.
- All work happens on the configured branch (default `agent`). User merges to `main` when satisfied.

## 6. Configuration

`config.yaml` is the single source of truth for every tunable parameter. The user fills in repo URL, deploy key path, and DigitalOcean credentials before the first run; the agent may revise model selections during early rounds and commit the changes.

```yaml
# ----- Project identity -----
repo:
  url:             "git@github.com:USER/REPO.git"   # FILL IN before first run
  branch:          "agent"
  deploy_key_path: "/secrets/deploy_key"            # mounted into developer-agent container

digitalocean:
  api_token_env:   "DO_API_TOKEN"                   # name of env var holding the token
  region:          "nyc2"                           # adjust to a region with H100 availability
  size:            "gpu-h100x1-80gb"                # adjust to current DO SKU
  snapshot_id:     ""                               # FILL IN after first `make build-snapshot`

# ----- Models (Ollama tags) -----
# Agent may revise these in early rounds and commit the change.
models:
  vision_small: "llama3.2-vision:11b"
  vision_large: "minicpm-v:latest"
  text_small:   "llama3.2:3b"
  text_large:   "qwen2.5:32b"
  judge:        "mistral:7b"           # different family from translators

# ----- Iteration control -----
iteration:
  wall_clock_hours:           12
  page_budget_seconds:        30
  epsilon:                    0.005    # min Δaggregate to count as improvement
  min_rounds_before_plateau:  3        # plateau detector inactive before this
  plateau_aggregate_floor:    0.30     # plateau detector inactive while aggregate < floor
  plateau_window:             3        # rounds since best to trigger stop

# ----- Broad-mode scoring weights (sum to 1.0) -----
weights:
  schema_validity: 0.10
  article_count:   0.10
  metadata:        0.15
  titles:          0.15
  text:            0.30
  order:           0.10
  pages:           0.10

# ----- Judge: non-leakage policy -----
leakage:
  allow_structural_hints:        true   # "missing article on page 3" allowed
  hint_overlap_filter_threshold: 0.30   # substring-overlap threshold for redaction

# ----- Local mode overrides -----
local:
  models:
    vision_large: null                  # disabled on local; vision_small only
    text_large:   null
  iteration:
    wall_clock_hours: 1                 # local is for plumbing, not real iteration
```

The agent reads from `config.yaml` at startup. Hardcoded values in code are forbidden.

## 7. Translation module — approach is the agent's call

The PDFs are scanned images with complex magazine layouts. There is no reliable text layer to lean on, and conventional extraction (PyMuPDF, pdfplumber) is expected to be insufficient on its own. The agent will likely need vision models doing significant work to:

- Recover article boundaries and reading order across multi-column layouts.
- Distinguish article body, headers, captions, ads, and editorial chrome.
- Trace continuation across pages.
- Preserve verse line breaks.

The spec deliberately does not prescribe an architecture. The module's interface is fixed:

```python
def pdf_to_json(pdf_path: str) -> dict: ...
```

Internals are the agent's design space. What the spec *does* fix:

- **Available models**: from `config.yaml`. Agent picks per call.
- **Per-page time budget**: from `config.yaml`. Enforced by timeout. On exhaustion, return whatever partial output is available for that page and emit a warning. The module always returns *some* JSON conforming (best effort) to the schema.
- **VRAM**: agent should query `nvidia-smi` at startup, log the available VRAM, and choose a model loadout that fits — ideally translator-large + judge co-resident on the H100 (typically 80GB) so neither has to hot-swap. If memory is tight, prefer keeping the judge resident and swapping translator-large in when needed. Agent records its chosen loadout at startup so the user can see what it picked.
- **Caching**: vision passes are expensive. The agent is encouraged to cache per-page intermediate representations across iteration rounds, **keyed by content hash of the PDF** (not filename, to avoid cache poisoning if a PDF is replaced). Implementation left to the agent.

## 8. Iteration loop

### 8.1 Loop

Within a round, the agent uses the judge in two modes (§9):

- **Marking** (interactive): probe specific articles or fields while editing. Cheap, frequent.
- **Broad** (gate): full-document evaluation against the weighted score battery at end of round.

```python
scores_history = []
best = -inf
rounds_since_best = 0
deadline = now() + timedelta(hours=cfg.iteration.wall_clock_hours)

while now() < deadline:
    developer_agent.work(...)          # may issue many marking queries

    submissions = {pdf.id: module.pdf_to_json(pdf.path) for pdf in val_set}
    feedback   = judge.broad_evaluate(submissions, timeout=...)
    scores_history.append(feedback.aggregate_score)

    write_trend_file(scores_history)
    git_commit_and_push(round, feedback)

    if feedback.aggregate_score > best + cfg.iteration.epsilon:
        best = feedback.aggregate_score
        rounds_since_best = 0
    else:
        rounds_since_best += 1

    plateau_active = (
        round_idx >= cfg.iteration.min_rounds_before_plateau
        and best >= cfg.iteration.plateau_aggregate_floor
    )
    if plateau_active and rounds_since_best >= cfg.iteration.plateau_window:
        log("plateau: stopping")
        break

    developer_agent.revise(feedback)
```

### 8.2 Stopping and trends

Stop conditions (whichever fires first):

- Wall-clock budget exceeded (`iteration.wall_clock_hours`).
- Plateau: `iteration.plateau_window` rounds since the last best aggregate, **once** plateau detection is active. Detection requires both `round >= min_rounds_before_plateau` AND `best >= plateau_aggregate_floor` — this gives the agent room to escape catastrophic first-round failures (e.g. broken JSON parsing) without early-stopping during debug.

A trend file `trends/round_history.json` is committed every round:

```json
[
  {
    "round": 1,
    "timestamp": "2025-04-25T14:02:11Z",
    "wall_clock_seconds": 612,
    "aggregate": 0.41,
    "components": {
      "schema_validity": 0.85,
      "article_count":   0.50,
      "metadata":        0.66,
      "titles":          0.30,
      "text":            0.22,
      "order":           0.40,
      "pages":           0.50
    },
    "marking_calls_in_round": 14,
    "model_loadout": "translator_large=qwen2.5:32b, judge=mistral:7b"
  }
]
```

A simple CLI report at run end (and on each round) prints a table:

```
Round | Agg   | schema | count | meta | titles | text | order | pages | Δ      | time
   1  | 0.41  | 0.85   | 0.50  | 0.66 | 0.30   | 0.22 | 0.40  | 0.50  | -      | 10m
   2  | 0.49  | 0.90   | 0.67  | 0.66 | 0.45   | 0.31 | 0.50  | 0.60  | +0.08  | 11m
   3  | 0.55  | 0.95   | 0.83  | 0.83 | 0.52   | 0.38 | 0.55  | 0.65  | +0.06  | 12m
```

This is the user's at-a-glance view of how the run is going.

## 9. Judge

### 9.1 Two modes

- **Broad mode** is the gate. Full-document submission. Returns weighted continuous scores plus categorical hints. No pass/fail anywhere.
- **Marking mode** is targeted. Agent submits a partial output asking for scoring on a specific scope. Used for iterative debugging.

Both modes obey non-leakage rules (§9.5).

### 9.2 Broad mode — weighted continuous scoring

Every component is a continuous score in [0, 1]. The aggregate is a weighted mean using `weights` from `config.yaml`.

| Component | Metric |
|---|---|
| `schema_validity` | fraction of `Document` fields that pydantic-validate (partial credit per field) |
| `article_count` | `1 - |n_pred − n_truth| / max(n_pred, n_truth)` |
| `metadata` | mean of per-field exact-match indicators across editor/issue/publisher/cost |
| `titles` | mean title similarity over Hungarian-aligned articles |
| `text` | mean text similarity over Hungarian-aligned articles (sentence embeddings, e.g. `nomic-embed-text`) |
| `order` | order score after Hungarian alignment (§9.4) |
| `pages` | mean Jaccard over `pages` arrays of aligned articles |

The aggregate is the user's primary signal. Component scores diagnose *what* went wrong.

### 9.3 Marking mode — targeted scoring

The agent submits:

```json
{
  "mode": "marking",
  "pdf_id": "issue_1892_06_03",
  "scope": {
    "article_index": 7,
    "fields": ["text", "title"]
  },
  "prediction": { ... full or partial document ... }
}
```

The judge returns scores only for the requested scope, plus a categorical hint if the score is low. Other valid scopes:

- `{"metadata_field": "publisher.address"}`
- `{"article_index": 7, "fields": "all"}`
- `{"section": "metadata"}`

This lets the agent test focused changes (e.g. "I changed how I detect verse, score article 7 again") without re-running the full document end-to-end.

### 9.4 Order scoring

After Hungarian alignment of predicted ↔ truth articles by combined title+text similarity:

```
matched_pairs = hungarian(pred_articles, truth_articles)
displacements = [|i_pred - i_truth| for (i_pred, i_truth) in matched_pairs]
order_score   = 1 - mean(displacements) / n_articles
```

Unmatched articles affect `article_count` rather than `order`.

### 9.5 Non-leakage rules

The judge must not transmit ground-truth content to the agent.

**ALLOWED:**
- Aggregate counts ("expected 9 articles, got 8").
- Per-field similarity scores as numbers.
- Generic error categories: `missing_article`, `extra_article`, `truncated_text`, `wrong_metadata_field`, `verse_misformatted_as_prose`, `paragraph_split_incorrectly`, `page_number_wrong`, `articles_out_of_order`.
- References to predicted output by index.
- Non-content structural hints if `leakage.allow_structural_hints: true` ("the missing article is on page 3").

**FORBIDDEN:**
- Any verbatim string from ground truth (titles, names, dates, phrases).
- Paraphrases of ground-truth content.
- Length-revealing hints beyond bucketed ranges ("~40% shorter than expected" allowed; "should be 312 words" not).
- Triangulating hints ("editor's last name starts with B").

LLM-generated hints pass through a deterministic post-filter that checks for substring overlap with ground truth above `leakage.hint_overlap_filter_threshold` and redacts/regenerates if found. Filter is the safety net; the judge's prompt is the first line of defense.

This is non-adversarial. Per-field numeric scores leak information bit-by-bit over many rounds; acceptable for the use case.

### 9.6 Output schemas

Broad mode:

```json
{
  "mode": "broad",
  "pdf_id": "issue_1892_06_03",
  "round": 2,
  "aggregate": 0.49,
  "components": {
    "schema_validity": {"score": 0.90},
    "article_count":   {"score": 0.67, "delta": 1},
    "metadata":        {"score": 0.66, "matched": 4, "total": 6},
    "titles":          {"score": 0.45},
    "text":            {"score": 0.31},
    "order":           {"score": 0.50},
    "pages":           {"score": 0.60}
  },
  "categorical_errors": [
    {"category": "missing_article", "count": 1},
    {"category": "verse_misformatted_as_prose", "predicted_index": 7}
  ],
  "hints": [
    "One article on page 1 is missing.",
    "Predicted article #7 appears to be verse but was emitted as prose paragraphs."
  ]
}
```

Marking mode:

```json
{
  "mode": "marking",
  "pdf_id": "issue_1892_06_03",
  "scope": {"article_index": 7, "fields": ["text", "title"]},
  "scores": {"text": 0.62, "title": 0.95},
  "categorical_errors": [{"category": "verse_misformatted_as_prose"}],
  "hints": ["This article appears to be verse; predicted text uses prose paragraph splits."]
}
```

## 10. Orchestration: local and remote

### 10.1 Common command interface

```
make run-remote      # provision H100, run loop, tear down
make run-local       # local docker-compose, smoke-test plumbing
make run-local-fast  # local, single PDF, stub model, <60s end-to-end
make build-snapshot  # rebuild the DO snapshot when models change
make test            # run final module against test set (user-only)
make report          # print latest round_history table
```

`run-remote` and `run-local` invoke the same agent loop; differences are compose overlay and config.

### 10.2 Remote mode (H100)

`make run-remote`:

1. **Provision**: DigitalOcean API creates an H100 droplet from the snapshot identified in `config.yaml` (Docker, NVIDIA drivers, Ollama, pre-pulled model weights).
2. **Sync**: agent's repo is cloned via deploy key directly on the droplet; the local script pushes its current branch first.
3. **Bring up**: `ssh` runs `docker compose -f docker-compose.yml -f docker-compose.remote.yml up --build`.
4. **Stream logs**: `ssh -t` tails the agent container's stdout to the local terminal. Persisted on the droplet too.
5. **Capture artifacts**: agent's commits/pushes are the artifacts; user pulls when run ends.
6. **Auto-shutdown**: a `trap` in the local script destroys the droplet on exit (success/error/`Ctrl-C`). **Plus** a watchdog cron on the droplet self-destructs after `iteration.wall_clock_hours` regardless of what the local script does. Belt-and-braces — a network blip on the laptop should not cost real money.

Cost printed at run start: estimated cap and hard kill timestamp.

### 10.3 Local mode (Mac, optional Windows via WSL2)

`make run-local`:

1. No provisioning. Docker Desktop on Mac (Apple Silicon: Metal; Intel: CPU only).
2. Same containers, same compose, but `docker-compose.local.yml` overlay: removes GPU mounts, applies the `local:` overrides from `config.yaml` (small models only, short wall-clock).
3. Repo is the local checkout (mounted directly, no push/pull).
4. Logs to local terminal directly.
5. **Caveat printed at run start**: "LOCAL MODE: plumbing test only. Output quality is not representative of remote H100 runs because Metal/CPU and CUDA produce different model outputs. Use this to verify containers come up and data flows, not to evaluate translation quality."

`make run-local-fast`: single tiny synthetic PDF, stubbed Ollama (returns canned JSON), one round. Completes in under a minute. CI-style sanity check.

### 10.4 Local mode is for plumbing only

The temptation to iterate locally to save money is real. Resist it. Vision models on Metal/CPU will be far slower and produce different outputs than CUDA on H100. If local broad mode says 0.4 and remote says 0.6, that's not evidence of anything. Use local for "containers came up, deploy key worked, judge container responded." That's it.

## 11. Suggested directory structure

```
project/
├── Makefile
├── docker-compose.yml
├── docker-compose.remote.yml
├── docker-compose.local.yml
├── orchestration/
│   ├── provision.py            # DO droplet up/down
│   ├── run.sh
│   └── teardown.sh
├── data/
│   ├── train/{pdfs,truth}/
│   ├── val/{pdfs,truth}/
│   └── test/{pdfs,truth}/
├── module/
│   ├── pyproject.toml
│   ├── pdf_to_json/
│   │   ├── pipeline.py         # orchestration of vision/text passes
│   │   ├── cache.py            # per-page intermediate caching (content-hashed)
│   │   ├── assemble.py         # IR → schema-conformant JSON
│   │   └── schema.py           # pydantic — source of truth for the schema
│   └── tests/
├── judge/
│   ├── judge.py
│   ├── broad.py                # weighted continuous scoring
│   ├── marking.py              # targeted scoring
│   ├── leakage_filter.py
│   ├── inbox/
│   └── feedback/
├── trends/
│   └── round_history.json      # appended each round, committed
├── scripts/
│   ├── run_validation.py
│   ├── run_test.py             # user-only
│   └── report.py               # print trend table
├── schema/
│   ├── schema.json             # exported from schema.py for docs
│   └── example_output.json
├── config.yaml                 # all tunables — see §6
└── AGENT.md                    # guardrails for Claude Code — see §12
```

## 12. Concerns to be aware of

A. **3 validation PDFs is a noisy aggregate signal.** The plateau detector at default `epsilon=0.005` may trigger on noise rather than real plateaus. Grow validation as soon as practical. Until then, the test-set human review is the real signal.

B. **Aggregate weights are guesses.** Defaults put 0.30 on text similarity. Reconsider after first run shows what's actually achievable per component.

C. **Vision model outputs are not deterministic across model/backend versions.** When a good loadout is found, pin model versions in `config.yaml` and rebuild the snapshot.

D. **Cache invalidation is the agent's problem.** Cache keys must be content hashes of the PDF, not filenames. A filename-keyed cache will silently return stale output if a PDF is replaced.

E. **12h × ~$4/h ≈ $50 per full run.** Each spec/agent-prompt iteration costs real money. Use `run-local` aggressively for plumbing. Reserve remote for runs where the plumbing is known-good.

F. **Catastrophic first-round failures are protected.** The plateau detector requires both `min_rounds_before_plateau` AND `plateau_aggregate_floor` to activate. Stuck at 0.05 for many rounds → still iterating, not stopped. If this turns out to be the wrong default, lower the floor in `config.yaml`.

G. **Per-round commits inflate git history.** The `agent` branch will accumulate many commits. Periodically squash, or accept the noise as the price of the audit trail.

H. **The non-leakage filter is best-effort.** A clever paraphrase from the judge LLM might evade the substring filter. Acceptable for non-adversarial use.

## 13. AGENT.md (required contents — lives in repo root, read by Claude Code first)

- **Hard rules.** Paths the agent must not read: `data/val/truth/`, `data/test/`. Not mounted; do not attempt via network or shell.
- **Source of truth.** All tunable parameters live in `config.yaml`. The schema's source of truth is `module/pdf_to_json/schema.py`. Hardcoded constants in code are forbidden — read from config.
- **Submission protocol.**
  - Marking: drop `judge/inbox/{pdf_id}__marking__{ts}.json`. Read `judge/feedback/{same-name}.json`.
  - Broad: drop `judge/inbox/{pdf_id}__broad__round{n}.json`. Read response from feedback dir.
- **Iteration discipline.** Use marking liberally during a round to debug specific issues. Run broad evaluation once at end of round.
- **Stop signals.** Wall-clock and plateau detection per `config.yaml`. Do not implement other stop conditions.
- **Posture.** Make best efforts to close every error. Graded scores are *not* permission to settle for low component scores — they are protection against indefinite blocking on a single hard issue. If a component score is stuck, try a different approach rather than quitting on it.
- **Commit discipline.** Per-round commit, message format: `round {n}: aggregate {score:.3f} (Δ{delta:+.3f}); {component summary}`.
- **Trend file.** Append a new entry to `trends/round_history.json` each round.
- **Coding conventions.** Type hints, pydantic for IO, pytest for any non-LLM helpers.
- **Model selection.** Use names from `config.yaml`. May revise these in early rounds and commit the change; do not pull arbitrary new models that aren't in the snapshot without rebuilding it.
- **VRAM allocation.** Query `nvidia-smi` at startup. Try to keep both translator-large and judge resident; fall back to swapping if necessary. Log the choice.
- **Caching.** Cache vision-model per-page outputs across rounds, keyed by **content hash** of the PDF.
- **The deliverable is the module.** Reasoning transcripts and scratch notes are not the product.