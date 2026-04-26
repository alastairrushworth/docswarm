# PDF-to-Structured-JSON Translation: Project Spec

> Status: **v1.0** — agent-ready. All tunable parameters live in `config.yaml` (§6) and can be revised without touching this document.

## 1. Overview

Build a system that develops a Python module which translates magazine PDFs (late-19th-century cycling press, scanned, no reliable OCR) into JSON conforming to a fixed schema. Development is driven by Claude Code as an iterative agent, using local Ollama models — with vision models likely doing significant work, given the input format.

**Run target**: a single H200 droplet on DigitalOcean. The user drives the run from their local terminal — `make run` provisions, runs, and tears down. Local-only modes were tried and removed: Metal/CPU vs CUDA divergence made local results misleading, and the `doctl`-driven flow is fast enough that the value of a local plumbing path didn't justify its maintenance cost.

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

Three services on a single Docker network:

1. **`ollama-main`** — the Ollama server. `OLLAMA_MAX_LOADED_MODELS` is set so two big models stay resident together: `models.coder` (drives Claude Code via Ollama's Anthropic-compatible API; also serves the judge's marking LLM) and `models.vision` (drives the translator). Embedding model loads alongside. All three services hit this one Ollama. Two coexist instead of two separate Ollama containers because the Ollama runtime already multiplexes models cleanly and the H200's 141 GB VRAM is plenty for both.
2. **`developer-agent`** — runs the iteration harness, which between rounds shells out to `claude --print` so Claude Code can edit the translator code. Mounts: project repo (rw), `data/train/` (ro), `data/val/pdfs/` (ro), `judge/inbox/` (rw), `judge/feedback/` (ro). **No mount** for `data/val/truth/` or anything under `data/test/`.
3. **`judge`** — runs `judge.judge` watching `judge/inbox/`. Mounts: `data/val/truth/` (ro), `judge/inbox/` (ro), `judge/feedback/` (rw).

```
host (local Mac, with doctl)
└── DigitalOcean H200 droplet
    ├── docker-compose.yml
    ├── ollama-main          (qwen3-coder + qwen2.5vl + nomic-embed)
    ├── developer-agent      (harness + Claude Code)
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
# Two big models loaded concurrently in ollama-main; embedding alongside.
# Agent may revise these in early rounds and commit the change.
models:
  coder:     "qwen3-coder:32b"     # Claude Code + judge LLM
  vision:    "qwen2.5vl:32b"       # translator (multimodal)
  judge:     "qwen3-coder:32b"
  embedding: "nomic-embed-text"

# ----- Iteration control -----
iteration:
  wall_clock_hours:           12
  page_concurrency:           4        # parallel pages per PDF
  pdf_concurrency:            3        # parallel PDFs per round
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
  allow_structural_hints:        true
  hint_overlap_filter_threshold: 0.30

# ----- Single Ollama, two big models loaded -----
ollama:
  url:                "http://ollama-main:11434"
  num_parallel:       4
  max_loaded_models:  3
  keep_alive:         "24h"
  context_length:     65536
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

- **Available models**: from `config.yaml`. Agent picks per call. Both `models.coder` and `models.vision` stay resident in `ollama-main` (max-loaded-models setting) so neither has to hot-swap.
- **Resilience**: the module always returns *some* JSON conforming (best effort) to the schema. On per-call timeout or extraction failure, fill what you can and continue. Never raise.
- **Caching**: vision passes are expensive. The agent is encouraged to cache per-page intermediate representations across iteration rounds, **keyed by content hash of the PDF** (not filename, to avoid cache poisoning if a PDF is replaced). Cache writes are atomic (tmp + rename) so concurrent writers and killed processes can't corrupt entries.

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

## 10. Orchestration

### 10.1 Command interface

```
make run       # provision H200 droplet, run loop, tear down on exit
make down      # destroy a droplet whose ID we recorded (recovery)
make snapshot  # walk through snapshot creation
make test      # run frozen module against held-out test set (user-only, local)
make report    # print latest round_history table
```

### 10.2 What `make run` does

`orchestration/launch.py up` drives `doctl` from your local terminal:

1. **Push**: `git push origin <current-branch>` so the droplet can fetch your latest code.
2. **Provision**: `doctl compute droplet create` from the snapshot in `config.yaml` (Docker, NVIDIA Container Toolkit, Ollama, pre-pulled model weights). Waits until the droplet has a public IP.
3. **Bring up**: `ssh root@<ip>` runs `cd /workspace && git pull && docker compose up --build --abort-on-container-exit developer-agent`.
4. **Stream**: `ssh -t` ties the agent container's stdout to your local terminal.
5. **Capture artifacts**: the agent's per-round commits/pushes are the artifacts — pull when you want to inspect.
6. **Auto-shutdown**: a SIGINT/SIGTERM/EXIT trap in the local launcher calls `doctl compute droplet delete` on exit (success, error, or Ctrl-C). The agent loop also exits naturally on plateau or wall-clock; `--abort-on-container-exit developer-agent` brings the whole stack down with it.

Recovery: if the local launcher crashes mid-run, `make down` reads the recorded droplet ID and destroys it. If that file is also gone, list with `doctl compute droplet list --tag-name docswarm`.

## 11. Suggested directory structure

```
project/
├── Makefile
├── docker-compose.yml
├── orchestration/
│   └── launch.py               # doctl-driven up / down / snapshot
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

E. **12h × H200 hourly rate is real money per full run.** Each spec/agent-prompt iteration costs. Iterate on harness changes locally (unit tests, small dry runs) before kicking off `make run`.

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