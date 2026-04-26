# docswarm

Translate scanned magazine PDFs into schema-conformant JSON via local Ollama models, driven by an iterative Claude Code agent. See `DESIGN.md` for the full spec; `AGENT.md` for the agent's guardrails.

## Quick start

```bash
# 1) Local sanity test — ~30s, no Ollama, no Docker. Verifies wiring end-to-end.
make run-local-fast

# 2) Local plumbing test — brings up the Docker stack against local Ollama.
#    Output quality is NOT representative of remote runs (Metal/CPU vs CUDA).
make run-local

# 3) Real iteration run on a DigitalOcean H100 droplet.
#    Provisions, runs the loop, tears down on exit (plus on-droplet watchdog).
make run-remote

# Inspect progress
make report

# Final user-only step: run the frozen module against the held-out test set.
make test
```

## Test modes

| Target | What it does | Needs Ollama? | Needs GPU? | Needs DO? | Time |
|---|---|---|---|---|---|
| `make run-local-fast` | Full Docker stack with **stubbed Ollama** (canned responses), one round on a synthetic PDF | no | no | no | <60s |
| `make run-local` | Full Docker stack with **real local Ollama** (small models) | yes (small) | no | no | minutes |
| `make run-remote` | Full iteration loop on H100 droplet | yes (full) | yes | yes | up to `iteration.wall_clock_hours` |

`run-local-fast` is the right thing to run after editing module/judge code — it brings up the real Docker stack (translator container, judge container, agent container, inbox/feedback file protocol, trend file) but swaps the Ollama image for a tiny stub (`docker/stub_ollama.py`) that returns canned JSON. No model weights downloaded.

## Configuration

All tunables live in `config.yaml` — that file is the only place to change models, scoring weights, iteration controls, and infrastructure settings. Hardcoded values in code are forbidden.

### Required before first remote run

```yaml
repo:
  url: "git@github.com:USER/REPO.git"            # your repo
  deploy_key_path: "/secrets/deploy_key"         # mount a key file at ./secrets/deploy_key

digitalocean:
  api_token_env: "DO_API_TOKEN"                  # export this env var locally
  snapshot_id: ""                                # set after `make build-snapshot`
```

Plus: `export DO_API_TOKEN=...` in your shell.

### Knobs you'll touch most

```yaml
models:
  vision_small: "llama3.2-vision:11b"
  vision_large: "minicpm-v:latest"
  text_large:   "qwen2.5:32b"
  judge:        "mistral:7b"            # different family from translators

iteration:
  wall_clock_hours:          12         # hard remote budget
  page_budget_seconds:       30         # per-page timeout in pdf_to_json
  plateau_window:            3          # rounds-since-best to trigger early stop
  plateau_aggregate_floor:   0.30       # plateau detector inactive below this

weights:                                # broad-mode aggregate, must sum to 1.0
  schema_validity: 0.10
  article_count:   0.10
  metadata:        0.15
  titles:          0.15
  text:            0.30
  order:           0.10
  pages:           0.10
```

### Local-mode overrides

The `local:` section in `config.yaml` is applied when `DOCSWARM_MODE=local`. By default it disables large models and shortens the wall-clock budget.

## Layout

```
config.yaml             # all tunables — single source of truth
AGENT.md                # guardrails for Claude Code (read first)
DESIGN.md               # full spec

module/pdf_to_json/     # the deliverable
  schema.py             # pydantic — schema source of truth
  pipeline.py           # pdf_to_json(pdf_path) entry point
  cache.py              # content-hash-keyed per-page cache
  ollama_client.py      # vision + text + embedding HTTP client
  assemble.py           # IR → schema-conformant Document

judge/                  # runs in its own container
  broad.py              # weighted continuous scoring (Hungarian-aligned)
  marking.py            # targeted per-article / per-field scoring
  leakage_filter.py     # n-gram overlap redactor (safety net)
  judge.py              # inbox/feedback file watcher

scripts/
  run_validation.py     # iteration loop (translate → broad eval → commit → push)
  run_test.py           # user-only: frozen module against test set
  run_local_fast.py     # stub-mode smoke test
  report.py             # print round_history table
  export_schema.py      # schema.py → schema/schema.json

orchestration/
  provision.py          # DigitalOcean droplet up/down
  run.sh                # one-command launcher (local | remote)
  teardown.sh           # manual droplet teardown

data/
  train/{pdfs,truth}/   # readable by the agent (3 hand-curated pairs)
  val/{pdfs,truth}/     # truth NOT mounted in agent container — judge only
  test/{pdfs,truth}/    # NOT mounted during dev; final user review only

trends/round_history.json   # appended each round, committed
```

## Where output goes

- Per-round broad-eval feedback: `judge/feedback/{pdf_id}__broad__round{n}__*.json`
- Trend history: `trends/round_history.json`
- Test-set predictions (after `make test`): `data/test/predictions/{pdf_id}.json`
- Audit trail: per-round commits on the `agent` branch

## Hard rules (non-leakage)

The developer-agent container does not mount `data/val/truth/` or `data/test/`. The judge container is the only place ground truth lives. Hints from the judge pass through a deterministic n-gram overlap filter before being returned. See `AGENT.md` for the full set of rules.
