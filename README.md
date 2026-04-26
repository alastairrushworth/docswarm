# docswarm

Translate scanned magazine PDFs into schema-conformant JSON via local Ollama models, driven by an iterative Claude Code agent. See `DESIGN.md` for the spec; `AGENT.md` for the agent's guardrails.

The whole stack runs on a DigitalOcean H200 droplet. You drive it from your local terminal: `make run` provisions, runs, and tears down.

## Quick start

```bash
# Install doctl locally and authenticate.
brew install doctl
doctl auth init

# One-time: build a snapshot with Docker, NVIDIA toolkit, Ollama and the
# models pre-pulled. `make snapshot` walks through the steps; paste the
# resulting ID into config.yaml `digitalocean.snapshot_id`.
make snapshot

# Run the iteration loop on a fresh droplet. Streams output to your terminal,
# tears down on exit (success, error, or Ctrl-C).
make run

# Inspect progress at any time (reads trends/round_history.json).
make report

# Final user-only step: run the frozen module against the held-out test set.
make test
```

## What `make run` does

1. `git push origin <current-branch>` so the droplet can fetch your latest code.
2. `doctl compute droplet create` from the snapshot in `config.yaml`.
3. SSH to the droplet, `git pull`, `docker compose up --build`.
4. Inside the stack:
   - `ollama-main` loads the coder + vision models.
   - `judge` watches `judge/inbox/` for marking + broad requests.
   - `developer-agent` runs `scripts/run_validation.py`, which:
     - Round 1: translates val PDFs with the starter pipeline → broad eval → commit.
     - Round n+1: invokes `claude --print` so Claude Code can edit the translator code based on round n's feedback (using marking probes if useful) → re-runs broad eval → commit.
5. On exit (plateau, wall-clock, or you hit Ctrl-C locally), `doctl compute droplet delete` runs.

## Configuration

`config.yaml` is the single source of truth. Hardcoded values in code are forbidden.

### Required before first run

```yaml
repo:
  url:             "git@github.com:USER/REPO.git"      # your repo
  branch:          "agent"
  deploy_key_path: "/secrets/deploy_key"               # mount a key file at ./secrets/deploy_key

digitalocean:
  region:        "nyc2"                                # adjust to a region with H200 availability
  size:          "gpu-h200x1-141gb"
  snapshot_id:   ""                                    # FILL IN after `make snapshot`
  ssh_key_id:    ""                                    # `doctl compute ssh-key list` → ID
```

### Knobs you'll touch most

```yaml
models:
  coder:     "qwen3-coder:32b"        # Claude Code + judge LLM
  vision:    "qwen2.5vl:32b"          # translator (multimodal)
  judge:     "qwen3-coder:32b"
  embedding: "nomic-embed-text"

iteration:
  wall_clock_hours: 12
  page_concurrency: 4                 # parallel page extractions per PDF
  pdf_concurrency: 3                  # parallel PDFs per round

weights:                              # broad-mode aggregate, sums to 1.0
  schema_validity: 0.10
  article_count:   0.10
  metadata:        0.15
  titles:          0.15
  text:            0.30
  order:           0.10
  pages:           0.10
```

## Layout

```
config.yaml             # all tunables — single source of truth
AGENT.md                # guardrails for Claude Code (read first)
DESIGN.md               # full spec

module/pdf_to_json/     # the deliverable
  schema.py             # pydantic — schema source of truth
  pipeline.py           # pdf_to_json(pdf_path) entry point (starter — agent rewrites)
  cache.py              # content-hash-keyed per-page cache
  ollama_client.py      # vision + text + embedding HTTP client
  assemble.py           # IR → schema-conformant Document
  config.py             # config loader

judge/                  # runs in its own container
  judge.py              # inbox/feedback file watcher
  broad.py              # weighted continuous scoring (Hungarian-aligned)
  marking.py            # freeform LLM-driven targeted Q&A
  llm_client.py         # judge → ollama-main /api/chat
  similarity.py         # token + embedding similarity
  alignment.py          # Hungarian alignment
  leakage_filter.py     # n-gram overlap redactor (safety net)
  path_resolver.py      # marking focus.path → truth slice

scripts/
  run_validation.py     # iteration loop (translate → broad → claude → repeat)
  run_test.py           # user-only: frozen module against test set
  report.py             # print round_history table
  export_schema.py      # schema.py → schema/schema.json

orchestration/
  launch.py             # doctl-driven up / down / snapshot

data/
  train/{pdfs,truth}/   # readable by the agent (3 hand-curated pairs)
  val/{pdfs,truth}/     # truth NOT mounted in agent container — judge only
  test/{pdfs,truth}/    # NOT mounted during dev; final user review only

trends/round_history.json   # appended each round, committed
```

## Hard rules (non-leakage)

The developer-agent container does not mount `data/val/truth/` or `data/test/`. The judge container is the only place ground truth lives. Marking feedback passes through a deterministic n-gram overlap filter before being returned. See `AGENT.md` for the full set of rules.
