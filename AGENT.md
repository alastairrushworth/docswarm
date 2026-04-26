# AGENT.md — guardrails for Claude Code

You are the developer agent for this project. Read this file first, then `DESIGN.md` for full context.

## Hard rules — non-negotiable

- **Never read** `data/val/truth/` or anything under `data/test/`. These are not mounted into your container; do not try to obtain them via network, shell, git history of other branches, or any other channel.
- **Source of truth for tunables**: `config.yaml`. Hardcoded constants in code are forbidden. Read from config.
- **Source of truth for the schema**: `module/pdf_to_json/schema.py`. Treat this file (not `DESIGN.md` §3) as authoritative; revise it in `schema.py` if needed.
- **The deliverable is the module** in `module/pdf_to_json/`. Reasoning transcripts and scratch notes are not the product.

## Submission protocol (judge)

The judge runs in a separate container. Communicate via the filesystem:

- **Marking**: write `judge/inbox/{pdf_id}__marking__{ts}.json`. Read response from `judge/feedback/{same-name}.json`.
- **Broad**: write `judge/inbox/{pdf_id}__broad__round{n}.json`. Read response from `judge/feedback/{same-name}.json`.

Request payload format: see `judge/judge.py` and `DESIGN.md` §9.3 / §9.6.

## Iteration discipline

- Use **marking mode** liberally during a round to debug specific articles or fields. Cheap, frequent.
- Run **broad mode** once at end of round, on the full validation set.
- Per round: edit code → marking probes → broad eval → commit & push.

## Stop signals

Wall-clock and plateau detection per `config.yaml`. Do not implement other stop conditions. The plateau detector is intentionally inactive while `round < min_rounds_before_plateau` or `best < plateau_aggregate_floor` — this is so first-round catastrophic failures do not stop the loop.

## Posture

Make best efforts to close every error. Graded scores are *not* permission to settle for low component scores — they are protection against indefinite blocking on a single hard issue. If a component score is stuck round-on-round, **try a different approach** rather than quitting on it.

## Commit discipline

- Per-round commit, message format:
  `round {n}: aggregate {score:.3f} (Δ{delta:+.3f}); titles {x:.2f}, text {y:.2f}, order {z:.2f}, ...`
- Append a new entry to `trends/round_history.json` each round and commit.
- Push to the branch named in `config.yaml` (`repo.branch`).

## Coding conventions

- Type hints throughout.
- Pydantic for all IO (judge requests/responses, schema).
- Pytest for non-LLM helpers (alignment, leakage filter, cache key derivation).
- No backwards-compatibility shims. No feature flags for hypothetical futures.

## Model selection

- Use names from `config.yaml.models`. You may revise these in early rounds and commit the change.
- Do **not** pull arbitrary new Ollama models that aren't in the snapshot without rebuilding it (`make build-snapshot`). Pulling at runtime on the H100 wastes wall-clock budget.

## VRAM allocation

At startup:

1. Run `nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits`. Log result.
2. Choose a loadout. Ideally translator-large + judge co-resident. If memory is tight, prefer keeping the judge resident and swapping translator-large in when needed.
3. Log the chosen loadout and include it in the round trend entry as `model_loadout`.

## Caching

Cache vision-model per-page outputs across rounds.

- **Key**: SHA-256 of (PDF file bytes, page index, model tag, prompt template version).
- **Not** filename. A filename-keyed cache will silently return stale output if a PDF is replaced.
- Cache lives at `paths.cache_dir`. Implementation in `module/pdf_to_json/cache.py`.

## Per-page time budget

The module enforces `iteration.page_budget_seconds` per page. On exhaustion, return whatever partial output is available for that page and emit a warning. The module **always** returns *some* JSON conforming (best effort) to the schema — never raise out of `pdf_to_json`.

## Non-leakage (you are on the receiving end)

The judge will not transmit ground truth. Hints are categorical or numeric. If you ever see what looks like a verbatim ground-truth phrase in feedback, treat it as a bug in the leakage filter and report it; do not exploit it.
