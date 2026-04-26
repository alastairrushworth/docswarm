# AGENT.md — guardrails for Claude Code

You are the developer agent for this project. Read this file first, then `DESIGN.md` for full context.

## Hard rules — non-negotiable

- **Never read** `data/val/truth/` or anything under `data/test/`. These are not mounted into your container; do not try to obtain them via network, shell, git history of other branches, or any other channel.
- **Source of truth for tunables**: `config.yaml`. Hardcoded constants in code are forbidden. Read from config.
- **Source of truth for the schema**: `module/pdf_to_json/schema.py`. Treat this file (not `DESIGN.md` §3) as authoritative; revise it in `schema.py` if needed.
- **The deliverable is the module** in `module/pdf_to_json/`. Reasoning transcripts and scratch notes are not the product.

## What is fixed vs what you design

**Fixed contract** (do not change):
- Public entry point: `pdf_to_json(pdf_path: str) -> dict`.
- Schema source-of-truth: `module/pdf_to_json/schema.py`.
- Per-page wall-clock budget: `iteration.page_budget_seconds` from `config.yaml`. Module returns best-effort JSON on exhaustion; never raises.
- Cache keying: SHA-256 of PDF bytes (+ page + model + prompt version), not filename.
- Submission protocol with the judge (see below).

**Your design space** (rewrite freely):
- Everything else inside `module/pdf_to_json/`. The starter `pipeline.py` is a *starting point* so round 1 produces some score — it is not the recommended architecture. Replace it. Add files. Delete files. Reorganize. The starter does one vision call per page; that is unlikely to be sufficient on this corpus.

**Translator is expected to be LLM-heavy and agentic.** Likely shapes:
- Multi-pass extraction (segment articles → extract per article → verify → reconcile).
- Specialist sub-modules each driving their own LLM loops (verse detector, metadata extractor, continuation tracer, OCR-typo preserver).
- Model routing — different models for different sub-tasks.
- Self-checks where the translator re-prompts itself or a different model to verify a candidate output before emitting.
- Multiple LLM calls per page within the page budget.

The bar is whatever produces a good aggregate score. Single-prompt-per-page is a baseline, not a target.

## Iteration shape within a round

A round ends when you run the full `pdf_to_json` over the val set and submit broad eval to the judge. **Inside a round you have full freedom**: edit code, run unit tests, run the translator on one page or one PDF, ask the judge marking questions about specific JSON slices, iterate on prompts. Multiple rounds of marking feedback are expected before each broad submission.

- Use **marking mode** liberally — freeform questions about specific JSON slices. Cheap, frequent.
- Run **broad mode** once per round on the full validation set. This is the gate.
- Edit code, write tests for the new code, run them, and only when you have a hypothesis worth grading do you trigger the full re-extract + broad eval.

## Submission protocol (judge)

The judge runs in a separate container. Communicate via the filesystem:

- **Marking**: write `judge/inbox/{pdf_id}__marking__{ts}.json`. Read response from `judge/feedback/{same-name}.json`.
- **Broad**: write `judge/inbox/{pdf_id}__broad__round{n}.json`. Read response from `judge/feedback/{same-name}.json`.

### Marking request shape

```json
{
  "mode": "marking",
  "pdf_id": "issue_1892_06_03",
  "question": "I think article 4 is cut off. Is the body materially incomplete?",
  "focus": {
    "path": "articles[4].text",
    "value": ["paragraph 1...", "paragraph 2..."]
  }
}
```

- `focus.path` is a JSON path into your prediction: `articles[i]`, `articles[i].title`, `articles[i].text`, `articles[i].kind`, `articles[i].pages`, `magazine.editor`, `magazine.publisher.address`, `magazine.cost`, `magazine.issue.date`, etc.
- `focus.value` is the slice of your prediction at that path. Keep it small — there is an 8KB cap. Send what you want graded, not the whole document.
- `question` is freeform natural language.

### Marking response shape

```json
{
  "verdict": "correct" | "incomplete" | "wrong" | "unverifiable",
  "feedback": "≤80 words, JSON-relative guidance with no quoted truth",
  "suggested_focus_path": "articles[4].text" | null
}
```

The judge sees only your predicted JSON and the truth JSON — **not the PDF**. Feedback is JSON-relative ("the body is materially shorter than truth", "this should be verse not prose", "publisher address looks truncated"), never layout-relative ("look at the right column"). If you need layout reasoning, do it in the translator side by re-rendering the page yourself.

## Stop signals

Wall-clock and plateau detection per `config.yaml`. Do not implement other stop conditions. The plateau detector is intentionally inactive while `round < min_rounds_before_plateau` or `best < plateau_aggregate_floor` — first-round catastrophic failures should not stop the loop.

## Posture

Make best efforts to close every error. Graded scores are *not* permission to settle for low component scores — they are protection against indefinite blocking on a single hard issue. If a component score is stuck round-on-round, **change the approach** rather than retuning the same approach.

## Commit discipline

- Per-round commit, message format:
  `round {n}: aggregate {score:.3f} (Δ{delta:+.3f}); titles {x:.2f}, text {y:.2f}, order {z:.2f}, ...`
- Append a new entry to `trends/round_history.json` each round and commit.
- Push to the branch named in `config.yaml` (`repo.branch`).

## Coding conventions

- Type hints throughout.
- Pydantic for all IO (judge requests/responses, schema).
- Pytest for non-LLM helpers (alignment, leakage filter, cache key derivation, anything in the translator that is purely structural).
- No backwards-compatibility shims. No feature flags for hypothetical futures.

## Model selection

- Use names from `config.yaml.models`. You may revise these in early rounds and commit the change.
- Do **not** pull arbitrary new Ollama models that aren't in the snapshot without rebuilding it (`make build-snapshot`). Pulling at runtime on the H100/H200 wastes wall-clock budget.
- Concurrent requests against Ollama are how you saturate the GPU on this workload — set `OLLAMA_NUM_PARALLEL` and fan out where it makes sense.

## VRAM allocation

At startup:

1. Run `nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits`. Log result.
2. Choose a loadout. Log it.
3. Include the chosen loadout in the round trend entry as `model_loadout`.

## Caching

Cache LLM-call outputs that depend on PDF content across rounds.

- **Key**: SHA-256 of (PDF file bytes, page or region index, model tag, prompt template version, any other input that changes the output).
- **Not** filename. A filename-keyed cache will silently return stale output if a PDF is replaced.
- Cache lives at `paths.cache_dir`. Helper in `module/pdf_to_json/cache.py` — feel free to extend or replace.
- When you change a prompt, bump the prompt version so the cache invalidates cleanly.

## Per-page time budget

The module enforces `iteration.page_budget_seconds` per page. On exhaustion, return whatever partial output is available for that page and emit a warning. The module **always** returns *some* JSON conforming (best effort) to the schema — never raise out of `pdf_to_json`.

## Non-leakage (you are on the receiving end)

The judge will not transmit ground truth verbatim. Feedback is qualitative/structural. If you see what looks like a verbatim ground-truth phrase in feedback, treat it as a bug in the leakage filter and report it; do not exploit it.
