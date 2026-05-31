# AGENT.md — guardrails for Claude Code

You are the developer agent for this project. Read this file first, then `DESIGN.md` for full context.

## Hard rules — non-negotiable

- **Never read** the ground-truth `data/val/<doc>/transcribed.json` files (co-located with each `original.pdf`) or anything under `data/test/`. `data/test/` is not mounted; the val truth files are mounted alongside the PDFs but reading them is a leakage violation — do not open them, and do not try to obtain test data via network, shell, git history of other branches, or any other channel.
- **Source of truth for tunables**: `config.yaml`. Hardcoded constants in code are forbidden. Read from config.
- **Source of truth for the schema**: `module/pdf_to_json/schema.py`. Treat this file (not `DESIGN.md` §3) as authoritative; revise it in `schema.py` if needed.
- **The deliverable is the module** in `module/pdf_to_json/`. Throwaway scratch (one-off probe scripts, debug dumps) is not the product — but durable approach notes ARE expected (see "Your memory across rounds" below). Keep the two separate: the module is the artifact; the notes are how you get there.

## What is fixed vs what you design

**Fixed contract** (do not change):
- Public entry point: `pdf_to_json(pdf_path: str) -> dict`.
- Schema source-of-truth: `module/pdf_to_json/schema.py`.
- The module never raises out of `pdf_to_json` — return best-effort JSON on any failure.
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

## Specialist agents & tools

The translator's internal shape is **your call** — the only fixed thing is the `pdf_to_json(pdf_path) -> dict` entry point. Two shapes are explicitly blessed; you may pick either, mix them, or change your mind between rounds:

1. **Specialist passes / sub-modules** — plain Python functions, each owning one concern (verse detection, metadata extraction, continuation tracing, OCR-typo preservation) and driving its own LLM prompts. Simplest; works with the starter `ollama_client`.
2. **Tool-using sub-agents** — a runtime agent loop where a model is given tools and decides which to call. If you go this way:
   - Put runtime tools in `module/pdf_to_json/tools/` and specialist-agent definitions/prompts in `module/pdf_to_json/agents/` (create these dirs). They are part of the deliverable and must be committed.
   - **The starter `ollama_client.generate` only calls `/api/generate` (single prompt + images) — it cannot do tool-calling.** You must extend the client to Ollama's `/api/chat` with a `tools=` array (qwen3.6 supports tool calls) before any runtime tool loop will work. This is allowed and expected if you choose this path.

You may also create **Claude Code sub-agents for your own development work** (not the deliverable) by committing `.claude/agents/*.md` definitions — e.g. a prompt-tuning agent or a schema agent. These persist across rounds like any committed file.

Prefer the simplest shape that hits the score. Don't build an agent framework speculatively; reach for it when a specialist pass plateaus.

## How rounds work

The harness (`scripts/run_validation.py`) owns the round loop. Each round:

1. The harness runs `pdf_to_json` over the val set and submits broad eval to the judge.
2. The harness commits the new `trends/round_history.json` entry.
3. The harness invokes you (`claude --print`) with a brief that includes the round-N broad feedback — component scores **and per-component hints** that point at what/where to investigate (never the fix). Treat the hints as your primary to-do list for the round.
4. You investigate, optionally probe the judge in marking mode, edit code, commit, and exit.
5. The harness starts round N+1.

**Inside your turn you have full freedom**: edit any file under `module/pdf_to_json/`, run unit tests, run the translator on one page or one PDF, ask the judge marking questions about specific JSON slices, iterate on prompts. Do **not** attempt to run broad eval yourself or call `pdf_to_json` against the val set with intent to submit — the harness will do that as soon as you exit.

- Use **marking mode** liberally — freeform questions about specific JSON slices. Cheap, frequent.
- Edit code, write tests for the new code, run them. When you have something worth grading, commit and exit.

### Your fast inner loop: TRAIN self-scoring

You can read the **train** pairs *with* ground truth (`data/train/<doc>/{original.pdf,transcribed.json}`) — only val/test truth is off-limits. Use this. `python scripts/score_train.py` runs your translator over all train docs and scores them with the **same** `judge.broad` metric and `config.yaml` weights the val gate uses, so the numbers are directly comparable to the round trend. Iterate against train (many cheap cycles) before you rely on the once-per-round val gate over 3 noisy PDFs. Pass doc-ids to score a subset: `python scripts/score_train.py the-bearings-vol5-18`.

### Your memory across rounds

Each round the harness starts you **fresh** — a new process with no recollection of prior rounds except (a) the committed code and (b) your notes file at `notes/approach.md`. The notes file is injected verbatim into your prompt every round, along with the full score trajectory. **It is your only working memory.** Treat maintaining it as part of the job:

- Read it first, before deciding what to do.
- Each round, record: the hypothesis you tested, what you changed, the train self-score and/or val result you observed, and what to try next.
- Log **dead-ends** explicitly so you don't burn future rounds re-trying them.
- Commit `notes/approach.md` alongside your code changes.

If the harness tells you the last round regressed vs your best, take it seriously: revert or change approach rather than digging deeper into a losing path.

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
  "verdict": "equivalent" | "minor_format_diff" | "partially_present" | "materially_different" | "missing" | "unverifiable",
  "feedback": "≤80 words, JSON-relative guidance, no quoted truth, no corrected value",
  "suggested_focus_path": "articles[4].text" | null
}
```

The judge grades **essence, not form**. It will not mark you down for punctuation, capitalization, abbreviation, OCR-style character noise, paragraph/line chunking, or reasonable segmentation/ordering choices — those return `equivalent`/`minor_format_diff`. It *will* flag missing or truncated content, wrong values, and verse-as-prose (`partially_present`/`materially_different`/`missing`). The judge sees only your predicted JSON and the truth JSON — **not the PDF**. Feedback is JSON-relative ("the body is materially shorter than expected", "this should be verse not prose", "publisher address looks truncated"), never layout-relative ("look at the right column") and never the corrected value. If you need layout reasoning, do it in the translator by re-rendering the page yourself.

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

## Resilience

The module **always** returns *some* JSON conforming (best effort) to the schema — never raise out of `pdf_to_json`. On per-call timeout or extraction failure, fill what you can and continue.

## Non-leakage (you are on the receiving end)

The judge will not transmit ground truth verbatim. Feedback is qualitative/structural. If you see what looks like a verbatim ground-truth phrase in feedback, treat it as a bug in the leakage filter and report it; do not exploit it.
