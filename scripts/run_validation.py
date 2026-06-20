"""Iteration loop on the droplet:
  round 1:           translate val PDFs (starter pipeline) → broad eval → commit
  round n (n>1):     invoke Claude Code with prior-round feedback → it edits code
                     and commits → harness re-runs broad eval

Claude Code is the developer agent. Between rounds the harness shells out to
`claude --print --permission-mode bypassPermissions` with a brief prompt that
hands the agent the latest broad feedback. Claude reads AGENT.md / DESIGN.md
itself, runs marking probes if useful, edits files in module/pdf_to_json/, and
commits before exiting. The harness then re-runs the broad evaluation.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("loop")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module"))

from pdf_to_json import pdf_to_json  # noqa: E402


def _load_cfg() -> dict[str, Any]:
    p = Path(os.environ.get("DOCSWARM_CONFIG", str(ROOT / "config.yaml")))
    with p.open() as f:
        return yaml.safe_load(f)


def _val_pdfs(cfg: dict) -> list[Path]:
    paths = cfg.get("paths", {})
    d = Path(paths.get("val_dir", str(ROOT / "data/val")))
    pdf_name = paths.get("pdf_filename", "original.pdf")
    if not d.is_dir():
        return []
    return sorted(
        sub / pdf_name for sub in d.iterdir() if sub.is_dir() and (sub / pdf_name).is_file()
    )


def _pdf_id(p: Path) -> str:
    # Each document lives in its own folder; the folder name is the id.
    return p.parent.name


def _submit_broad(cfg: dict, pdf_id: str, round_n: int, prediction: dict) -> dict:
    paths = cfg.get("paths", {})
    inbox = Path(paths.get("inbox_dir", str(ROOT / "judge/inbox")))
    feedback = Path(paths.get("feedback_dir", str(ROOT / "judge/feedback")))
    inbox.mkdir(parents=True, exist_ok=True)
    feedback.mkdir(parents=True, exist_ok=True)

    name = f"{pdf_id}__broad__round{round_n}__{uuid.uuid4().hex[:6]}.json"
    req = {
        "mode": "broad",
        "pdf_id": pdf_id,
        "round": round_n,
        "prediction": prediction,
    }
    (inbox / name).write_text(json.dumps(req))

    out = feedback / name
    deadline = time.monotonic() + 600.0
    while time.monotonic() < deadline:
        if out.is_file():
            try:
                return json.loads(out.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(0.5)
    raise TimeoutError(f"judge did not respond for {name} within 600s")


def _aggregate_per_pdf(feedbacks: list[dict]) -> dict[str, Any]:
    aggs = [f.get("aggregate", 0.0) for f in feedbacks]
    avg = sum(aggs) / len(aggs) if aggs else 0.0
    component_keys = ["schema_validity", "article_count", "precision", "metadata", "titles", "text", "order", "pages"]
    components: dict[str, float] = {}
    for k in component_keys:
        vals = [(f.get("components") or {}).get(k, {}).get("score", 0.0) for f in feedbacks]
        components[k] = sum(vals) / len(vals) if vals else 0.0
    return {"aggregate": avg, "components": components}


def _append_trend(cfg: dict, entry: dict) -> Path:
    p = Path(cfg.get("paths", {}).get("trends_file", str(ROOT / "trends/round_history.json")))
    p.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if p.is_file():
        try:
            history = json.loads(p.read_text())
        except json.JSONDecodeError:
            history = []
    history.append(entry)
    p.write_text(json.dumps(history, indent=2))
    return p


def _ensure_git_ssh(cfg: dict) -> None:
    """Make `git push` work over SSH from inside the container.

    Reads `repo.deploy_key_path` from config (default /secrets/deploy_key),
    copies it to a writable path with 0600 perms, and sets GIT_SSH_COMMAND.
    """
    if os.environ.get("GIT_SSH_COMMAND"):
        return
    src = Path(cfg.get("repo", {}).get("deploy_key_path", "/secrets/deploy_key"))
    if not src.is_file():
        logger.info("no deploy key at %s; git push will likely fail", src)
        return
    dst = Path("/tmp/deploy_key")
    try:
        dst.write_bytes(src.read_bytes())
        os.chmod(dst, 0o600)
    except OSError as e:
        logger.warning("could not stage deploy key: %s", e)
        return
    os.environ["GIT_SSH_COMMAND"] = (
        f"ssh -i {dst} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    )


def _git(*args: str, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


_NO_CHANGE = "(no code changes)"


def _code_change_summary() -> str:
    """One-line summary of the agent's uncommitted edits to the deliverable since
    the last round's commit. Recorded into the trend so the agent SEES that its
    rewrites moved (or, as has been the case, did not move) the score — the
    cross-round signal that breaks the Groundhog-Day loop where it re-derived the
    same edit every round with no memory of the last attempt's null result."""
    try:
        diff = _git("diff", "--stat", "HEAD", "--", "module/pdf_to_json", check=False).stdout.strip()
    except Exception:
        return _NO_CHANGE
    if not diff:
        return _NO_CHANGE
    # Keep the compact files-changed / insertions / deletions summary line.
    last = diff.splitlines()[-1].strip()
    files = [ln.split("|")[0].strip() for ln in diff.splitlines()[:-1] if "|" in ln]
    head = ", ".join(files[:4]) + (" …" if len(files) > 4 else "")
    return f"{head} ({last})" if head else last


def _invalidate_vision_cache(cfg: dict) -> int:
    """Delete the persistent per-page vision cache so the next translation reflects
    the agent's code/prompt edits. The cache lives on the network volume and at
    temperature=0 returns byte-identical pages, which silently masked every agent
    change for 9 rounds across 3 runs. Cleared only when code actually changed, so
    unchanged rounds still get the (correct, cheap) cache."""
    cache_dir = Path(cfg.get("paths", {}).get("cache_dir", str(ROOT / ".cache/pdf_to_json")))
    if not cache_dir.is_dir():
        return 0
    n = 0
    for p in cache_dir.glob("*.json"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


def _ensure_git_identity(cfg: dict) -> None:
    """Fresh pods have no git identity, so every per-round `git commit` died with
    `Author identity unknown` (exit 128) and no round progress was persisted.
    Set a repo-local identity (configurable via config.repo) if unset."""
    repo = cfg.get("repo", {})
    name = repo.get("git_user_name", "docswarm-agent")
    email = repo.get("git_user_email", "agent@docswarm.local")
    try:
        if not _git("config", "user.email", check=False).stdout.strip():
            _git("config", "user.email", email)
        if not _git("config", "user.name", check=False).stdout.strip():
            _git("config", "user.name", name)
    except subprocess.CalledProcessError as e:
        logger.warning("could not set git identity: %s", e)


def _commit_and_push(cfg: dict, message: str) -> None:
    branch = cfg.get("repo", {}).get("branch", "agent")
    try:
        _git("add", "-A")
        status = _git("status", "--porcelain").stdout.strip()
        if not status:
            logger.info("nothing to commit")
            return
        _git("commit", "-m", message)
        # Push failures were silently swallowed (check=False, stderr discarded),
        # so a non-pushing pod looked healthy while origin/development drifted and
        # the *next* pod's `git pull --ff-only` aborted on divergence. Surface it.
        push = _git("push", "origin", branch, check=False)
        if push.returncode != 0:
            logger.warning(
                "git push to origin/%s failed (rc=%d): %s",
                branch, push.returncode, (push.stderr or push.stdout).strip(),
            )
        else:
            logger.info("pushed commit to origin/%s", branch)
    except subprocess.CalledProcessError as e:
        logger.warning("git operation failed: %s\n%s", e, e.stderr)


def _component_short(entry: dict) -> str:
    c = entry["components"]
    return (
        f"titles {c['titles']:.2f}, text {c['text']:.2f}, order {c['order']:.2f}, "
        f"meta {c['metadata']:.2f}, count {c['article_count']:.2f}, "
        f"prec {c.get('precision', 0.0):.2f}, "
        f"schema {c['schema_validity']:.2f}, pages {c['pages']:.2f}"
    )


def _model_loadout(cfg: dict) -> str:
    m = cfg.get("models", {})
    parts = []
    for k in ("coder", "vision", "judge", "embedding"):
        if m.get(k):
            parts.append(f"{k}={m[k]}")
    return ", ".join(parts)


def _translate_and_submit(cfg: dict, round_n: int, p: Path) -> dict:
    pid = _pdf_id(p)
    logger.info("round %d: translating %s", round_n, pid)
    prediction = pdf_to_json(str(p))
    logger.info("round %d: submitting broad eval for %s", round_n, pid)
    fb = _submit_broad(cfg, pid, round_n, prediction)
    logger.info("round %d: %s aggregate=%.3f", round_n, pid, fb.get("aggregate", 0.0))
    return fb


def run_round(cfg: dict, round_n: int) -> dict[str, Any]:
    pdfs = _val_pdfs(cfg)
    if not pdfs:
        raise SystemExit(f"no validation PDFs found in {cfg['paths']['val_dir']}")

    pdf_concurrency = max(1, int(cfg.get("iteration", {}).get("pdf_concurrency", 1)))

    # The agent has just edited the deliverable (round > 1). If it changed
    # anything, bust the persistent vision cache so this round's translation
    # actually reflects the edit — otherwise temp=0 cache hits freeze the output.
    code_change = _code_change_summary()
    if code_change != _NO_CHANGE:
        cleared = _invalidate_vision_cache(cfg)
        logger.info("round %d: deliverable changed [%s] — cleared %d cached page(s)",
                    round_n, code_change, cleared)

    t0 = time.monotonic()
    feedbacks_by_pdf: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=pdf_concurrency) as ex:
        futs = {ex.submit(_translate_and_submit, cfg, round_n, p): p for p in pdfs}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                feedbacks_by_pdf[_pdf_id(p)] = fut.result()
            except Exception as e:
                logger.warning("round %d: %s failed: %s", round_n, _pdf_id(p), e)
                feedbacks_by_pdf[_pdf_id(p)] = {"aggregate": 0.0, "components": {}}

    feedbacks = [feedbacks_by_pdf[_pdf_id(p)] for p in pdfs]
    agg = _aggregate_per_pdf(feedbacks)
    elapsed = time.monotonic() - t0

    entry = {
        "round": round_n,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_clock_seconds": round(elapsed, 1),
        "aggregate": round(agg["aggregate"], 4),
        "components": {k: round(v, 4) for k, v in agg["components"].items()},
        "code_change": code_change,
        "model_loadout": _model_loadout(cfg),
        "per_pdf": [
            {"pdf_id": _pdf_id(p), "aggregate": fb.get("aggregate", 0.0)}
            for p, fb in zip(pdfs, feedbacks)
        ],
        "categorical_errors_per_pdf": [
            {"pdf_id": _pdf_id(p), "errors": fb.get("categorical_errors", [])}
            for p, fb in zip(pdfs, feedbacks)
        ],
        "hints_per_pdf": [
            {"pdf_id": _pdf_id(p), "hints": fb.get("hints", [])}
            for p, fb in zip(pdfs, feedbacks)
        ],
    }
    _append_trend(cfg, entry)
    return entry


def _read_notes(cfg: dict) -> str:
    p = Path(cfg.get("paths", {}).get("notes_file", str(ROOT / "notes/approach.md")))
    if p.is_file():
        return p.read_text().strip()
    return "(empty — no notes recorded yet)"


def _trend_summary(cfg: dict) -> str:
    """Compact one-line-per-round history so the agent sees the whole trajectory,
    not just the last round."""
    p = Path(cfg.get("paths", {}).get("trends_file", str(ROOT / "trends/round_history.json")))
    if not p.is_file():
        return "(no rounds graded yet)"
    try:
        history = json.loads(p.read_text())
    except json.JSONDecodeError:
        return "(trend file unreadable)"
    lines = []
    prev_agg = None
    for e in history:
        c = e.get("components", {})
        agg = e.get("aggregate", 0.0)
        delta = f" Δ{agg - prev_agg:+.3f}" if prev_agg is not None else ""
        prev_agg = agg
        change = e.get("code_change")
        change_str = f"  ← changed {change}{delta}" if change and change != _NO_CHANGE else ""
        lines.append(
            f"  round {e.get('round')}: agg={agg:.3f} "
            f"[titles {c.get('titles', 0.0):.2f}, text {c.get('text', 0.0):.2f}, "
            f"order {c.get('order', 0.0):.2f}, meta {c.get('metadata', 0.0):.2f}, "
            f"count {c.get('article_count', 0.0):.2f}, prec {c.get('precision', 0.0):.2f}, "
            f"schema {c.get('schema_validity', 0.0):.2f}, pages {c.get('pages', 0.0):.2f}]"
            f"{change_str}"
        )
    return "\n".join(lines) if lines else "(no rounds graded yet)"


def _format_hints(hints_per_pdf: list[dict]) -> str:
    """Render the judge's per-component guidance hints for the prompt. These are
    non-prescriptive — they point at what/where to investigate, not the fix."""
    blocks = []
    for entry in hints_per_pdf:
        hints = entry.get("hints") or []
        if not hints:
            continue
        lines = "\n".join(f"  - {h}" for h in hints)
        blocks.append(f"{entry.get('pdf_id')}:\n{lines}")
    return "\n".join(blocks) if blocks else "(no hints — components at full credit)"


def _best_status(prev_entry: dict, best_entry: dict | None) -> str:
    prev_agg = prev_entry.get("aggregate", 0.0)
    if best_entry is None or best_entry.get("round") == prev_entry.get("round"):
        return f"Round {prev_entry.get('round')} ({prev_agg:.3f}) is your best so far. Build on it."
    best_agg = best_entry.get("aggregate", 0.0)
    return (
        f"WARNING: round {prev_entry.get('round')} ({prev_agg:.3f}) REGRESSED vs your best, "
        f"round {best_entry.get('round')} ({best_agg:.3f}). Consider reverting that change "
        f"(`git revert`/`git checkout`) or trying a materially different approach — do not "
        f"keep iterating on a path that lost ground."
    )


def _plateau_directive(rounds_since_best: int) -> str:
    """Escalation injected when the aggregate has stalled. The agent has been
    re-deriving the same incremental edit each round (Δ0.000) with no memory of
    the prior null result; this forces an approach change and a train-score check
    that the edit actually alters output before it is committed."""
    if rounds_since_best < 2:
        return ""
    return (
        f"\n*** PLATEAU: {rounds_since_best} round(s) with no aggregate improvement. ***\n"
        "Your recent edits (see the 'changed …' annotations in the trajectory) did NOT move "
        "the score. Do not make another incremental edit to the same code path — that has "
        "repeatedly produced Δ0.000. Instead:\n"
        "  1. Pick the SINGLE weakest component from the trajectory.\n"
        "  2. Change the APPROACH wholesale, not the wording — e.g. replace the regex noise "
        "filter with a different segmentation strategy; change how/where the masthead region "
        "is located and parsed; or correct a systematic page-numbering offset.\n"
        "  3. Run `python scripts/score_train.py` and CONFIRM the train self-score changed "
        "before committing. An identical train score means your edit had no effect and will "
        "plateau again — keep iterating until output actually moves.\n"
        "  4. Record the new hypothesis explicitly in your notes so you do not repeat this.\n"
    )


def _developer_agent_prompt(
    cfg: dict, round_n: int, prev_entry: dict, best_entry: dict | None,
    rounds_since_best: int = 0,
) -> str:
    components = prev_entry.get("components", {})
    per_pdf = prev_entry.get("per_pdf", [])
    cat_errors = prev_entry.get("categorical_errors_per_pdf", [])
    hints_block = _format_hints(prev_entry.get("hints_per_pdf", []))
    notes = _read_notes(cfg)
    history = _trend_summary(cfg)
    best_status = _best_status(prev_entry, best_entry)
    directive = _plateau_directive(rounds_since_best)
    notes_path = cfg.get("paths", {}).get("notes_file", "notes/approach.md")
    return f"""You are the developer agent for docswarm. This is round {round_n}.

Read AGENT.md first if you have not. The deliverable is `module/pdf_to_json/`;
the public entry point `pdf_to_json(pdf_path: str) -> dict` is fixed but
everything inside is yours to rewrite. You may build the translator as
specialist passes OR as tool-using sub-agents — your call. See AGENT.md
"Specialist agents & tools".

=== YOUR NOTES (durable across rounds — `{notes_path}`) ===
This is your memory. It is the accumulated record of what you have tried, what
worked, what failed, and what to try next. Read it before deciding anything.
{notes}
=== END NOTES ===

Score trajectory so far:
{history}

{best_status}
{directive}
Latest round ({round_n - 1}) detail:
- aggregate (weighted mean across val PDFs): {prev_entry.get('aggregate', 0.0):.3f}
- components: {json.dumps(components)}
- per-pdf aggregates: {json.dumps(per_pdf)}
- categorical errors per pdf: {json.dumps(cat_errors)}

Judge hints (non-prescriptive — they tell you what/where to investigate, never
the fix or the expected value; use them to improve the translator process):
{hints_block}

Your task this turn:
1. Read your notes and the trajectory. Identify the weakest component(s) and
   form a hypothesis. If a component has been stuck for several rounds, change
   the approach rather than retuning the same one.
2. Develop and self-check against the TRAIN pairs, which you ARE allowed to read
   with ground truth: `python scripts/score_train.py` runs your translator over
   `data/train/` and scores it with the same broad metric the judge uses. This
   is your fast inner loop — iterate here before relying on the val gate.
3. Optionally probe the judge in marking mode by writing JSON to
   `judge/inbox/{{pdf_id}}__marking__{{tag}}.json` and reading the matching
   file from `judge/feedback/`. Marking shape is in AGENT.md.
4. Edit code in `module/pdf_to_json/` (or add files/tools/sub-agents there) to
   address the weakness. Keep the public entry point and schema source-of-truth.
5. Write a quick unit test for any new non-LLM helper and run `pytest`.
6. UPDATE YOUR NOTES at `{notes_path}`: record this round's hypothesis, what you
   changed, the train self-score you observed, and what to try next. This is the
   only state that survives to the next round besides your committed code.
7. `git add` your changes (including the notes file), commit with a clear
   message, and exit.

The harness will run the next broad evaluation as soon as you exit. Do NOT
attempt to run broad eval yourself. Do NOT read the ground truth
`data/val/<doc>/transcribed.json` (co-located with the PDFs) or anything under
`data/test/` — reading val/test truth is a leakage violation even though val is
mounted. (Train truth is fair game — that is what `score_train.py` uses.)
"""


def _run_developer_agent(
    cfg: dict, round_n: int, prev_entry: dict, best_entry: dict | None,
    rounds_since_best: int = 0,
) -> None:
    if shutil.which("claude") is None:
        logger.warning("claude CLI not on PATH; skipping developer-agent turn")
        return
    model = cfg.get("models", {}).get("coder", "qwen3.6:35b")
    prompt = _developer_agent_prompt(cfg, round_n, prev_entry, best_entry, rounds_since_best)
    logger.info("round %d: invoking Claude Code (model=%s, local Ollama)", round_n, model)
    cmd = [
        "claude", "--print",
        "--permission-mode", "bypassPermissions",
        "--model", model,
    ]
    # The pod runs as root, where Claude Code refuses bypassPermissions
    # ("--dangerously-skip-permissions cannot be used with root/sudo privileges")
    # and exits 1 before doing anything — silently turning every round into a
    # no-op. IS_SANDBOX=1 is the documented escape hatch for disposable
    # containers like this ephemeral pod. The 64k output cap stays as a backstop.
    #
    # The coder (qwen3.6:35b) runs on local Ollama and is a reasoning model: left
    # to think it pours a runaway chain-of-thought into the response and blows the
    # output cap (observed "exceeded the 64000 output token maximum", exit 1,
    # Δ+0.000, no edit made). MAX_THINKING_TOKENS=0 disables extended thinking so
    # it emits the edit directly — the analogue of vision_think=false on the
    # vision path.
    env = {
        **os.environ,
        "IS_SANDBOX": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
        "MAX_THINKING_TOKENS": "0",
    }
    rc = subprocess.run(
        cmd, cwd=ROOT, check=False, input=prompt, text=True, env=env
    ).returncode
    if rc != 0:
        logger.warning("claude exited with code %d", rc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-git", action="store_true", help="skip git commit/push")
    parser.add_argument("--single-round", action="store_true", help="run one round and exit")
    parser.add_argument("--no-agent", action="store_true",
                        help="skip the Claude Code turn between rounds (baseline only)")
    args = parser.parse_args()

    cfg = _load_cfg()
    if not args.no_git:
        _ensure_git_ssh(cfg)
        _ensure_git_identity(cfg)
    iter_cfg = cfg.get("iteration", {})
    wall_clock_s = float(iter_cfg.get("wall_clock_hours", 12)) * 3600
    epsilon = float(iter_cfg.get("epsilon", 0.005))
    min_rounds = int(iter_cfg.get("min_rounds_before_plateau", 3))
    plateau_floor = float(iter_cfg.get("plateau_aggregate_floor", 0.30))
    plateau_window = int(iter_cfg.get("plateau_window", 3))

    deadline = time.monotonic() + wall_clock_s
    best = -math.inf
    rounds_since_best = 0
    round_n = 0
    prev_aggregate = None
    prev_entry: dict[str, Any] | None = None
    best_entry: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        round_n += 1

        if round_n > 1 and prev_entry is not None and not args.no_agent:
            _run_developer_agent(cfg, round_n, prev_entry, best_entry, rounds_since_best)

        entry = run_round(cfg, round_n)
        agg = entry["aggregate"]
        delta = (agg - prev_aggregate) if prev_aggregate is not None else 0.0

        message = (
            f"round {round_n}: aggregate {agg:.3f}"
            + (f" (Δ{delta:+.3f})" if prev_aggregate is not None else " (baseline)")
            + f"; {_component_short(entry)}"
        )
        logger.info(message)
        if not args.no_git:
            _commit_and_push(cfg, message)

        if agg > best + epsilon:
            best = agg
            rounds_since_best = 0
        else:
            rounds_since_best += 1
        # Track the best entry for regression-awareness in the next prompt.
        if best_entry is None or agg > best_entry.get("aggregate", -math.inf):
            best_entry = entry

        prev_aggregate = agg
        prev_entry = entry

        plateau_active = round_n >= min_rounds and best >= plateau_floor
        if plateau_active and rounds_since_best >= plateau_window:
            logger.info("plateau: stopping after round %d", round_n)
            break

        if args.single_round:
            break

    logger.info("loop finished after round %d (best aggregate=%.3f)", round_n, best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
