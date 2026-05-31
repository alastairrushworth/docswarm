# Approach notes (developer agent's durable memory)

This file persists across iteration rounds and is injected into your prompt at
the start of every round. It is the only working memory you carry forward
besides your committed code. Keep it current and honest — future-you relies on
it to avoid repeating dead-ends.

Recommended structure (edit freely):

## Current strategy
One paragraph: what architecture the translator currently uses and why.

## What has been tried
Round-by-round log. For each attempt: hypothesis → change → train self-score
(`scripts/score_train.py`) and/or val result → verdict (kept / reverted / why).

## Open problems / next ideas
Ranked list of what to try next and what evidence would confirm/refute it.

## Dead-ends (do not retry without new information)
Approaches that demonstrably did not work, with the reason.

---

(Round 1 runs the starter pipeline with no agent turn. Begin filling this in
from round 2 onward.)
