# DocSwarm Process Flow

## High-Level Pipeline

```mermaid
flowchart LR
    START((Start)) --> NEXT["get_next_unstudied_page()"]
    NEXT -->|"page dict"| CLASSIFY

    CLASSIFY["<b>Classify page</b><br/><i>Vision LLM tool call</i><br/><br/>classify_page_content()"]

    CLASSIFY -->|"advertisement"| MARK["log_page_study()"]
    CLASSIFY -->|"editorial / mixed"| RESEARCH

    RESEARCH["<b>Researcher</b><br/><i>LLM call (JSON mode)</i><br/><br/>Extracts entity dicts<br/>from page text"]

    RESEARCH -->|"list of entity dicts"| LOOP

    LOOP["<b>For each entity</b>"]
    LOOP --> READ_EXISTING["Read existing<br/>wiki article (if any)"]
    READ_EXISTING --> WRITER

    WRITER["<b>Writer</b><br/><i>LLM call per entity</i><br/><br/>Creates or updates<br/>article markdown"]

    WRITER -->|"article text"| WRITE["Write to<br/>wiki/type/slug.md"]
    WRITE --> LOOP

    LOOP -->|"all entities done"| MARK
    MARK --> NEXT
    NEXT -->|"no more pages"| DONE((Done))

    subgraph LLM["LLM Backend (USE_OLLAMA)"]
        direction TB
        OLLAMA["Ollama (JSON: format=json)"]
        OPENAI["OpenAI (response_format=json_object)"]
    end

    LLM -.-> CLASSIFY
    LLM -.-> RESEARCH
    LLM -.-> WRITER
```

---

## Researcher (detail)

A single LLM call in **JSON mode** — no tools, no agents. The page text is already in the input message.

```mermaid
flowchart LR
    IN["<b>Input:</b><br/>System: RESEARCHER_PROMPT<br/>User: page text + metadata"]

    IN -->|"LLM call<br/>(JSON mode)"| LLM["<b>LLM</b><br/>Reads page text,<br/>returns JSON"]

    LLM --> OUT

    OUT["<b>Output:</b><br/>{entities: [{entity, type, info, source}, ...]}"]
```

Example output:
```json
{
  "entities": [
    {
      "entity": "Reg Harris",
      "type": "person",
      "info": "British track cycling champion who won multiple world sprint titles.",
      "source": "Cycling Weekly Vol.12, p.3"
    }
  ]
}
```

---

## Writer (detail)

Called **once per entity** in a loop. Receives the entity dict from the researcher plus any existing article content.

```mermaid
flowchart LR
    IN["<b>Input:</b><br/>System: WRITER_PROMPT<br/>User: entity dict<br/>+ existing article (if any)"]

    IN -->|"LLM call"| LLM["<b>LLM</b><br/>Writes new article or<br/>merges into existing"]

    LLM --> OUT["<b>Output:</b><br/>Article markdown"]

    OUT -->|"Python writes file"| FILE["wiki/type/slug.md<br/>(YAML front matter + body)"]
```

---

## Summary

| Step | Type | Input | Output |
|------|------|-------|--------|
| **Classify** | Tool call (vision LLM) | Page image + OCR | `advertisement` / `editorial` / `mixed` |
| **Researcher** | LLM (JSON mode) | Page text | `{entities: [{entity, type, info, source}]}` |
| **Writer** (per entity) | LLM | Entity dict + existing article | Article markdown |
| **File write** | Python | Article text | `wiki/type/slug.md` |

### LLM Backend

Controlled by `USE_OLLAMA` env var:

| Setting | Researcher JSON mode | Writer | Classification |
|---------|---------------------|--------|----------------|
| `USE_OLLAMA=true` | `ChatOllama(format="json")` | `ChatOllama` | Raw `/api/generate` with vision |
| `USE_OLLAMA=false` | `ChatOpenAI(response_format=json_object)` | `ChatOpenAI` | OpenAI chat completions with vision |
