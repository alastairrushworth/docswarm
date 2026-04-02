# DocSwarm Process Flow

## High-Level Pipeline

```mermaid
flowchart LR
    subgraph KEY["Data Sources (colour key)"]
        direction TB
        DB[(DuckDB)]
        PDF["PDF Pages"]
        WIKI_IN["wiki/*.md"]
    end

    START((Start)) --> NEXT["get_next_unstudied_page()"]
    NEXT -->|"page dict"| RESEARCHER

    RESEARCHER["<b>Researcher</b><br/><i>ReAct LLM agent</i><br/>8 tools<br/><br/>Classifies page,<br/>extracts entities,<br/>saves to DB"]

    RESEARCHER -->|"all messages"| ROUTE{"Classification?"}

    ROUTE -->|"advertisement"| MARK["log_page_study()"]
    ROUTE -->|"editorial / mixed"| WRITER

    WRITER["<b>Writer</b><br/><i>ReAct LLM agent</i><br/>4 tools<br/><br/>Drafts wiki articles<br/>for each entity"]

    WRITER -->|"all messages"| EDITOR

    EDITOR["<b>Editor</b><br/><i>Deterministic Python</i><br/>no LLM<br/><br/>Parses output,<br/>writes markdown files"]

    EDITOR -->|"new articles"| WIKI_IN
    EDITOR --> MARK
    MARK --> NEXT
    NEXT -->|"no more pages"| DONE((Done))

    subgraph LLM["LLM Backend"]
        direction TB
        OLLAMA["Ollama"]
        OPENAI["OpenAI"]
    end

    LLM -.->|"USE_OLLAMA"| RESEARCHER
    LLM -.->|"USE_OLLAMA"| WRITER

    %% Styling
    classDef srcDb fill:#f0ad4e,stroke:#c77f1a,color:#000
    classDef srcPdf fill:#e74c3c,stroke:#c0392b,color:#fff
    classDef srcWiki fill:#1abc9c,stroke:#16a085,color:#fff
    classDef agent fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef deterministic fill:#5cb85c,stroke:#3d8b3d,color:#fff
    classDef decision fill:#9b59b6,stroke:#7d3c98,color:#fff

    class DB srcDb
    class PDF srcPdf
    class WIKI_IN srcWiki
    class RESEARCHER,WRITER agent
    class EDITOR deterministic
    class ROUTE decision
```

---

## Researcher (detail)

The researcher runs as a **single ReAct LLM invoke()** — the LLM autonomously decides which tools to call and in what order. The steps below are the typical sequence.

```mermaid
flowchart TD
    IN["<b>Input:</b> page text, image path, metadata"]

    IN --> S1

    S1["<b>1. Classify page</b><br/><i>Tool:</i> classify_page_content<br/><i>How:</i> page image + OCR text &rarr; vision LLM<br/><i>Returns:</i> editorial / advertisement / mixed"]

    S1 -->|"classification label"| S2

    S2["<b>2. Read page content</b><br/><i>Tools:</i> get_page_text, search_chunks<br/><i>How:</i> SQL queries on chunks table<br/><i>Returns:</i> full OCR text + relevant chunks"]

    S2 -->|"page text + chunks"| S3

    S3["<b>3. Identify entities</b><br/><i>How:</i> LLM reads text, identifies<br/>people, places, organisations, events,<br/>objects, concepts<br/><i>Returns:</i> candidate entity list with types"]

    S3 -->|"entity names + types"| S4

    S4["<b>4. Save entities</b><br/><i>Tools:</i> search_entities, save_entity<br/><i>How:</i> check for duplicates first,<br/>then SQL insert per entity<br/><i>Returns:</i> entity IDs stored in DB"]

    S4 -->|"saved entity IDs"| S5

    S5["<b>5. Check existing articles</b><br/><i>Tool:</i> search_article_files<br/><i>How:</i> glob + read wiki/*.md<br/><i>Returns:</i> which entities already<br/>have articles vs need new ones"]

    S5 --> OUT["<b>Output:</b> messages with classification<br/>+ entities saved in DB"]

    %% Styling
    classDef srcDb fill:#f0ad4e,stroke:#c77f1a,color:#000
    classDef srcWiki fill:#1abc9c,stroke:#16a085,color:#fff
    classDef srcMixed fill:#e74c3c,stroke:#f0ad4e,color:#fff,stroke-width:4px
    classDef srcLlm fill:#4a90d9,stroke:#2c5f8a,color:#fff

    class S1 srcMixed
    class S2 srcDb
    class S3 srcLlm
    class S4 srcDb
    class S5 srcWiki
```

---

## Writer (detail)

The writer also runs as a **single ReAct LLM invoke()**, receiving the full message history from the researcher.

```mermaid
flowchart TD
    IN["<b>Input:</b> researcher messages + entity list"]

    IN --> S1

    S1["<b>1. Gather source material</b><br/><i>Tools:</i> search_chunks, get_page_text<br/><i>How:</i> SQL queries for each entity<br/><i>Returns:</i> relevant passages from corpus"]

    S1 -->|"source passages"| S2

    S2["<b>2. Check existing articles</b><br/><i>Tools:</i> search_article_files, read_article_file<br/><i>How:</i> glob + read wiki/*.md<br/><i>Returns:</i> existing content to avoid duplication"]

    S2 -->|"all context gathered"| S3

    S3["<b>3. Draft wiki articles</b><br/><i>How:</i> LLM generates article text<br/>with inline citations &lt;sup&gt;[N]&lt;/sup&gt;<br/>and ## References section<br/><i>Returns:</i><br/>=== ARTICLE: Name ===<br/>..body..<br/>=== END ARTICLE ==="]

    S3 --> OUT["<b>Output:</b> all messages including<br/>delimited article blocks"]

    %% Styling
    classDef srcDb fill:#f0ad4e,stroke:#c77f1a,color:#000
    classDef srcWiki fill:#1abc9c,stroke:#16a085,color:#fff
    classDef srcLlm fill:#4a90d9,stroke:#2c5f8a,color:#fff

    class S1 srcDb
    class S2 srcWiki
    class S3 srcLlm
```

---

## Editor (detail)

The editor is **deterministic Python** — no LLM calls. It receives the full message history and writes files to disk.

```mermaid
flowchart TD
    IN["<b>Input:</b> all messages (researcher + writer)<br/>+ entity list from DB"]

    IN -->|"regex scan all AI messages"| S1

    S1["<b>1. Parse article blocks</b><br/><i>Primary:</i> regex for === ARTICLE === delimiters<br/><i>Fallback:</i> markdown heading boundaries<br/><i>Returns:</i> dict of title &rarr; body"]

    S1 -->|"raw blocks"| S2

    S2["<b>2. Filter</b><br/><i>Remove:</i> meta-content headings<br/>(entity summaries, page analysis)<br/><i>Remove:</i> masthead entities<br/>(editor, publisher, photographer)<br/><i>Returns:</i> cleaned entities + blocks"]

    S2 -->|"filtered"| S3

    S3["<b>3. Match entities to articles</b><br/><i>How:</i> normalize to lowercase alphanumeric,<br/>try exact match, then substring match<br/><i>Returns:</i> matched pairs + unmatched remainder"]

    S3 -->|"matched + unmatched"| S4

    S4["<b>4. Write markdown files</b><br/><i>Phase a:</i> entity + matched block &rarr; full article<br/><i>Phase b:</i> unmatched block &rarr; infer type from DB, write<br/><i>Phase c:</i> unmatched entity &rarr; stub from context text<br/><br/><i>Each file:</i> YAML front matter + body<br/><i>Path:</i> wiki/entity_type/slug.md"]

    S4 --> OUT["<b>Output:</b> written article paths"]

    %% Styling
    classDef deterministic fill:#5cb85c,stroke:#3d8b3d,color:#fff

    class S1,S2,S3,S4 deterministic
```

---

## Summary Tables

| Stage | Type | Tools | Input | Output |
|-------|------|-------|-------|--------|
| **Researcher** | ReAct LLM | 8 | Page text + image | Classification + entities in DB |
| **Router** | Deterministic | 0 | ToolMessage from classify | `"writer"` or `END` |
| **Writer** | ReAct LLM | 4 | Researcher messages | `=== ARTICLE ===` delimited blocks |
| **Editor** | Deterministic Python | 0 | All messages + DB entities | Markdown files in `wiki/` |

### LLM Backend

Controlled by `USE_OLLAMA` env var:

| Setting | Agent LLM | Classification |
|---------|-----------|---------------|
| `USE_OLLAMA=true` | `ChatOllama` (local) | Raw `/api/generate` with vision |
| `USE_OLLAMA=false` | `ChatOpenAI` | OpenAI chat completions with vision |
