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

    RESEARCHER["<b>Researcher</b><br/><i>ReAct LLM agent</i><br/>1 tool<br/><br/>Classifies page,<br/>extracts structured<br/>entity:info:source triples"]

    RESEARCHER -->|"all messages"| ROUTE{"Classification?"}

    ROUTE -->|"advertisement"| MARK["log_page_study()"]
    ROUTE -->|"editorial / mixed"| WRITER

    WRITER["<b>Writer</b><br/><i>ReAct LLM agent</i><br/>4 tools<br/><br/>For each entity: checks wiki,<br/>creates or updates article"]

    WRITER -->|"all messages"| EDITOR

    EDITOR["<b>Editor</b><br/><i>Deterministic Python</i><br/>no LLM<br/><br/>Parses article blocks,<br/>writes/updates markdown files"]

    EDITOR -->|"new / updated articles"| WIKI_IN
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

The researcher runs as a **single ReAct LLM invoke()**. It has one tool (`classify_page_content`) and the page text is already in the input message. After classification, it reads the page text and outputs structured entity:info:source blocks — no database reads or writes.

```mermaid
flowchart LR
    IN["<b>Input:</b> page text,<br/>image path, metadata"]

    IN --> S1

    S1["<b>1. Classify page</b><br/><i>Tool:</i> classify_page_content<br/><i>How:</i> page image + OCR text &rarr; vision LLM<br/><i>Returns:</i> editorial / advertisement / mixed"]

    S1 -->|"advertisement"| OUT_AD["<b>Output:</b> classification only<br/>(pipeline stops here)"]
    S1 -->|"editorial / mixed"| S2

    S2["<b>2. Extract entity information</b><br/><i>How:</i> LLM reads page text (already in input),<br/>outputs one block per notable subject<br/>with encyclopedia-ready info + source ref<br/><i>Filters out:</i> masthead staff, generic terms,<br/>ad-only mentions"]

    S2 --> OUT

    OUT["<b>Output:</b> structured blocks:<br/>=== ENTITY: Name (type) ===<br/>info + Source: title, p.N<br/>=== END ENTITY ==="]

    %% Styling
    classDef srcMixed fill:#e74c3c,stroke:#f0ad4e,color:#fff,stroke-width:4px
    classDef srcLlm fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef srcStop fill:#999,stroke:#666,color:#fff

    class S1 srcMixed
    class S2 srcLlm
    class OUT_AD srcStop
```

---

## Writer (detail)

The writer runs as a **single ReAct LLM invoke()**, receiving the full message history from the researcher (including entity blocks). For each entity, the writer decides whether to **create** a new article or **update** an existing one.

```mermaid
flowchart LR
    IN["<b>Input:</b> researcher messages<br/>+ entity blocks"]

    IN --> S1

    S1["<b>1. Check existing articles</b><br/><i>Tools:</i> search_article_files, read_article_file<br/><i>How:</i> glob + read wiki/*.md<br/><i>Returns:</i> existing content (if any) per entity"]

    S1 -->|"existing or new"| S2

    S2["<b>2. Gather extra context</b><br/><i>Tools:</i> search_chunks, get_page_text<br/><i>How:</i> SQL queries for supporting material<br/><i>Returns:</i> additional passages from corpus"]

    S2 -->|"all context gathered"| S3

    S3["<b>3. Write / update articles</b><br/><i>How:</i> LLM creates new article or merges<br/>new info into existing article<br/>with &lt;sup&gt;[N]&lt;/sup&gt; citations<br/><i>Returns:</i> full article text per entity<br/>=== ARTICLE: Name ===<br/>..body..<br/>=== END ARTICLE ==="]

    S3 --> OUT["<b>Output:</b> all messages including<br/>delimited article blocks<br/>(new + updated)"]

    %% Styling
    classDef srcDb fill:#f0ad4e,stroke:#c77f1a,color:#000
    classDef srcWiki fill:#1abc9c,stroke:#16a085,color:#fff
    classDef srcLlm fill:#4a90d9,stroke:#2c5f8a,color:#fff

    class S1 srcWiki
    class S2 srcDb
    class S3 srcLlm
```

---

## Editor (detail)

The editor is **deterministic Python** — no LLM calls. It receives the full message history and writes files to disk. It can both create new files and overwrite existing ones (for updates from the writer).

```mermaid
flowchart LR
    IN["<b>Input:</b> all messages<br/>(researcher + writer)"]

    IN -->|"regex scan all AI messages"| S1

    S1["<b>1. Parse blocks</b><br/><i>Entity blocks:</i> === ENTITY === from researcher<br/><i>Article blocks:</i> === ARTICLE === from writer<br/><i>Fallback:</i> markdown heading boundaries<br/><i>Returns:</i> entity list + article dict"]

    S1 -->|"raw blocks"| S2

    S2["<b>2. Filter</b><br/><i>Remove:</i> meta-content headings<br/>(entity summaries, page analysis)<br/><i>Remove:</i> masthead entities<br/>(editor, publisher, photographer)<br/><i>Returns:</i> cleaned entities + blocks"]

    S2 -->|"filtered"| S3

    S3["<b>3. Match entities to articles</b><br/><i>How:</i> normalize to lowercase alphanumeric,<br/>try exact match, then substring match<br/><i>Returns:</i> matched pairs + unmatched remainder"]

    S3 -->|"matched + unmatched"| S4

    S4["<b>4. Write markdown files</b><br/><i>Matched:</i> entity + article &rarr; write (create or update)<br/><i>Unmatched article:</i> infer type, write<br/><i>Unmatched entity:</i> stub from researcher context<br/><br/><i>Each file:</i> YAML front matter + body<br/><i>Path:</i> wiki/entity_type/slug.md"]

    S4 --> OUT["<b>Output:</b> written / updated article paths"]

    %% Styling
    classDef deterministic fill:#5cb85c,stroke:#3d8b3d,color:#fff

    class S1,S2,S3,S4 deterministic
```

---

## Summary Tables

| Stage | Type | Tools | Input | Output |
|-------|------|-------|-------|--------|
| **Researcher** | ReAct LLM | 1 | Page text + image | Classification + entity:info:source blocks |
| **Router** | Deterministic | 0 | ToolMessage from classify | `"writer"` or `END` |
| **Writer** | ReAct LLM | 4 | Researcher messages + entity blocks | `=== ARTICLE ===` blocks (new + updated) |
| **Editor** | Deterministic Python | 0 | All messages (entity + article blocks) | Markdown files in `wiki/` |

### Researcher Tools (1)

| Tool | Source | Purpose |
|------|--------|---------|
| `classify_page_content` | PDF image + OCR | Determine if page is ad/editorial/mixed |

### Writer Tools (4)

| Tool | Source | Purpose |
|------|--------|---------|
| `search_article_files` | wiki/*.md | Check if an article already exists |
| `read_article_file` | wiki/*.md | Read existing article content (for updates) |
| `search_chunks` | DuckDB | Find supporting text passages |
| `get_page_text` | DuckDB | Get full OCR text for a page |

### LLM Backend

Controlled by `USE_OLLAMA` env var:

| Setting | Agent LLM | Classification |
|---------|-----------|---------------|
| `USE_OLLAMA=true` | `ChatOllama` (local) | Raw `/api/generate` with vision |
| `USE_OLLAMA=false` | `ChatOpenAI` | OpenAI chat completions with vision |
