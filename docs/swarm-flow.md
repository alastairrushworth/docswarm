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

    RESEARCHER["<b>Researcher</b><br/><i>ReAct LLM agent</i><br/>5 tools<br/><br/>Classifies page,<br/>gathers entity info,<br/>outputs entity blocks"]

    RESEARCHER -->|"all messages"| ROUTE{"Classification?"}

    ROUTE -->|"advertisement"| MARK["log_page_study()"]
    ROUTE -->|"editorial / mixed"| WRITER

    WRITER["<b>Writer</b><br/><i>ReAct LLM agent</i><br/>4 tools<br/><br/>Drafts wiki articles<br/>for each entity"]

    WRITER -->|"all messages"| EDITOR

    EDITOR["<b>Editor</b><br/><i>Deterministic Python</i><br/>no LLM<br/><br/>Parses entity + article blocks<br/>from messages, writes markdown"]

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
flowchart LR
    IN["<b>Input:</b> page text,<br/>image path, metadata"]

    IN --> S1

    S1["<b>1. Classify page</b><br/><i>Tool:</i> classify_page_content<br/><i>How:</i> page image + OCR text &rarr; vision LLM<br/><i>Returns:</i> editorial / advertisement / mixed"]

    S1 -->|"advertisement"| OUT_AD["<b>Output:</b> classification only<br/>(LLM stops, no entities)"]
    S1 -->|"editorial / mixed"| S2

    S2["<b>2. Read page content</b><br/><i>Tools:</i> get_page_text, search_chunks<br/><i>How:</i> SQL queries on chunks table<br/><i>Returns:</i> full OCR text + relevant chunks"]

    S2 -->|"page text + chunks"| S3

    S3["<b>3. Identify entities</b><br/><i>How:</i> LLM reads text, identifies<br/>people, places, organisations, events,<br/>objects, concepts<br/><i>Returns:</i> candidate entity list with types"]

    S3 -->|"entity names + types"| S4

    S4["<b>4. Check existing articles</b><br/><i>Tool:</i> search_article_files<br/><i>How:</i> glob + read wiki/*.md<br/><i>Returns:</i> which entities already<br/>have articles vs need new ones"]

    S4 --> S5

    S5["<b>5. Output entity blocks</b><br/><i>How:</i> LLM outputs structured<br/>=== ENTITY: Name (type) ===<br/>facts + sources<br/>=== END ENTITY ===<br/><i>Returns:</i> entity:info pairs in messages"]

    S5 --> OUT["<b>Output:</b> messages with classification<br/>+ entity blocks (no DB writes)"]

    %% Styling
    classDef srcDb fill:#f0ad4e,stroke:#c77f1a,color:#000
    classDef srcWiki fill:#1abc9c,stroke:#16a085,color:#fff
    classDef srcMixed fill:#e74c3c,stroke:#f0ad4e,color:#fff,stroke-width:4px
    classDef srcLlm fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef srcStop fill:#999,stroke:#666,color:#fff

    class S1 srcMixed
    class S2 srcDb
    class S3 srcLlm
    class S4 srcWiki
    class S5 srcLlm
    class OUT_AD srcStop
```

---

## Writer (detail)

The writer also runs as a **single ReAct LLM invoke()**, receiving the full message history from the researcher.

```mermaid
flowchart LR
    IN["<b>Input:</b> researcher messages<br/>+ entity blocks"]

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
flowchart LR
    IN["<b>Input:</b> all messages<br/>(researcher + writer)"]

    IN -->|"regex scan all AI messages"| S1

    S1["<b>1. Parse blocks</b><br/><i>Entity blocks:</i> === ENTITY === from researcher<br/><i>Article blocks:</i> === ARTICLE === from writer<br/><i>Fallback:</i> markdown heading boundaries<br/><i>Returns:</i> entity list + article dict"]

    S1 -->|"raw blocks"| S2

    S2["<b>2. Filter</b><br/><i>Remove:</i> meta-content headings<br/>(entity summaries, page analysis)<br/><i>Remove:</i> masthead entities<br/>(editor, publisher, photographer)<br/><i>Returns:</i> cleaned entities + blocks"]

    S2 -->|"filtered"| S3

    S3["<b>3. Match entities to articles</b><br/><i>How:</i> normalize to lowercase alphanumeric,<br/>try exact match, then substring match<br/><i>Returns:</i> matched pairs + unmatched remainder"]

    S3 -->|"matched + unmatched"| S4

    S4["<b>4. Write markdown files</b><br/><i>Phase a:</i> entity + matched article &rarr; full article<br/><i>Phase b:</i> unmatched article block &rarr; infer type, write<br/><i>Phase c:</i> unmatched entity &rarr; stub from context<br/><br/><i>Each file:</i> YAML front matter + body<br/><i>Path:</i> wiki/entity_type/slug.md"]

    S4 --> OUT["<b>Output:</b> written article paths"]

    %% Styling
    classDef deterministic fill:#5cb85c,stroke:#3d8b3d,color:#fff

    class S1,S2,S3,S4 deterministic
```

---

## Summary Tables

| Stage | Type | Tools | Input | Output |
|-------|------|-------|-------|--------|
| **Researcher** | ReAct LLM | 5 | Page text + image | Classification + entity blocks in messages |
| **Router** | Deterministic | 0 | ToolMessage from classify | `"writer"` or `END` |
| **Writer** | ReAct LLM | 4 | Researcher messages | `=== ARTICLE ===` delimited blocks |
| **Editor** | Deterministic Python | 0 | All messages (entity + article blocks) | Markdown files in `wiki/` |

### Researcher Tools (5)

| Tool | Source | Purpose |
|------|--------|---------|
| `classify_page_content` | PDF image + OCR | Determine if page is ad/editorial/mixed |
| `search_chunks` | DuckDB | Find text chunks matching a query |
| `get_page_text` | DuckDB | Get full OCR text for a page |
| `list_documents` | DuckDB | List available documents |
| `search_article_files` | wiki/*.md | Check if an article already exists |

### Writer Tools (4)

| Tool | Source | Purpose |
|------|--------|---------|
| `search_chunks` | DuckDB | Find supporting text passages |
| `get_page_text` | DuckDB | Get full OCR text for a page |
| `search_article_files` | wiki/*.md | Check existing articles |
| `read_article_file` | wiki/*.md | Read an existing article's content |

### LLM Backend

Controlled by `USE_OLLAMA` env var:

| Setting | Agent LLM | Classification |
|---------|-----------|---------------|
| `USE_OLLAMA=true` | `ChatOllama` (local) | Raw `/api/generate` with vision |
| `USE_OLLAMA=false` | `ChatOpenAI` | OpenAI chat completions with vision |
