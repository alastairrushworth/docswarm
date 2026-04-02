# DocSwarm Process Flow

```mermaid
flowchart LR
    subgraph KEY["Data Sources (colour key)"]
        direction TB
        DB[(DuckDB/DuckLake<br/>documents, pages,<br/>chunks, entities)]
        PDF["Scanned PDF Pages<br/>(images + OCR text)"]
        WIKI_IN["wiki/*.md<br/>(existing articles)"]
    end

    START((Start)) --> NEXT["get_next_<br/>unstudied_page()"]
    NEXT -->|"page dict"| R

    subgraph LLM_BACKEND["LLM Backend"]
        direction TB
        OLLAMA["Ollama"]
        OPENAI["OpenAI<br/>gpt-5.4-nano"]
    end

    subgraph RESEARCHER["Researcher Agent — single ReAct LLM invoke()"]
        direction TB

        R["Receive page text + metadata"]

        R -->|"LLM emits tool call"| R_CLASSIFY

        R_CLASSIFY["<b>1. Classify page</b><br/><i>Tool:</i> classify_page_content<br/><i>How:</i> page image + OCR &rarr; vision LLM<br/><i>Returns:</i> editorial / advert / mixed"]

        R_CLASSIFY -->|"classification label"| R_EXTRACT

        R_EXTRACT["<b>2. Read page text</b><br/><i>Tools:</i> get_page_text, search_chunks<br/><i>How:</i> SQL queries on chunks table<br/><i>Returns:</i> full OCR text + matching chunks"]

        R_EXTRACT -->|"page text"| R_IDENTIFY

        R_IDENTIFY["<b>3. Identify entities</b><br/><i>How:</i> LLM reads text, picks out<br/>people, places, orgs, events<br/><i>Returns:</i> candidate entity list"]

        R_IDENTIFY -->|"entity names + types"| R_SAVE

        R_SAVE["<b>4. Save entities</b><br/><i>Tools:</i> save_entity, search_entities<br/><i>How:</i> check duplicates, SQL insert<br/><i>Returns:</i> entity IDs in DB"]

        R_SAVE -->|"entity IDs"| R_CHECK

        R_CHECK["<b>5. Check existing articles</b><br/><i>Tool:</i> search_article_files<br/><i>How:</i> glob + read wiki/*.md<br/><i>Returns:</i> which entities need new articles"]
    end

    R_CHECK -->|"researcher done;<br/>read ToolMessage<br/>from step 1"| ROUTE

    ROUTE{"Route:<br/>classification?"}
    ROUTE -->|"advertisement"| MARK
    ROUTE -->|"editorial /<br/>mixed"| W

    subgraph WRITER["Writer Agent — single ReAct LLM invoke()"]
        direction TB

        W["Receive researcher messages + entity list"]

        W -->|"LLM emits tool call"| W_READ

        W_READ["<b>1. Gather source material</b><br/><i>Tools:</i> search_chunks, get_page_text<br/><i>How:</i> SQL queries for each entity<br/><i>Returns:</i> relevant passages"]

        W_READ -->|"source passages"| W_SEARCH

        W_SEARCH["<b>2. Check existing articles</b><br/><i>Tools:</i> search_article_files,<br/>read_article_file<br/><i>How:</i> glob + read wiki/*.md<br/><i>Returns:</i> existing content"]

        W_SEARCH -->|"all context gathered"| W_DRAFT

        W_DRAFT["<b>3. Draft wiki articles</b><br/><i>How:</i> LLM generates text with<br/>inline citations<br/><i>Returns:</i> === ARTICLE: Name ===<br/>..body..<br/>=== END ARTICLE ==="]
    end

    W_DRAFT -->|"all messages"| E

    subgraph EDITOR["Editor — deterministic Python, no LLM"]
        direction TB

        E["Receive all messages"]

        E -->|"regex scan"| E_PARSE

        E_PARSE["<b>1. Parse article blocks</b><br/><i>How:</i> regex for === delimiters<br/><i>Fallback:</i> markdown headings<br/><i>Returns:</i> title &rarr; body dict"]

        E_PARSE -->|"blocks + entities"| E_FILTER

        E_FILTER["<b>2. Filter</b><br/><i>How:</i> remove meta-content headings<br/>+ masthead entities<br/><i>Returns:</i> cleaned set"]

        E_FILTER -->|"filtered"| E_MATCH

        E_MATCH["<b>3. Match entities to articles</b><br/><i>How:</i> normalize names, exact +<br/>substring match<br/><i>Returns:</i> matched pairs + remainder"]

        E_MATCH -->|"matched + unmatched"| E_WRITE

        E_WRITE["<b>4. Write markdown files</b><br/><i>a)</i> matched &rarr; full article<br/><i>b)</i> unmatched block &rarr; infer type<br/><i>c)</i> unmatched entity &rarr; stub<br/><i>Output:</i> wiki/type/slug.md"]
    end

    E_WRITE -->|"new articles"| WIKI_IN
    E_WRITE --> MARK["log_page_study()"]
    MARK --> NEXT

    NEXT -->|"no more pages"| DONE((Done))

    LLM_BACKEND -.->|"USE_OLLAMA"| RESEARCHER
    LLM_BACKEND -.->|"USE_OLLAMA"| WRITER

    %% Styling
    classDef deterministic fill:#5cb85c,stroke:#3d8b3d,color:#fff
    classDef decision fill:#9b59b6,stroke:#7d3c98,color:#fff

    %% Data source colours
    classDef srcDb fill:#f0ad4e,stroke:#c77f1a,color:#000
    classDef srcPdf fill:#e74c3c,stroke:#c0392b,color:#fff
    classDef srcWiki fill:#1abc9c,stroke:#16a085,color:#fff
    classDef srcMixed fill:#e74c3c,stroke:#f0ad4e,color:#fff,stroke-width:4px
    classDef srcLlm fill:#4a90d9,stroke:#2c5f8a,color:#fff

    %% Researcher steps by data source
    class R srcLlm
    class R_CLASSIFY srcMixed
    class R_EXTRACT srcDb
    class R_IDENTIFY srcLlm
    class R_SAVE srcDb
    class R_CHECK srcWiki

    %% Writer steps by data source
    class W srcLlm
    class W_READ srcDb
    class W_SEARCH srcWiki
    class W_DRAFT srcLlm

    %% Editor steps
    class E,E_PARSE,E_FILTER,E_MATCH,E_WRITE deterministic

    %% Decision + key
    class ROUTE decision
    class DB srcDb
    class PDF srcPdf
    class WIKI_IN srcWiki
```

## Pipeline Summary

| Stage | Type | Tools | Input | Output |
|-------|------|-------|-------|--------|
| **Researcher** | LLM (ReAct agent) | 8 | Page text + image | Classification + entities in DB |
| **Router** | Deterministic | 0 | ToolMessage from classify | `"writer"` or `END` |
| **Writer** | LLM (ReAct agent) | 4 | Researcher messages | `=== ARTICLE ===` delimited blocks |
| **Editor** | Deterministic Python | 0 | All messages + DB entities | Markdown files in `wiki/` |

## Article Writing Strategy (Editor)

1. **Parse** writer output for `=== ARTICLE: Name === ... === END ARTICLE ===` blocks
2. **Fallback** to markdown heading parsing (`# Name`) if no delimiters found
3. **Filter** meta-content headings (entity summaries, page analysis, etc.)
4. **Match** DB entities to article blocks using normalized fuzzy matching
5. **Phase 1**: Write files for entities with matching writer content
6. **Phase 2**: Write files for unmatched article blocks (infer type from DB or body text)
7. **Stubs**: Create minimal articles for entities with no writer match (using entity context)

## Entity Quality Filters

- Masthead roles (editor, publisher, photographer, etc.) with short context are excluded
- Canonical type normalization (`organisations` -> `organisation`, `people` -> `person`)
- Duplicate detection via `search_entities` and `search_article_files`

## LLM Backend

Controlled by `USE_OLLAMA` env var:

| Setting | Agent LLM | Classification |
|---------|-----------|---------------|
| `USE_OLLAMA=true` | `ChatOllama` (local) | Raw `/api/generate` with vision |
| `USE_OLLAMA=false` | `ChatOpenAI` | OpenAI chat completions with vision |
