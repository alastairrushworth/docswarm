# DocSwarm Process Flow

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        DB[(DuckDB/DuckLake<br/>documents, pages,<br/>chunks, entities)]
        PDF["Scanned PDF Pages<br/>(page images + OCR text)"]
    end

    START((Start)) --> NEXT["get_next_unstudied_page()"]
    NEXT -->|"page dict<br/>(id, raw_text, image_path)"| R

    subgraph LLM_BACKEND["LLM Backend (configurable)"]
        direction LR
        OLLAMA["Ollama<br/>qwen3.5:4b / gemma3:4b"]
        OPENAI["OpenAI<br/>gpt-5.4-nano"]
    end

    subgraph RESEARCHER["Researcher Agent (LLM)"]
        R["Receive page text + metadata"]
        R --> R_CLASSIFY["1. Classify page"]
        R_CLASSIFY --> R_EXTRACT["2. Extract entities"]
        R_EXTRACT --> R_SEARCH["3. Search for<br/>supporting material"]
        R_SEARCH --> R_SAVE["4. Save entities<br/>to database"]
        R_SAVE --> R_CHECK["5. Check existing<br/>wiki articles"]

        RT_VISION["classify_page_content<br/><i>multimodal vision LLM</i>"]
        RT_DB["search_chunks &middot; get_page_text<br/>&middot; list_documents<br/><i>DuckDB read queries</i>"]
        RT_ENTITY["save_entity &middot; search_entities<br/>&middot; get_entities_for_page<br/><i>entity table read/write</i>"]
        RT_FILES["search_article_files<br/><i>wiki filesystem search</i>"]

        R_CLASSIFY -.-> RT_VISION
        R_EXTRACT -.-> RT_DB
        R_SEARCH -.-> RT_DB
        R_SAVE -.-> RT_ENTITY
        R_CHECK -.-> RT_FILES
    end

    R_CHECK --> ROUTE{"Route:<br/>page classification?"}
    ROUTE -->|"advertisement"| SKIP["Skip page"]
    SKIP --> MARK
    ROUTE -->|"editorial / mixed"| W

    subgraph WRITER["Writer Agent (LLM)"]
        W["Receive researcher<br/>messages + entities"]
        W --> W_READ["1. Read source chunks"]
        W_READ --> W_SEARCH["2. Search for<br/>additional context"]
        W_SEARCH --> W_DRAFT["3. Draft wiki articles<br/>=== ARTICLE: Name ===<br/>..body..<br/>=== END ARTICLE ==="]

        WT_DB["search_chunks &middot; get_page_text<br/><i>DuckDB read queries</i>"]
        WT_FILES["read_article_file<br/>&middot; search_article_files<br/><i>wiki filesystem read/search</i>"]

        W_READ -.-> WT_DB
        W_SEARCH -.-> WT_DB
        W_SEARCH -.-> WT_FILES
    end

    W_DRAFT --> E

    subgraph EDITOR["Editor (Deterministic Python -- no LLM)"]
        direction TB
        E["Receive all messages"]
        E --> E_PARSE["1. Parse article blocks<br/><i>primary: === delimiters</i><br/><i>fallback: markdown headings</i>"]
        E_PARSE --> E_FILTER["2. Filter meta-content<br/>& masthead entities"]
        E_FILTER --> E_MATCH["3. Match entities to articles<br/><i>(fuzzy name matching)</i>"]
        E_MATCH --> E_WRITE["4. Write markdown files<br/><i>a) matched entities</i><br/><i>b) unmatched blocks</i><br/><i>c) stubs for remainder</i>"]
    end

    subgraph OUTPUT["Output"]
        WIKI["wiki/<br/>  person/*.md<br/>  organisation/*.md<br/>  place/*.md<br/>  event/*.md<br/>  concept/*.md<br/>  object/*.md"]
    end

    E_WRITE --> WIKI
    E_WRITE --> MARK["log_page_study()"]
    MARK --> NEXT

    NEXT -->|"no more pages"| DONE((Done))

    %% Data flow connections
    PDF -.->|"page image (base64)"| RT_VISION
    DB -.-> RT_DB
    DB -.-> RT_ENTITY
    DB -.-> WT_DB
    LLM_BACKEND -.->|"USE_OLLAMA"| RESEARCHER
    LLM_BACKEND -.->|"USE_OLLAMA"| WRITER

    %% Styling
    classDef agent fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef tool fill:#f0ad4e,stroke:#c77f1a,color:#000
    classDef deterministic fill:#5cb85c,stroke:#3d8b3d,color:#fff
    classDef data fill:#d9534f,stroke:#b52b27,color:#fff
    classDef decision fill:#9b59b6,stroke:#7d3c98,color:#fff

    class R,R_CLASSIFY,R_EXTRACT,R_SEARCH,R_SAVE,R_CHECK agent
    class W,W_READ,W_SEARCH,W_DRAFT agent
    class E,E_PARSE,E_FILTER,E_MATCH,E_WRITE deterministic
    class RT_VISION,RT_DB,RT_ENTITY,RT_FILES,WT_DB,WT_FILES tool
    class DB,PDF,WIKI data
    class ROUTE decision
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
