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
        direction TB
        R["Receive page text<br/>+ metadata"]
        R --> R_CLASSIFY["1. Classify page"]
        R_CLASSIFY --> R_EXTRACT["2. Extract entities"]
        R_EXTRACT --> R_SEARCH["3. Search for<br/>supporting material"]
        R_SEARCH --> R_SAVE["4. Save entities<br/>to database"]
        R_SAVE --> R_CHECK["5. Check existing<br/>wiki articles"]

        subgraph R_TOOLS["Researcher Tools (8)"]
            direction LR
            RT1["classify_page_content<br/><i>multimodal vision call</i>"]
            RT2["search_chunks"]
            RT3["get_page_text"]
            RT4["list_documents"]
            RT5["save_entity"]
            RT6["search_entities"]
            RT7["get_entities_for_page"]
            RT8["search_article_files"]
        end
    end

    R_CHECK --> ROUTE{"Route:<br/>page classification?"}
    ROUTE -->|"advertisement"| SKIP["Skip page"]
    SKIP --> MARK
    ROUTE -->|"editorial / mixed"| W

    subgraph WRITER["Writer Agent (LLM)"]
        direction TB
        W["Receive researcher<br/>messages + entities"]
        W --> W_READ["1. Read source chunks"]
        W_READ --> W_SEARCH["2. Search for<br/>additional context"]
        W_SEARCH --> W_DRAFT["3. Draft wiki articles<br/>=== ARTICLE: Name ===<br/>..body..<br/>=== END ARTICLE ==="]

        subgraph W_TOOLS["Writer Tools (4)"]
            direction LR
            WT1["search_chunks"]
            WT2["get_page_text"]
            WT3["read_article_file"]
            WT4["search_article_files"]
        end
    end

    W_DRAFT --> E

    subgraph EDITOR["Editor (Deterministic Python -- no LLM)"]
        direction TB
        E["Receive all messages"]
        E --> E_PARSE["1. Parse article blocks<br/><i>primary: === delimiters</i><br/><i>fallback: markdown headings</i>"]
        E_PARSE --> E_FILTER["2. Filter meta-content<br/>& masthead entities"]
        E_FILTER --> E_MATCH["3. Match entities<br/>to article blocks<br/><i>(fuzzy name matching)</i>"]
        E_MATCH --> E_PHASE1["4a. Phase 1: Write matched<br/>entity articles"]
        E_PHASE1 --> E_PHASE2["4b. Phase 2: Write unmatched<br/>article blocks"]
        E_PHASE2 --> E_STUB["4c. Stub articles for<br/>unmatched entities"]
    end

    subgraph OUTPUT["Output"]
        WIKI["wiki/<br/>  person/*.md<br/>  organisation/*.md<br/>  place/*.md<br/>  event/*.md<br/>  concept/*.md<br/>  object/*.md"]
    end

    E_STUB --> WIKI
    E_STUB --> MARK["log_page_study()"]
    MARK --> NEXT

    NEXT -->|"no more pages"| DONE((Done))

    %% Data flow connections
    DB -.->|"chunks, pages,<br/>entities"| R_TOOLS
    PDF -.->|"page image<br/>(base64)"| RT1
    DB -.->|"chunks, pages"| W_TOOLS
    LLM_BACKEND -.->|"USE_OLLAMA<br/>env var"| RESEARCHER
    LLM_BACKEND -.->|"USE_OLLAMA<br/>env var"| WRITER

    %% Styling
    classDef agent fill:#4a90d9,stroke:#2c5f8a,color:#fff
    classDef tool fill:#f0ad4e,stroke:#c77f1a,color:#000
    classDef deterministic fill:#5cb85c,stroke:#3d8b3d,color:#fff
    classDef data fill:#d9534f,stroke:#b52b27,color:#fff
    classDef decision fill:#9b59b6,stroke:#7d3c98,color:#fff

    class R,R_CLASSIFY,R_EXTRACT,R_SEARCH,R_SAVE,R_CHECK agent
    class W,W_READ,W_SEARCH,W_DRAFT agent
    class E,E_PARSE,E_FILTER,E_MATCH,E_PHASE1,E_PHASE2,E_STUB deterministic
    class RT1,RT2,RT3,RT4,RT5,RT6,RT7,RT8,WT1,WT2,WT3,WT4 tool
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
