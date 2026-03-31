"""Run the agent swarm for every document in the lakehouse.

Configure via environment variables or a .env file:
    DOCSWARM_WIKI_URL       Wiki.js base URL (default: http://localhost:3000)
    DOCSWARM_WIKI_API_KEY   Wiki.js API key
    DOCSWARM_MODEL          Ollama model name (default: gemma3:4b)
    DOCSWARM_OLLAMA_BASE_URL Ollama server URL (default: http://localhost:11434)
    DOCSWARM_CATALOG_PATH   DuckLake catalog file (default: docswarm_catalog.db)
    DOCSWARM_DATA_PATH      Parquet data directory (default: docswarm_data/)
"""

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from docswarm.agents.swarm import DocSwarm
    from docswarm.config import Config
    from docswarm.storage.database import DatabaseManager
    from docswarm.wiki.client import WikiJSClient

    config = Config.from_env()
    db = DatabaseManager(config.catalog_path, config.data_path)
    db.initialize()

    wiki_client = WikiJSClient(
        base_url=config.wiki_url or "http://localhost:3000",
        api_key=config.wiki_api_key,
    )

    try:
        docs = db.list_documents()
        if not docs:
            print("No documents found. Run extract.py first.")
        else:
            print(f"Generating wiki articles for {len(docs)} document(s).")
            swarm = DocSwarm(config=config, db=db, wiki_client=wiki_client)
            results = swarm.run_full_wiki_generation()
            print(f"Done. Generated {len(results)} article(s).")
    finally:
        db.close()
        wiki_client.close()
