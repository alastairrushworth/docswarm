"""Purge all swarm-generated data: DB studies/entities, logs, and wiki output."""

from pathlib import Path
from shutil import rmtree

from dotenv import load_dotenv

load_dotenv()

from docswarm.config import Config
from docswarm.storage.database import DatabaseManager

config = Config.from_env()
db = DatabaseManager(config.catalog_path, config.data_path)

# Clear swarm tables
for table in ("page_studies", "entities", "entity_mentions"):
    db._exec(f"DELETE FROM docswarm.{table}")
    print(f"  Cleared {table}")

db.close()

# Clear logs
logs_dir = Path("logs")
if logs_dir.exists():
    for f in logs_dir.iterdir():
        f.write_text("")
    print(f"  Cleared {sum(1 for _ in logs_dir.iterdir())} log file(s)")

# Clear wiki output
wiki_dir = Path(config.wiki_output_dir)
if wiki_dir.exists():
    rmtree(wiki_dir)
    wiki_dir.mkdir()
    print(f"  Cleared wiki output: {wiki_dir}")

print("Done.")
