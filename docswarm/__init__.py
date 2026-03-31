"""docswarm — transform PDF corpora into structured wikis using an AI agent swarm."""

from docswarm.agents.swarm import DocSwarm
from docswarm.config import Config
from docswarm.extraction.pipeline import ExtractionPipeline
from docswarm.storage.database import DatabaseManager
from docswarm.wiki.client import WikiJSClient

__version__ = "0.1.0"
__all__ = ["Config", "DatabaseManager", "ExtractionPipeline", "DocSwarm", "WikiJSClient"]
