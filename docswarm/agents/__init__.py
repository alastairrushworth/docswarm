"""Agent swarm sub-package."""

from docswarm.agents.personas import RESEARCHER_PROMPT
from docswarm.agents.personas import WRITER_PROMPT
from docswarm.agents.swarm import DocSwarm

__all__ = [
    "DocSwarm",
    "RESEARCHER_PROMPT",
    "WRITER_PROMPT",
]
