"""Agent swarm sub-package."""

from docswarm.agents.personas import (
    EDITOR_PROMPT,
    RESEARCHER_PROMPT,
    REVIEWER_PROMPT,
    WRITER_PROMPT,
)
from docswarm.agents.swarm import DocSwarm, SwarmState

__all__ = [
    "DocSwarm",
    "SwarmState",
    "RESEARCHER_PROMPT",
    "WRITER_PROMPT",
    "REVIEWER_PROMPT",
    "EDITOR_PROMPT",
]
