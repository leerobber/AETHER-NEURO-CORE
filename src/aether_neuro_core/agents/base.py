"""Core agent abstractions: task/result contracts and the NeuroAgent base class."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentType(Enum):
    """Granularity tier of an agent in the coordination hierarchy."""

    SUPER = "super"
    SUB = "sub"
    MICRO = "micro"
    NANO = "nano"


@dataclass
class ResearchTask:
    """A unit of work handed to a NeuroAgent."""

    query: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parameters: Dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class LiteratureDocument:
    """A single ingested piece of literature/metadata (e.g. a UniProt or PubMed record)."""

    doc_id: str
    source: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """The outcome of executing a ResearchTask."""

    task_id: str
    agent_name: str
    success: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    documents: List[LiteratureDocument] = field(default_factory=list)
    error: Optional[str] = None
    completed_at: float = field(default_factory=time.time)


class NeuroAgent(ABC):
    """Base class for all agents in the AetherNeuroCore coordination hierarchy.

    Subclasses implement `execute` to perform their specific work. Each agent
    keeps a small mutable knowledge base (`self.knowledge_base`) that accumulates
    state across calls and can be exported/imported as JSON via `export_state`
    and `load_state`.
    """

    def __init__(self, name: str, agent_type: AgentType) -> None:
        self.name = name
        self.agent_type = agent_type
        self.knowledge_base: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, task: ResearchTask) -> ResearchResult:
        """Execute a research task and return its result."""
        raise NotImplementedError

    def export_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of this agent's knowledge base."""
        return {
            "name": self.name,
            "agent_type": self.agent_type.value,
            "knowledge_base": self.knowledge_base,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore knowledge-base state previously produced by `export_state`."""
        self.knowledge_base = state.get("knowledge_base", {})

    def _result(
        self,
        task: ResearchTask,
        success: bool,
        payload: Optional[Dict[str, Any]] = None,
        documents: Optional[List[LiteratureDocument]] = None,
        error: Optional[str] = None,
    ) -> ResearchResult:
        return ResearchResult(
            task_id=task.task_id,
            agent_name=self.name,
            success=success,
            payload=payload or {},
            documents=documents or [],
            error=error,
        )
