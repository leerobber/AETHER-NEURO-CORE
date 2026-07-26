"""Literature/metadata ingestion and querying agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from aether_neuro_core.agents.base import (
    AgentType,
    LiteratureDocument,
    NeuroAgent,
    ResearchResult,
    ResearchTask,
)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


class KnowledgeEngine(NeuroAgent):
    """Ingests LiteratureDocuments and answers keyword-overlap queries over them.

    This is a deliberately simple TF-free keyword-overlap scorer (no external
    embedding/search service dependency) rather than a "semantic" search engine.
    It's exact and testable; swapping in a real embedding index later is a
    drop-in replacement for `_score`.
    """

    def __init__(self, name: str = "knowledge_engine") -> None:
        super().__init__(name=name, agent_type=AgentType.MICRO)
        self.knowledge_base["documents"] = {}  # doc_id -> LiteratureDocument (as dict)

    async def execute(self, task: ResearchTask) -> ResearchResult:
        """Dispatch on `task.parameters["action"]`: "ingest" or "query"."""
        action = task.parameters.get("action", "query")
        try:
            if action == "ingest":
                doc = task.parameters["document"]
                if not isinstance(doc, LiteratureDocument):
                    doc = LiteratureDocument(**doc)
                self.ingest(doc)
                return self._result(task, success=True, payload={"ingested": doc.doc_id})
            elif action == "query":
                top_k = task.parameters.get("top_k", 5)
                results = self.query(task.query, top_k=top_k)
                return self._result(
                    task,
                    success=True,
                    payload={"num_results": len(results)},
                    documents=results,
                )
            else:
                return self._result(task, success=False, error=f"Unknown action: {action}")
        except Exception as exc:  # noqa: BLE001 - surfaced via ResearchResult.error
            return self._result(task, success=False, error=str(exc))

    def ingest(self, document: LiteratureDocument) -> None:
        """Add a document to the knowledge base, indexed by doc_id."""
        self.knowledge_base["documents"][document.doc_id] = {
            "doc_id": document.doc_id,
            "source": document.source,
            "title": document.title,
            "text": document.text,
            "metadata": document.metadata,
        }

    def query(self, query_text: str, top_k: int = 5) -> List[LiteratureDocument]:
        """Return up to `top_k` ingested documents ranked by keyword overlap with `query_text`."""
        query_tokens = set(_tokenize(query_text))
        if not query_tokens:
            return []

        scored: List[tuple[float, str]] = []
        for doc_id, raw in self.knowledge_base["documents"].items():
            doc_tokens = set(_tokenize(raw["title"] + " " + raw["text"]))
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens)
            if overlap > 0:
                score = overlap / len(query_tokens | doc_tokens)
                scored.append((score, doc_id))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for _, doc_id in scored[:top_k]:
            raw = self.knowledge_base["documents"][doc_id]
            results.append(
                LiteratureDocument(
                    doc_id=raw["doc_id"],
                    source=raw["source"],
                    title=raw["title"],
                    text=raw["text"],
                    metadata=raw["metadata"],
                )
            )
        return results

    def export_json(self, path: Union[str, Path, None] = None) -> str:
        """Serialize the knowledge base to a JSON string, optionally writing to `path`."""
        payload = json.dumps(self.export_state(), indent=2)
        if path is not None:
            Path(path).write_text(payload)
        return payload

    def load_json(self, path_or_text: Union[str, Path]) -> None:
        """Load knowledge-base state from a JSON file path or a raw JSON string."""
        text = str(path_or_text)
        if Path(text).exists():
            text = Path(text).read_text()
        state: Dict[str, Any] = json.loads(text)
        self.load_state(state)
