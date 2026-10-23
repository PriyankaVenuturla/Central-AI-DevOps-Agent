"""ChromaDB Vector Store & Semantic Memory Layer

Implements institutional memory using ChromaDB and all-MiniLM-L6-v2 embeddings.
Stores and retrieves:
1. Past resolved CI/CD incidents (error signature -> fix)
2. Team operational runbooks and troubleshooting SOPs
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger("central_agent.memory")


class ChromaMemoryManager:
    """Manages vector collections in ChromaDB with all-MiniLM-L6-v2 embeddings."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.model_name = model_name or settings.embedding_model_name

        self._client = None
        self._embedding_function = None
        self._fixes_collection = None
        self._runbooks_collection = None

    def _init_chroma(self):
        """Initializes ChromaDB persistent client and all-MiniLM-L6-v2 embedding function."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)

            try:
                # Use sentence-transformers with all-MiniLM-L6-v2
                self._embedding_function = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.model_name
                    )
                )
                logger.info("Initialized SentenceTransformerEmbeddingFunction with %s", self.model_name)
            except Exception as ef_err:
                logger.warning(
                    "Could not load SentenceTransformerEmbeddingFunction (%s). Falling back to DefaultEmbeddingFunction.",
                    ef_err,
                )
                self._embedding_function = embedding_functions.DefaultEmbeddingFunction()

            self._fixes_collection = self._client.get_or_create_collection(
                name="past_incident_fixes",
                embedding_function=self._embedding_function,
            )

            self._runbooks_collection = self._client.get_or_create_collection(
                name="team_runbooks",
                embedding_function=self._embedding_function,
            )

            # Seed default SOP runbooks if empty
            if self._runbooks_collection.count() == 0:
                self.seed_default_runbooks()

        except ImportError:
            logger.warning(
                "chromadb is not installed. Vector memory will run in mock/in-memory mode."
            )
            self._client = False

    def seed_default_runbooks(self):
        """Populates baseline team runbooks for frequent DevOps build failures."""
        if not self._runbooks_collection:
            return

        default_runbooks = [
            {
                "title": "Java Compiler Syntax and Missing Delimiters",
                "content": (
                    "When javac reports error: ')' expected or ';' expected at line N, "
                    "check for unclosed method calls, missing parentheses, or forgotten semicolons. "
                    "Verify System.out.println calls and multi-line chained builder methods."
                ),
                "category": "compilation",
            },
            {
                "title": "NPM Peer Dependency Conflict",
                "content": (
                    "When npm install fails with ERESOLVE could not resolve dependency or peer dependency conflicts, "
                    "check package.json for mismatched major versions of react/vue/angular plugins. "
                    "Resolution: update package versions or use --legacy-peer-deps in the CI workflow file if safe."
                ),
                "category": "dependency",
            },
            {
                "title": "Python ModuleNotFoundError in CI",
                "content": (
                    "When pytest or python execution fails with ModuleNotFoundError: No module named 'xyz', "
                    "ensure the module is included in requirements.txt or pyproject.toml and installed via pip in the CI step. "
                    "Also verify PYTHONPATH if imports are relative to the repo root."
                ),
                "category": "dependency",
            },
            {
                "title": "Docker Daemon Socket Permission Denied",
                "content": (
                    "When docker build fails with 'Got permission denied while trying to connect to the Docker daemon socket', "
                    "ensure the CI runner user is in the 'docker' group or the container workflow has privileged: true enabled."
                ),
                "category": "infrastructure",
            },
        ]

        for doc in default_runbooks:
            doc_id = str(uuid.uuid4())
            self._runbooks_collection.add(
                ids=[doc_id],
                documents=[doc["content"]],
                metadatas=[{"title": doc["title"], "category": doc["category"]}],
            )
        logger.info("Seeded %d default runbooks into ChromaDB.", len(default_runbooks))

    def search_past_fixes(self, error_text: str, n_results: int = 2) -> List[Dict[str, Any]]:
        """Queries the past_incident_fixes collection for similar error signatures."""
        self._init_chroma()
        if not self._fixes_collection:
            return []

        try:
            results = self._fixes_collection.query(
                query_texts=[error_text],
                n_results=min(n_results, max(1, self._fixes_collection.count())),
            )
            hits = []
            if results and results.get("documents"):
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                for doc, meta in zip(docs, metas):
                    hits.append({"document": doc, "metadata": meta})
            return hits
        except Exception as e:
            logger.warning("Error querying past fixes from ChromaDB: %s", e)
            return []

    def search_runbooks(self, query: str, n_results: int = 2) -> List[Dict[str, Any]]:
        """Queries the team_runbooks collection for relevant SOP instructions."""
        self._init_chroma()
        if not self._runbooks_collection:
            return []

        try:
            results = self._runbooks_collection.query(
                query_texts=[query],
                n_results=min(n_results, max(1, self._runbooks_collection.count())),
            )
            hits = []
            if results and results.get("documents"):
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                for doc, meta in zip(docs, metas):
                    hits.append({"document": doc, "metadata": meta})
            return hits
        except Exception as e:
            logger.warning("Error querying runbooks from ChromaDB: %s", e)
            return []

    def save_resolved_fix(
        self,
        error_signature: str,
        root_cause: str,
        fix: str,
        repo: str = "unknown",
    ):
        """Indexes a newly verified resolution into ChromaDB for future reference."""
        self._init_chroma()
        if not self._fixes_collection:
            return

        doc = (
            f"Error Signature: {error_signature}\n"
            f"Root Cause: {root_cause}\n"
            f"Verified Fix: {fix}"
        )
        doc_id = str(uuid.uuid4())
        self._fixes_collection.add(
            ids=[doc_id],
            documents=[doc],
            metadatas=[{"repo": repo, "root_cause": root_cause}],
        )
        logger.info("Saved verified fix for '%s' to ChromaDB.", repo)
