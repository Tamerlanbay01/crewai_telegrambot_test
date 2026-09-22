"""Construct optional semantic-memory integrations from configuration."""

import logging

from core.config import config
from integrations.memory.embedding import OpenAICompatibleEmbeddingProvider
from integrations.memory.index import EmbeddingProvider, MemorySearchIndex
from integrations.memory.qdrant import QdrantMemorySearchIndex

logger = logging.getLogger(__name__)


def create_semantic_memory_dependencies(
) -> tuple[EmbeddingProvider | None, MemorySearchIndex | None]:
    if not (
        config.embedding.base_url
        and config.embedding.model
        and config.embedding.dimension > 0
        and config.qdrant.url
    ):
        return None, None
    try:
        return (
            OpenAICompatibleEmbeddingProvider(config.embedding),
            QdrantMemorySearchIndex(config.qdrant, dimension=config.embedding.dimension),
        )
    except Exception:
        logger.exception("Semantic memory is disabled because its integrations could not initialize")
        return None, None
