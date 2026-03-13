from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from pinecone import Pinecone, ServerlessSpec  # pinecone==8.0.0 :contentReference[oaicite:2]{index=2}


@dataclass
class PineconeVectorRepository:
    api_key: str
    index_name: str
    cloud: str
    region: str
    dimension: int

    def __post_init__(self) -> None:
        self._pc = Pinecone(api_key=self.api_key)
        self._index = None

    async def ensure_index(self) -> None:
        existing = {i["name"] for i in self._pc.list_indexes()}
        if self.index_name not in existing:
            logger.info("Creating Pinecone index: {}", self.index_name)
            self._pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self._index = self._pc.Index(self.index_name)

    def upsert(
        self,
        *,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
        namespace: str | None = None,
    ) -> None:
        """
        Upsert vector to Pinecone.
        
        Args:
            vector_id: Unique ID for the vector
            vector: Embedding vector
            metadata: Metadata dict (must include 'category')
            namespace: Optional namespace (defaults to category from metadata)
        """
        assert self._index is not None
        
        # Use category as namespace for fast category-based search
        # This allows us to query only within a specific category
        if namespace is None:
            category = metadata.get("category", "default")
            namespace = self._normalize_namespace(category)
        
        logger.debug(f"Upserting to namespace: {namespace}, vector_id: {vector_id}")
        
        self._index.upsert(
            vectors=[{"id": vector_id, "values": vector, "metadata": metadata}],
            namespace=namespace,
        )
    
    def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
        category: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Query vectors from Pinecone.
        
        Args:
            vector: Query embedding vector
            top_k: Number of results to return
            namespace: Optional explicit namespace
            category: Category to search in (converted to namespace)
            metadata_filter: Additional metadata filters
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching vectors with metadata
        """
        assert self._index is not None
        
        # Use category as namespace for fast category-based search
        if namespace is None and category:
            namespace = self._normalize_namespace(category)
        elif namespace is None:
            namespace = "default"
        
        logger.debug(f"Querying namespace: {namespace}, top_k: {top_k}")
        
        res = self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=metadata_filter,
            include_metadata=include_metadata,
            include_values=False,  # Not needed — Pinecone already ranks by cosine similarity
        )
        matches = res.get("matches") or []
        
        logger.debug(f"Retrieved {len(matches)} matches from namespace '{namespace}'")
        
        return matches
    
    def get_stats(self, category: str | None = None) -> dict[str, Any]:
        """
        Get index statistics including vector count.
        Used to estimate catalog size for adaptive candidate retrieval.
        
        Args:
            category: Optional category to get stats for specific namespace
            
        Returns:
            Dictionary with stats including:
            - total_vector_count: Number of vectors in namespace/index
            - dimension: Vector dimension
        """
        assert self._index is not None
        
        try:
            stats = self._index.describe_index_stats()
            
            if category:
                # Get stats for specific namespace
                namespace = self._normalize_namespace(category)
                namespace_stats = stats.get('namespaces', {}).get(namespace, {})
                vector_count = namespace_stats.get('vector_count', 0)
                logger.debug(f"Namespace '{namespace}' has {vector_count:,} vectors")
            else:
                # Total across all namespaces
                vector_count = stats.get('total_vector_count', 0)
                logger.debug(f"Total index has {vector_count:,} vectors")
            
            return {
                'total_vector_count': vector_count,
                'dimension': stats.get('dimension', self.dimension),
            }
            
        except Exception as e:
            logger.warning(f"Failed to get index stats: {e}")
            # Return default values on error
            return {
                'total_vector_count': 0,
                'dimension': self.dimension,
            }
    
    def _normalize_namespace(self, category: str) -> str:
        """
        Normalize category name to valid Pinecone namespace.
        Namespaces must be alphanumeric with hyphens/underscores.
        """
        if not category:
            return "default"
        
        # Convert to lowercase, replace spaces with hyphens
        normalized = category.lower().strip().replace(" ", "-")
        
        # Remove any non-alphanumeric characters except hyphens and underscores
        normalized = "".join(c for c in normalized if c.isalnum() or c in "-_")
        
        # Ensure it's not empty
        if not normalized:
            return "default"
        
        return normalized