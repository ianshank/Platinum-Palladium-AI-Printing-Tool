"""
Vertex AI Search integration for grounded Pt/Pd knowledge retrieval.

Provides RAG (Retrieval-Augmented Generation) capabilities by querying a
Vertex AI Search data store containing Pt/Pd printing literature, paper
profiles, chemistry references, troubleshooting guides, and calibration
documentation.

Usage:
    from ptpd_calibration.vertex.search import PtPdSearchClient

    client = PtPdSearchClient(project_id="my-project", data_store_id="ptpd-knowledge")
    results = client.search("What Pt/Pd ratio for warm tones on Hahnemühle?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ptpd_calibration.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from Vertex AI Search.

    Args:
        title: Document title or filename.
        snippet: Relevant text snippet from the document.
        document_id: Unique document identifier in the data store.
        relevance_score: Search relevance score (0.0 to 1.0).
        metadata: Additional document metadata.
    """

    title: str
    snippet: str
    document_id: str
    relevance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PtPdSearchClient:
    """Client for querying the Pt/Pd knowledge base via Vertex AI Search.

    This client wraps the Vertex AI Discovery Engine API to provide
    domain-specific search over the Pt/Pd printing knowledge corpus.

    Args:
        project_id: Google Cloud project ID. Falls back to config if not provided.
        location: Google Cloud region. Defaults to "global" for Search.
        data_store_id: Vertex AI Search data store ID.
        serving_config: Search serving configuration name.
    """

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "global",
        data_store_id: str | None = None,
        serving_config: str = "default_search",
    ):
        settings = get_settings().vertex
        self.project_id = project_id or settings.project_id
        self.location = location
        self.data_store_id = data_store_id or settings.search_data_store_id
        self.serving_config = serving_config or settings.search_serving_config
        self._client = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Discovery Engine client."""
        if self._client is None:
            try:
                from google.cloud import discoveryengine_v1 as discoveryengine
            except ImportError as err:
                raise ImportError(
                    "google-cloud-discoveryengine required. "
                    "Install with: pip install ptpd-calibration[vertex]"
                ) from err

            self._client = discoveryengine.SearchServiceClient()
        return self._client

    @property
    def _serving_config_path(self) -> str:
        """Build the full serving config resource path."""
        return (
            f"projects/{self.project_id}"
            f"/locations/{self.location}"
            f"/collections/default_collection"
            f"/dataStores/{self.data_store_id}"
            f"/servingConfigs/{self.serving_config}"
        )

    def search(
        self,
        query: str,
        max_results: int | None = None,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """Search the Pt/Pd knowledge base.

        Args:
            query: Natural language search query about Pt/Pd printing.
            max_results: Maximum number of results to return.
            filter_expr: Optional filter expression for narrowing results.

        Returns:
            List of SearchResult objects ordered by relevance.
        """
        from google.cloud import discoveryengine_v1 as discoveryengine

        settings = get_settings().vertex
        max_results = max_results or settings.search_max_results

        logger.info("Searching Pt/Pd knowledge base: query=%r, max_results=%d", query, max_results)

        client = self._get_client()

        request = discoveryengine.SearchRequest(
            serving_config=self._serving_config_path,
            query=query,
            page_size=max_results,
        )

        if filter_expr:
            request.filter = filter_expr

        response = client.search(request=request)

        results = []
        for result in response.results:
            doc = result.document
            doc_data = _extract_document_data(doc)

            results.append(
                SearchResult(
                    title=doc_data.get("title", doc.id or "Unknown"),
                    snippet=doc_data.get("snippet", ""),
                    document_id=doc.id or "",
                    relevance_score=getattr(result, "relevance_score", 0.0),
                    metadata=doc_data.get("metadata", {}),
                )
            )

        logger.debug("Search returned %d results", len(results))
        return results

    def search_with_summary(
        self,
        query: str,
        max_results: int | None = None,
    ) -> tuple[str, list[SearchResult]]:
        """Search with a generative AI summary of results.

        Requires Enterprise edition with generative answers enabled.

        Args:
            query: Natural language query.
            max_results: Maximum search results.

        Returns:
            Tuple of (generative_summary, search_results).
        """
        from google.cloud import discoveryengine_v1 as discoveryengine

        settings = get_settings().vertex
        max_results = max_results or settings.search_max_results

        logger.info("Search with summary: query=%r, max_results=%d", query, max_results)

        client = self._get_client()

        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=min(max_results, 5),
                include_citations=True,
                model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                    version="gemini-2.5-flash",
                ),
            ),
            extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                max_extractive_answer_count=3,
            ),
        )

        request = discoveryengine.SearchRequest(
            serving_config=self._serving_config_path,
            query=query,
            page_size=max_results,
            content_search_spec=content_search_spec,
        )

        response = client.search(request=request)

        # Extract summary
        summary = ""
        if hasattr(response, "summary") and response.summary:
            summary = response.summary.summary_text or ""

        # Extract results
        results = []
        for result in response.results:
            doc = result.document
            doc_data = _extract_document_data(doc)
            results.append(
                SearchResult(
                    title=doc_data.get("title", doc.id or "Unknown"),
                    snippet=doc_data.get("snippet", ""),
                    document_id=doc.id or "",
                    relevance_score=getattr(result, "relevance_score", 0.0),
                    metadata=doc_data.get("metadata", {}),
                )
            )

        return summary, results

    def format_context_for_llm(
        self,
        results: list[SearchResult],
        max_context_length: int = 4000,
    ) -> str:
        """Format search results as context for an LLM prompt.

        Args:
            results: Search results to format.
            max_context_length: Maximum character length of context.

        Returns:
            Formatted context string suitable for LLM system/user prompts.
        """
        context_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            entry = f"[Source {i}: {result.title}]\n{result.snippet}\n"
            if current_length + len(entry) > max_context_length:
                break
            context_parts.append(entry)
            current_length += len(entry)

        if not context_parts:
            return ""

        return (
            "The following information is from the Pt/Pd printing knowledge base:\n\n"
            + "\n".join(context_parts)
        )


def _extract_document_data(doc: Any) -> dict[str, Any]:
    """Extract structured data from a Discovery Engine document.

    Args:
        doc: A discoveryengine Document object.

    Returns:
        Dict with title, snippet, and metadata.
    """
    data: dict[str, Any] = {"metadata": {}}

    # Try struct_data first (structured documents)
    if hasattr(doc, "struct_data") and doc.struct_data:
        struct = dict(doc.struct_data)
        data["title"] = struct.get("title", "")
        data["snippet"] = struct.get("snippet", struct.get("content", ""))
        data["metadata"] = {
            k: v for k, v in struct.items() if k not in ("title", "snippet", "content")
        }

    # Fall back to derived_struct_data
    elif hasattr(doc, "derived_struct_data") and doc.derived_struct_data:
        derived = dict(doc.derived_struct_data)

        # Extract snippets from extractive answers
        snippets = []
        for answer in derived.get("extractive_answers", []):
            if isinstance(answer, dict) and "content" in answer:
                snippets.append(answer["content"])

        data["title"] = derived.get("title", derived.get("link", ""))
        data["snippet"] = "\n".join(snippets) if snippets else str(derived)

    # Try content (unstructured)
    elif hasattr(doc, "content") and doc.content:
        raw = doc.content.raw_bytes
        if raw:
            try:
                text = raw.decode("utf-8")
                data["snippet"] = text[:2000]
            except (UnicodeDecodeError, AttributeError):
                data["snippet"] = str(raw)[:2000]

    if not data.get("title"):
        data["title"] = getattr(doc, "name", "") or getattr(doc, "id", "Unknown")

    return data
