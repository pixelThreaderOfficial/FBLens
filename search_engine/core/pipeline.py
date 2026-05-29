from typing import List, Any
from search_engine.core.models import SearchResult, Document
from search_engine.storage.sqlite import SQLiteStorage
from search_engine.retrieval.hybrid import HybridRetriever

class SearchPipeline:
    """
    Coordinates the search process by connecting retrievers, rankers,
    and storage hydration in a highly decoupled, state-free workflow.
    """

    def __init__(self, storage: SQLiteStorage, retriever: HybridRetriever, ranker: Any) -> None:
        self.storage = storage
        self.retriever = retriever
        self.ranker = ranker

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Runs the end-to-end autocomplete search process:
        query -> retrieval -> ranking -> top N slice -> bulk document hydration.
        
        Args:
            query: The raw query input.
            limit: The maximum number of ranked results to return.
            
        Returns:
            A list of fully hydrated, ranked SearchResult objects.
        """
        # 1. Edge Case Mitigation: return empty list immediately on empty/whitespace query
        stripped_query = query.strip()
        if not stripped_query:
            return []

        # 2. Retrieve lightweight candidate pool (raw signals only)
        candidates = self.retriever.retrieve(stripped_query)
        if not candidates:
            # Lazy import to avoid circular dependencies
            from search_engine.core.spelling import SpellingCorrector
            from search_engine.indexing.tokenizer import Tokenizer
            
            corrector = SpellingCorrector(self.storage, Tokenizer())
            corrected_query = corrector.correct_query(stripped_query)
            if corrected_query and corrected_query != stripped_query:
                # Retry retrieval with the corrected query
                candidates = self.retriever.retrieve(corrected_query)
                if not candidates:
                    return []
                # Update query to corrected_query for ranking
                stripped_query = corrected_query
            else:
                return []

        # 3. Perform in-memory feature derivation and ranking
        ranked_candidates = self.ranker.rank(stripped_query, candidates)

        # 4. Slice to top N results
        top_candidates = ranked_candidates[:limit]
        top_ids = [c.doc_id for c in top_candidates]

        # 5. Bulk Hydration: fetch only the top N document bodies from SQLite
        # Resolves the 'Fat Candidate' memory and I/O bottleneck
        hydrated_docs = self.storage.get_documents_bulk(top_ids)

        # 6. Hydrate and construct the final SearchResult list in correct rank order
        results: List[SearchResult] = []
        for cand in top_candidates:
            doc_data = hydrated_docs.get(cand.doc_id)
            if doc_data:
                results.append(
                    SearchResult(
                        document=Document(**doc_data),
                        score=cand.score
                    )
                )

        return results
