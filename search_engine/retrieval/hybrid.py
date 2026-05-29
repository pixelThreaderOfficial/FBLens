from typing import List, Dict
from search_engine.core.models import ScoredCandidate
from search_engine.retrieval.prefix import PrefixRetriever
from search_engine.retrieval.trigram import TrigramRetriever

class HybridRetriever:
    """
    Orchestrates PrefixRetriever and TrigramRetriever.
    Merges their candidate lists (Union) and preserves raw evidence metrics
    to keep retrieval fully decoupled from ranking.
    """

    def __init__(self, prefix_retriever: PrefixRetriever, trigram_retriever: TrigramRetriever) -> None:
        self.prefix_retriever = prefix_retriever
        self.trigram_retriever = trigram_retriever

    def retrieve(self, query: str) -> List[ScoredCandidate]:
        """
        Runs Prefix and Trigram retrievers, merges their candidate sets by doc_id,
        and ensures that every candidate carries a full raw score vector.
        
        Args:
            query: The raw query typed by the user.
            
        Returns:
            A unified list of ScoredCandidate objects with raw features populated.
        """
        prefix_candidates = self.prefix_retriever.retrieve(query)
        trigram_candidates = self.trigram_retriever.retrieve(query)

        # Map to merge candidates: doc_id -> ScoredCandidate
        merged: Dict[int, ScoredCandidate] = {}

        # 1. Process prefix candidates
        for cand in prefix_candidates:
            doc_id = cand.doc_id
            merged[doc_id] = ScoredCandidate(
                doc_id=doc_id,
                scores={
                    "prefix_raw": cand.scores.get("prefix_raw", 0.0),
                    "trigram_raw": 0.0,
                    "query_trigram_count": 0.0
                }
            )

        # Find the absolute query trigram count from trigram candidates if it exists
        query_trigram_count = 0.0
        for cand in trigram_candidates:
            if "query_trigram_count" in cand.scores:
                query_trigram_count = cand.scores["query_trigram_count"]
                break

        # 2. Merge trigram candidates
        for cand in trigram_candidates:
            doc_id = cand.doc_id
            if doc_id in merged:
                # Document was already retrieved by prefix matching. Enrich it.
                merged[doc_id].scores["trigram_raw"] = cand.scores.get("trigram_raw", 0.0)
                merged[doc_id].scores["query_trigram_count"] = query_trigram_count
            else:
                # Brand new document only retrieved via trigrams
                merged[doc_id] = ScoredCandidate(
                    doc_id=doc_id,
                    scores={
                        "prefix_raw": 0.0,
                        "trigram_raw": cand.scores.get("trigram_raw", 0.0),
                        "query_trigram_count": query_trigram_count
                    }
                )

        # 3. Post-processing: Ensure query_trigram_count is populated globally
        # If any candidate matched trigrams, all candidates share the same query properties.
        for cand in merged.values():
            cand.scores["query_trigram_count"] = query_trigram_count

        return list(merged.values())
