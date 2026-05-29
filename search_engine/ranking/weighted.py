from typing import List, Dict, Any
from search_engine.core.models import ScoredCandidate

class WeightedRanker:
    """
    Weighted ranker that normalizes raw prefix and trigram features in-memory,
    applies configurable weights, and ranks candidates.
    """

    def __init__(self, config: Dict[str, float] = None) -> None:
        # Default weights if no configuration is passed
        self.config = config or {
            "prefix_weight": 0.7,
            "trigram_weight": 0.3
        }
        self.prefix_weight = self.config.get("prefix_weight", 0.7)
        self.trigram_weight = self.config.get("trigram_weight", 0.3)

    def rank(self, query: str, candidates: List[ScoredCandidate]) -> List[ScoredCandidate]:
        """
        Derives normalized scores, computes weighted final scores, and sorts
        candidates in descending order.
        
        Args:
            query: The user's search query (retained for future-proofing interfaces).
            candidates: The list of lightweight ScoredCandidates with raw scores.
            
        Returns:
            A sorted list of ScoredCandidates with their final score updated.
        """
        if not candidates:
            return []

        # 1. Prefix Normalization: find the maximum raw prefix score in the current candidate pool
        max_prefix_raw = max(
            (cand.scores.get("prefix_raw", 0.0) for cand in candidates),
            default=0.0
        )

        ranked_candidates: List[ScoredCandidate] = []

        # 2. Derive normalized features and compute final weighted score
        for cand in candidates:
            prefix_raw = cand.scores.get("prefix_raw", 0.0)
            trigram_raw = cand.scores.get("trigram_raw", 0.0)
            query_trigram_count = cand.scores.get("query_trigram_count", 0.0)

            # Normalize Prefix Score relative to max in pool
            prefix_score = prefix_raw / max_prefix_raw if max_prefix_raw > 0.0 else 0.0

            # Normalize Trigram Similarity relative to total query trigrams
            trigram_similarity = (
                trigram_raw / query_trigram_count if query_trigram_count > 0.0 else 0.0
            )

            # Compute Final Score
            final_score = (prefix_score * self.prefix_weight) + (trigram_similarity * self.trigram_weight)

            # Update the candidate score property
            cand.score = final_score
            ranked_candidates.append(cand)

        # 3. Sort candidates in descending order of final score
        # Secondary sort key on doc_id to ensure deterministic results for identical scores
        ranked_candidates.sort(key=lambda x: (-x.score, x.doc_id))

        return ranked_candidates
