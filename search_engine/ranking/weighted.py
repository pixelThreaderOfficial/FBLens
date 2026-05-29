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
            query: The user's search query.
            candidates: The list of lightweight ScoredCandidates with raw scores.
            
        Returns:
            A sorted list of ScoredCandidates with their final score updated.
        """
        if not candidates:
            return []

        # Calculate query-dependent normalizers to prevent Rank Reversal (Task 8)
        from search_engine.indexing.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(query)
        query_token_count = float(len(tokens)) if tokens else 1.0

        # Calculate total trigrams in the query
        query_trigrams_count = 0.0
        for token in tokens:
            if len(token) >= 3:
                query_trigrams_count += float(len(token) - 2)
        if query_trigrams_count == 0.0:
            query_trigrams_count = 1.0

        ranked_candidates: List[ScoredCandidate] = []

        # 2. Derive normalized features and compute final weighted score
        for cand in candidates:
            prefix_raw = cand.scores.get("prefix_raw", 0.0)
            trigram_raw = cand.scores.get("trigram_raw", 0.0)

            # Normalize Prefix Score relative to the number of tokens in query
            prefix_score = prefix_raw / query_token_count

            # Normalize Trigram Similarity relative to query trigram count
            cand_query_trigram_count = cand.scores.get("query_trigram_count", query_trigrams_count)
            if cand_query_trigram_count <= 0.0:
                cand_query_trigram_count = 1.0
            trigram_similarity = trigram_raw / cand_query_trigram_count

            # Compute Final Score
            final_score = (prefix_score * self.prefix_weight) + (trigram_similarity * self.trigram_weight)

            # Update the candidate score property
            cand.score = final_score
            ranked_candidates.append(cand)

        # 3. Sort candidates in descending order of final score
        # Secondary sort key on doc_id to ensure deterministic results for identical scores
        ranked_candidates.sort(key=lambda x: (-x.score, x.doc_id))

        return ranked_candidates
