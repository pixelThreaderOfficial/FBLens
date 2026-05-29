from collections import defaultdict
from typing import List, Dict
from search_engine.storage.sqlite import SQLiteStorage
from search_engine.indexing.tokenizer import Tokenizer
from search_engine.core.models import ScoredCandidate

class PrefixRetriever:
    """
    Retrieves candidate documents matching query term prefixes.
    """

    def __init__(self, storage: SQLiteStorage, tokenizer: Tokenizer) -> None:
        self.storage = storage
        self.tokenizer = tokenizer

    def retrieve(self, query: str) -> List[ScoredCandidate]:
        """
        Tokenizes the query, finds matching documents in the SQLite prefix index,
        and computes a normalized prefix score based on cumulative prefix frequency.
        
        Args:
            query: The raw query typed by the user.
            
        Returns:
            A list of lightweight ScoredCandidates with 'prefix' score populated.
        """
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        # Accumulator: doc_id -> sum of matching prefix frequencies
        raw_scores: Dict[int, int] = defaultdict(int)

        # Retrieve matching doc IDs for each prefix token
        for token in tokens:
            matches = self.storage.get_documents_by_prefix(token)
            for doc_id, frequency in matches:
                raw_scores[doc_id] += frequency

        if not raw_scores:
            return []

        candidates: List[ScoredCandidate] = []
        for doc_id, freq in raw_scores.items():
            candidates.append(
                ScoredCandidate(
                    doc_id=doc_id,
                    scores={"prefix_raw": float(freq)}
                )
            )

        return candidates


