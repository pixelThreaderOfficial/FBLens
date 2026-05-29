from typing import List, Set
from search_engine.storage.sqlite import SQLiteStorage
from search_engine.indexing.tokenizer import Tokenizer
from search_engine.core.models import ScoredCandidate

class TrigramRetriever:
    """
    Retrieves candidate documents using character trigram overlap for typo tolerance.
    """

    MIN_WORD_LENGTH = 3

    def __init__(self, storage: SQLiteStorage, tokenizer: Tokenizer) -> None:
        self.storage = storage
        self.tokenizer = tokenizer

    def generate_query_trigrams(self, tokens: List[str]) -> Set[str]:
        """
        Decomposes query tokens into unique character trigrams.
        Only processes tokens with length >= MIN_WORD_LENGTH.
        """
        query_trigrams: Set[str] = set()
        for token in tokens:
            if len(token) >= self.MIN_WORD_LENGTH:
                for i in range(len(token) - 2):
                    query_trigrams.add(token[i : i + 3])
        return query_trigrams

    def retrieve(self, query: str) -> List[ScoredCandidate]:
        """
        Tokenizes the query, extracts unique character trigrams, performs a database
        lookup of intersecting trigram rows, and returns candidates with their raw match count.
        
        Args:
            query: The raw query typed by the user.
            
        Returns:
            A list of lightweight ScoredCandidates with the 'trigram' raw match count populated.
        """
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        # Generate unique query trigrams
        query_trigrams = self.generate_query_trigrams(tokens)
        if not query_trigrams:
            # Query has no tokens long enough to produce trigrams. Return empty.
            return []

        # Retrieve matches from the trigram index
        # SQLite: get_documents_by_trigrams returns List[Tuple[doc_id, match_count, sum_frequency]]
        matches = self.storage.get_documents_by_trigrams(list(query_trigrams))
        if not matches:
            return []

        candidates: List[ScoredCandidate] = []
        for doc_id, match_count, _ in matches:
            candidates.append(
                ScoredCandidate(
                    doc_id=doc_id,
                    scores={
                        "trigram_raw": float(match_count),
                        "query_trigram_count": float(len(query_trigrams))
                    }
                )
            )

        return candidates

