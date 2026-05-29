from collections import defaultdict
from typing import Dict, List, Tuple
from search_engine.storage.sqlite import SQLiteStorage
from search_engine.indexing.tokenizer import Tokenizer

class PrefixIndexer:
    """
    Analyzes document text, generates progressive prefix slices, 
    and writes them to the SQLite prefix_index table.
    """

    MAX_PREFIX_LENGTH = 12
    TITLE_WEIGHT = 5
    CONTENT_WEIGHT = 1

    def __init__(self, storage: SQLiteStorage, tokenizer: Tokenizer) -> None:
        self.storage = storage
        self.tokenizer = tokenizer

    def generate_prefixes_for_token(self, token: str) -> List[str]:
        """
        Generates progressive left-to-right slices for a normalized token.
        E.g. 'deep' -> ['d', 'de', 'dee', 'deep'] up to MAX_PREFIX_LENGTH.
        """
        limit = min(len(token), self.MAX_PREFIX_LENGTH)
        return [token[:i] for i in range(1, limit + 1)]

    def index_document(self, doc_id: int, title: str, content: str) -> None:
        """
        Extracts tokens from the title and content, generates prefixes,
        aggregates their frequencies (with field weights applied),
        and batch inserts them into SQLite.
        """
        # Dictionary mapping prefix -> aggregated frequency
        prefix_frequencies: Dict[str, int] = defaultdict(int)

        # 1. Process Title tokens (Higher Weight)
        title_tokens = self.tokenizer.tokenize(title)
        for token in title_tokens:
            for prefix in self.generate_prefixes_for_token(token):
                prefix_frequencies[prefix] += self.TITLE_WEIGHT

        # 2. Process Content tokens (Standard Weight)
        content_tokens = self.tokenizer.tokenize(content)
        for token in content_tokens:
            for prefix in self.generate_prefixes_for_token(token):
                prefix_frequencies[prefix] += self.CONTENT_WEIGHT

        # If no tokens generated (empty document), return early
        if not prefix_frequencies:
            return

        # Convert to list of Tuples for fast batch insertion: (prefix, doc_id, frequency)
        entries: List[Tuple[str, int, int]] = [
            (prefix, doc_id, freq)
            for prefix, freq in prefix_frequencies.items()
        ]

        # 3. Batch commit to storage layer
        self.storage.insert_prefixes(entries)
