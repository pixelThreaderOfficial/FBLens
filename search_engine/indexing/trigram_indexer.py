from collections import defaultdict
from typing import Dict, List, Set, Tuple
from search_engine.storage.sqlite import SQLiteStorage
from search_engine.indexing.tokenizer import Tokenizer

class TrigramIndexer:
    """
    Analyzes document text, extracts 3-character substrings (trigrams) 
    for words of length >= 3, and saves them to the SQLite trigram_index table.
    """

    MIN_WORD_LENGTH = 3

    def __init__(self, storage: SQLiteStorage, tokenizer: Tokenizer) -> None:
        self.storage = storage
        self.tokenizer = tokenizer

    def generate_trigrams_for_token(self, token: str) -> List[str]:
        """
        Slices a token into 3-character sequences (trigrams).
        Only tokens with length >= MIN_WORD_LENGTH are processed.
        E.g. 'research' -> ['res', 'ese', 'sea', 'ear', 'arc', 'rch']
        """
        if len(token) < self.MIN_WORD_LENGTH:
            return []
        
        return [token[i : i + 3] for i in range(len(token) - 2)]

    def index_document(self, doc_id: int, title: str, content: str) -> None:
        """
        Extracts tokens from the title and content, generates trigrams,
        aggregates their occurrences, and batch inserts them into SQLite.
        """
        # Map of trigram -> aggregated frequency within this document
        trigram_frequencies: Dict[str, int] = defaultdict(int)

        # Tokenize title and content together
        # We merge them as trigrams are field-agnostic for general typo tolerance
        all_text = f"{title} {content}"
        tokens = self.tokenizer.tokenize(all_text)

        for token in tokens:
            trigrams = self.generate_trigrams_for_token(token)
            for trigram in trigrams:
                trigram_frequencies[trigram] += 1

        # If no trigrams generated (e.g. document only has short words), exit early
        if not trigram_frequencies:
            return

        # Prepare entries for database: (trigram, doc_id, frequency)
        entries: List[Tuple[str, int, int]] = [
            (trigram, doc_id, freq)
            for trigram, freq in trigram_frequencies.items()
        ]

        # Batch insert into trigram table
        self.storage.insert_trigrams(entries)
