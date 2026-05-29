from typing import Set, Optional
from search_engine.storage.sqlite import SQLiteStorage
from search_engine.indexing.tokenizer import Tokenizer

class SpellingCorrector:
    """
    Provides lightweight, dynamic programming based spelling correction suggestion.
    Utilizes the SQLite document index vocabulary to dynamically identify typos
    and suggest close replacements based on Levenshtein distance.
    """

    def __init__(self, storage: SQLiteStorage, tokenizer: Tokenizer) -> None:
        self.storage = storage
        self.tokenizer = tokenizer

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Computes the Levenshtein distance between two strings using dynamic programming."""
        if len(s1) < len(s2):
            return SpellingCorrector.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def correct_query(self, query: str) -> Optional[str]:
        """
        Tokenizes the query, checks each token against the system vocabulary.
        If a token is misspelled (not in vocabulary), finds the closest match
        with Levenshtein distance <= 2.
        Returns the corrected query string if corrections were made, or None otherwise.
        """
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return None

        vocab = self.storage.get_vocabulary()
        if not vocab:
            return None

        corrected_tokens = []
        changed = False

        for token in tokens:
            if token in vocab:
                corrected_tokens.append(token)
                continue

            # Token is misspelled! Find the best match in the vocabulary
            best_candidate = token
            best_distance = 3  # We only correct for distance <= 2

            for vocab_word in vocab:
                # Early length filter for performance optimization
                if abs(len(vocab_word) - len(token)) >= best_distance:
                    continue

                dist = self.levenshtein_distance(token, vocab_word)
                if dist < best_distance:
                    best_distance = dist
                    best_candidate = vocab_word

            if best_distance <= 2:
                corrected_tokens.append(best_candidate)
                changed = True
            else:
                corrected_tokens.append(token)

        if changed:
            return " ".join(corrected_tokens)
        return None
