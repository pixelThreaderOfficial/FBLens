import re
import unicodedata
from typing import List

# Compile regex at module-level to make the class stateless and avoid recompiling
WORD_PATTERN = re.compile(r"\w+")

class Tokenizer:
    """
    Tokenizer responsible for converting text into clean, normalized search tokens.
    Provides a separate normalize() method for cases where tokenization is not required.
    """

    def normalize(self, text: str) -> str:
        """
        Normalizes a string of text by lowercasing and stripping diacritics/accents.
        
        Args:
            text: The raw input string.
            
        Returns:
            A clean, normalized string.
        """
        if not text:
            return ""

        # 1. Lowercase conversion
        text_lower = text.lower()

        # 2. Unicode decomposition and diacritics removal (e.g. 'résumé' -> 'resume')
        decomposed = unicodedata.normalize("NFKD", text_lower)
        return "".join(c for c in decomposed if not unicodedata.combining(c))

    def tokenize(self, text: str) -> List[str]:
        """
        Normalizes and tokenizes a string of text.
        
        Args:
            text: The raw input string.
            
        Returns:
            A list of clean, normalized word tokens.
        """
        normalized = self.normalize(text)
        if not normalized:
            return []

        # 3. Extract word boundaries (strips symbols, punctuation, brackets)
        raw_tokens = WORD_PATTERN.findall(normalized)

        # 4. Filter empty entries
        return [token for token in raw_tokens if token]

