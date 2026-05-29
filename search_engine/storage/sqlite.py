import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from contextlib import contextmanager

class SQLiteStorage:
    """
    Manages SQLite connection, schema initialization, and database interactions.
    Provides methods for index insertion and retrieval of candidates.
    Supports connection reuse and transaction-aware commits.
    """

    def __init__(self, db_path: str = "search.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._in_transaction = False
        self._vocab_cache: Optional[Set[str]] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection with dictionary-like row factories and foreign keys enabled."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
        return self._conn

    def close(self) -> None:
        """Closes the cached connection if it exists."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def begin(self) -> None:
        """Starts a transaction if not already in one."""
        conn = self._get_connection()
        if not self._in_transaction:
            conn.execute("BEGIN;")
            self._in_transaction = True

    def commit(self) -> None:
        """Commits the current transaction."""
        if self._conn is not None and self._in_transaction:
            self._conn.commit()
            self._in_transaction = False

    def rollback(self) -> None:
        """Rolls back the current transaction."""
        if self._conn is not None and self._in_transaction:
            self._conn.rollback()
            self._in_transaction = False

    @contextmanager
    def transaction(self):
        """Context manager for running operations in a transaction block."""
        self.begin()
        try:
            yield
            self.commit()
        except Exception:
            self.rollback()
            raise

    def init_db(self) -> None:
        """Reads schema.sql and runs it to initialize the database tables and indexes."""
        self._vocab_cache = None
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        schema_sql = schema_path.read_text()
        conn = self._get_connection()
        conn.executescript(schema_sql)

    def insert_document(self, title: str, content: str, doc_type: str, url: Optional[str] = None) -> int:
        """
        Inserts a new document into the database.
        Returns the auto-generated document ID.
        """
        self._vocab_cache = None
        query = """
            INSERT INTO documents (title, content, type, url)
            VALUES (?, ?, ?, ?)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, (title, content, doc_type, url))
        if not self._in_transaction:
            conn.commit()
        return cursor.lastrowid

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a document by its primary key ID."""
        query = "SELECT id, title, content, type, url, created_at FROM documents WHERE id = ?"
        conn = self._get_connection()
        row = conn.execute(query, (doc_id,)).fetchone()
        if row:
            return dict(row)
        return None

    def get_documents_bulk(self, doc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Retrieves multiple documents in a single query by their primary key IDs.
        Returns a dictionary mapping doc_id -> document raw record.
        """
        if not doc_ids:
            return {}

        placeholders = ",".join(["?"] * len(doc_ids))
        query = f"SELECT id, title, content, type, url, created_at FROM documents WHERE id IN ({placeholders})"
        conn = self._get_connection()
        rows = conn.execute(query, doc_ids).fetchall()
        return {row["id"]: dict(row) for row in rows}

    def insert_prefixes(self, entries: List[Tuple[str, int, int]]) -> None:
        """
        Batch inserts prefix index entries.
        entries: List of Tuple[prefix, doc_id, frequency]
        """
        query = """
            INSERT INTO prefix_index (prefix, doc_id, frequency)
            VALUES (?, ?, ?)
            ON CONFLICT(prefix, doc_id) DO UPDATE SET
                frequency = frequency + excluded.frequency
        """
        conn = self._get_connection()
        conn.executemany(query, entries)
        if not self._in_transaction:
            conn.commit()

    def insert_trigrams(self, entries: List[Tuple[str, int]]) -> None:
        """
        Batch inserts trigram index entries.
        entries: List of Tuple[trigram, doc_id]
        """
        query = """
            INSERT OR IGNORE INTO trigram_index (trigram, doc_id)
            VALUES (?, ?)
        """
        conn = self._get_connection()
        conn.executemany(query, entries)
        if not self._in_transaction:
            conn.commit()

    def get_documents_by_prefix(self, prefix: str) -> List[Tuple[int, int]]:
        """
        Finds all doc_ids matching the given prefix.
        Returns a list of Tuple[doc_id, frequency].
        """
        query = """
            SELECT doc_id, frequency
            FROM prefix_index
            WHERE prefix = ?
        """
        conn = self._get_connection()
        rows = conn.execute(query, (prefix,)).fetchall()
        return [(row["doc_id"], row["frequency"]) for row in rows]

    def get_documents_by_trigrams(self, trigrams: List[str]) -> List[Tuple[int, int]]:
        """
        Given a list of query trigrams, finds matching documents.
        Returns a list of Tuple[doc_id, match_count] indicating:
        - doc_id: the document ID
        - match_count: how many of the query's trigrams are present in the document
        """
        if not trigrams:
            return []

        # Generate placeholders like (?, ?, ?)
        placeholders = ",".join(["?"] * len(trigrams))
        query = f"""
            SELECT doc_id, COUNT(*) as match_count
            FROM trigram_index
            WHERE trigram IN ({placeholders})
            GROUP BY doc_id
        """
        conn = self._get_connection()
        rows = conn.execute(query, trigrams).fetchall()
        return [(row["doc_id"], row["match_count"]) for row in rows]

    def get_vocabulary(self) -> Set[str]:
        """Retrieves and caches the vocabulary of all unique tokens from all documents."""
        if self._vocab_cache is not None:
            return self._vocab_cache

        query = "SELECT title, content FROM documents"
        conn = self._get_connection()
        rows = conn.execute(query).fetchall()

        from search_engine.indexing.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        vocab = set()
        for row in rows:
            vocab.update(tokenizer.tokenize(row["title"]))
            vocab.update(tokenizer.tokenize(row["content"]))

        self._vocab_cache = vocab
        return vocab
