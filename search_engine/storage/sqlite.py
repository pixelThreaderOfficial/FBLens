import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

class SQLiteStorage:
    """
    Manages SQLite connection, schema initialization, and database interactions.
    Provides methods for index insertion and retrieval of candidates.
    """

    def __init__(self, db_path: str = "search.db"):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection with dictionary-like row factories and foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Reads schema.sql and runs it to initialize the database tables and indexes."""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        schema_sql = schema_path.read_text()
        with self._get_connection() as conn:
            conn.executescript(schema_sql)

    def insert_document(self, title: str, content: str, doc_type: str, url: Optional[str] = None) -> int:
        """
        Inserts a new document into the database.
        Returns the auto-generated document ID.
        """
        query = """
            INSERT INTO documents (title, content, type, url)
            VALUES (?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (title, content, doc_type, url))
            conn.commit()
            return cursor.lastrowid

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a document by its primary key ID."""
        query = "SELECT id, title, content, type, url, created_at FROM documents WHERE id = ?"
        with self._get_connection() as conn:
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
        with self._get_connection() as conn:
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
        with self._get_connection() as conn:
            conn.executemany(query, entries)
            conn.commit()

    def insert_trigrams(self, entries: List[Tuple[str, int, int]]) -> None:
        """
        Batch inserts trigram index entries.
        entries: List of Tuple[trigram, doc_id, frequency]
        """
        query = """
            INSERT INTO trigram_index (trigram, doc_id, frequency)
            VALUES (?, ?, ?)
            ON CONFLICT(trigram, doc_id) DO UPDATE SET
                frequency = frequency + excluded.frequency
        """
        with self._get_connection() as conn:
            conn.executemany(query, entries)
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
        with self._get_connection() as conn:
            rows = conn.execute(query, (prefix,)).fetchall()
            return [(row["doc_id"], row["frequency"]) for row in rows]

    def get_documents_by_trigrams(self, trigrams: List[str]) -> List[Tuple[int, int, int]]:
        """
        Given a list of query trigrams, finds matching documents.
        Returns a list of Tuple[doc_id, match_count, sum_frequency] indicating:
        - doc_id: the document ID
        - match_count: how many of the query's trigrams are present in the document
        - sum_frequency: the sum of the frequencies of the matched trigrams in that document
        """
        if not trigrams:
            return []

        # Generate placeholders like (?, ?, ?)
        placeholders = ",".join(["?"] * len(trigrams))
        query = f"""
            SELECT doc_id, COUNT(*) as match_count, SUM(frequency) as sum_frequency
            FROM trigram_index
            WHERE trigram IN ({placeholders})
            GROUP BY doc_id
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, trigrams).fetchall()
            return [(row["doc_id"], row["match_count"], row["sum_frequency"]) for row in rows]
