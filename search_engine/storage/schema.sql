-- SQLite database schema for the autocomplete search engine.
-- Enforces foreign key constraints and sets up optimized indexes for prefix and trigram lookups.

-- Enable foreign keys (must also be enabled per-connection in Python)
PRAGMA foreign_keys = ON;

-- 1. Documents Table
-- Stores the original document source data. 
-- Includes title, content, type (e.g. blog, project, journal, page), and optional URL.
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Prefix Index Table
-- Maps a word prefix (e.g., 'd', 'de', 'dee', 'deep') to a document.
-- The PRIMARY KEY on (prefix, doc_id) ensures we do not have duplicate entries
-- and automatically creates a clustered index optimized for prefix lookup queries.
CREATE TABLE IF NOT EXISTS prefix_index (
    prefix TEXT NOT NULL,
    doc_id INTEGER NOT NULL,
    frequency INTEGER DEFAULT 1,
    PRIMARY KEY (prefix, doc_id),
    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- 3. Trigram Index Table
-- Maps a character trigram (3-character slice of a word, e.g., 'res', 'ese', 'sea') to a document.
-- Character trigrams are used for typo-tolerant / fuzzy search by matching overlapping trigrams.
-- The PRIMARY KEY on (trigram, doc_id) provides uniqueness and index-backed lookups.
CREATE TABLE IF NOT EXISTS trigram_index (
    trigram TEXT NOT NULL,
    doc_id INTEGER NOT NULL,
    frequency INTEGER DEFAULT 1,
    PRIMARY KEY (trigram, doc_id),
    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
);
