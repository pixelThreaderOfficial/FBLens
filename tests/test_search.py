from pathlib import Path
import pytest
from search_engine.storage.sqlite import SQLiteStorage
from search_engine.indexing.tokenizer import Tokenizer
from search_engine.indexing.prefix_indexer import PrefixIndexer
from search_engine.indexing.trigram_indexer import TrigramIndexer
from search_engine.retrieval.prefix import PrefixRetriever
from search_engine.retrieval.trigram import TrigramRetriever
from search_engine.retrieval.hybrid import HybridRetriever
from search_engine.ranking.weighted import WeightedRanker
from search_engine.core.pipeline import SearchPipeline
from search_engine.core.models import ScoredCandidate

@pytest.fixture
def test_db():
    """Fixture that initializes a temporary SQLite database and cleans it up after tests."""
    db_file = Path("test_search_fixture.db")
    if db_file.exists():
        db_file.unlink()
        
    storage = SQLiteStorage(str(db_file))
    storage.init_db()
    
    yield storage
    
    if db_file.exists():
        db_file.unlink()

@pytest.fixture
def tokenizer():
    return Tokenizer()

def test_tokenizer_normalization(tokenizer):
    # Test lowercase and accents
    assert tokenizer.tokenize("RÉSUMÉ") == ["resume"]
    # Test symbols and punctuation removal
    assert tokenizer.tokenize("deep-researcher v2.0!") == ["deep", "researcher", "v2", "0"]
    # Test empty inputs
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ") == []

def test_prefix_indexer_weights(test_db, tokenizer):
    indexer = PrefixIndexer(test_db, tokenizer)
    
    # Satisfy SQLite Foreign Key constraints by inserting the document first
    doc_id = test_db.insert_document("Deep Project", "Design systems", "Project")
    
    indexer.index_document(doc_id, "Deep Project", "Design systems")
    
    matches = test_db.get_documents_by_prefix("de")
    # 'de' matches once in title ('deep' -> de: 5) and once in content ('design' -> de: 1)
    # Aggregated frequency in doc 1 should be 6
    assert len(matches) == 1
    assert matches[0] == (doc_id, 6)

def test_trigram_indexer_generates_trigrams(test_db, tokenizer):
    indexer = TrigramIndexer(test_db, tokenizer)
    
    # Satisfy SQLite Foreign Key constraints by inserting the document first
    doc_id = test_db.insert_document("Deep", "Researcher", "Project")
    
    indexer.index_document(doc_id, "Deep", "Researcher")
    
    # 'Researcher' triggers trigrams: res, ese, sea, ear, arc, rch, che, her
    # 'Deep' has length 4, triggers: dee, eep
    matches = test_db.get_documents_by_trigrams(["res", "ese", "rch"])
    assert len(matches) == 1
    # Match count should be 3 unique matches
    assert matches[0][0] == doc_id
    assert matches[0][1] == 3  # match_count

def test_weighted_ranker_computes_correct_scores():
    ranker = WeightedRanker({"prefix_weight": 0.6, "trigram_weight": 0.4})
    
    # Candidate 1: prefix_raw=10, trigram_raw=3, query_trigram_count=5 (score = 0.54)
    # Candidate 2: prefix_raw=20, trigram_raw=1, query_trigram_count=5 (score = 0.68)
    # max_prefix_raw in pool is 20.0
    candidates = [
        ScoredCandidate(doc_id=1, scores={"prefix_raw": 10.0, "trigram_raw": 3.0, "query_trigram_count": 5.0}),
        ScoredCandidate(doc_id=2, scores={"prefix_raw": 20.0, "trigram_raw": 1.0, "query_trigram_count": 5.0}),
    ]
    
    ranked = ranker.rank("test query", candidates)
    
    # Candidate 2 should be first in rank because 0.68 > 0.54
    assert ranked[0].doc_id == 2
    assert abs(ranked[0].score - 0.68) < 1e-6

    # Candidate 1 should be second
    assert ranked[1].doc_id == 1
    assert abs(ranked[1].score - 0.54) < 1e-6

def test_pipeline_end_to_end(test_db, tokenizer):
    prefix_idx = PrefixIndexer(test_db, tokenizer)
    trigram_idx = TrigramIndexer(test_db, tokenizer)

    # Index two documents
    doc_1 = test_db.insert_document("Deep Researcher V2", "Autonomous research agent", "Project")
    doc_2 = test_db.insert_document("SQLite schema", "Optimized index parameters", "Blog")

    prefix_idx.index_document(doc_1, "Deep Researcher V2", "Autonomous research agent")
    trigram_idx.index_document(doc_1, "Deep Researcher V2", "Autonomous research agent")

    prefix_idx.index_document(doc_2, "SQLite schema", "Optimized index parameters")
    trigram_idx.index_document(doc_2, "SQLite schema", "Optimized index parameters")

    # Set up pipeline
    pref_retriever = PrefixRetriever(test_db, tokenizer)
    trig_retriever = TrigramRetriever(test_db, tokenizer)
    hybrid_retriever = HybridRetriever(pref_retriever, trig_retriever)
    ranker = WeightedRanker({"prefix_weight": 0.7, "trigram_weight": 0.3})
    pipeline = SearchPipeline(test_db, hybrid_retriever, ranker)

    # Search: 'deep resercher' (typo tolerant prefix)
    results = pipeline.search("deep resercher", limit=5)
    # Both documents match due to 'deep/researcher' and a minor trigram syllable overlap ('che' in 'schema')
    assert len(results) >= 1
    # The high confidence match must rank #1
    assert results[0].document.title == "Deep Researcher V2"
    assert results[0].score > 0.0

    # Search: 'sqlite'
    results_sqlite = pipeline.search("sqlite", limit=5)
    assert len(results_sqlite) == 1
    assert results_sqlite[0].document.title == "SQLite schema"
