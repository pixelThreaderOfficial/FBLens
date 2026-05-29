import pytest
from pathlib import Path
from typing import List, Dict, Any, Tuple
from search_engine.storage.sqlite import SQLiteStorage
from search_engine.indexing.tokenizer import Tokenizer
from search_engine.indexing.prefix_indexer import PrefixIndexer
from search_engine.indexing.trigram_indexer import TrigramIndexer
from search_engine.retrieval.prefix import PrefixRetriever
from search_engine.retrieval.trigram import TrigramRetriever
from search_engine.retrieval.hybrid import HybridRetriever
from search_engine.ranking.weighted import WeightedRanker
from search_engine.core.pipeline import SearchPipeline
from search_engine.core.spelling import SpellingCorrector

@pytest.fixture
def quality_db():
    """Initializes a clean SQLite database for quality testing."""
    db_file = Path("test_search_quality_fixture.db")
    if db_file.exists():
        db_file.unlink()
        
    storage = SQLiteStorage(str(db_file))
    storage.init_db()
    
    yield storage
    
    storage.close()
    if db_file.exists():
        db_file.unlink()

def test_search_quality(quality_db):
    tokenizer = Tokenizer()
    prefix_indexer = PrefixIndexer(quality_db, tokenizer)
    trigram_indexer = TrigramIndexer(quality_db, tokenizer)

    # 1. Populate Corpus
    corpus = [
        ("Deep Researcher", "An autonomous research agent optimized for search and database retrieval tasks.", "Project"),
        ("SQLite Optimization", "A detailed guide to sqlite query latency and index database optimization.", "Blog"),
        ("Personalization Engine", "Designing a local-first personalization and visualization platform.", "Project"),
        ("智能搜索", "基于人工智能的本地自动补全搜索引擎。", "Tutorial"),
    ]

    with quality_db.transaction():
        for title, content, doc_type in corpus:
            doc_id = quality_db.insert_document(title, content, doc_type)
            prefix_indexer.index_document(doc_id, title, content)
            trigram_indexer.index_document(doc_id, title, content)

    # 2. Setup Search Pipeline
    pref_ret = PrefixRetriever(quality_db, tokenizer)
    trig_ret = TrigramRetriever(quality_db, tokenizer)
    hyb_ret = HybridRetriever(pref_ret, trig_ret)
    ranker = WeightedRanker({"prefix_weight": 0.6, "trigram_weight": 0.4})
    pipeline = SearchPipeline(quality_db, hyb_ret, ranker)

    # 3. Define Benchmark Queries & Expected Results
    benchmarks = [
        # (Query, Expected Document Title)
        ("deep re", "Deep Researcher"),
        ("deep resercher", "Deep Researcher"),
        ("agent", "Deep Researcher"),
        ("agnt", "Deep Researcher"),
        ("sqltie", "SQLite Optimization"),
        ("sqlite", "SQLite Optimization"),
        ("personalization", "Personalization Engine"),
        ("research", "Deep Researcher"),
        ("智能", "智能搜索"),
    ]

    results_report = []
    top_1_hits = 0
    top_3_hits = 0
    top_5_hits = 0

    print("\n" + "=" * 80)
    print("                 SEARCH QUALITY BENCHMARK REPORT                 ")
    print("=" * 80)
    print(f"{'Query':<25} | {'Expected Title':<25} | {'Actual Top Title':<25} | {'Status':<6}")
    print("-" * 88)

    for query, expected_title in benchmarks:
        search_results = pipeline.search(query, limit=5)
        titles = [res.document.title for res in search_results]

        top_1_match = len(titles) > 0 and titles[0] == expected_title
        top_3_match = expected_title in titles[:3]
        top_5_match = expected_title in titles[:5]

        if top_1_match:
            top_1_hits += 1
        if top_3_match:
            top_3_hits += 1
        if top_5_match:
            top_5_hits += 1

        actual_top = titles[0] if titles else "NO MATCH"
        status = "PASS" if top_1_match else "FAIL"

        print(f"{query:<25} | {expected_title:<25} | {actual_top:<25} | {status:<6}")
        results_report.append({
            "query": query,
            "expected": expected_title,
            "actual": actual_top,
            "status": status,
            "top_1": top_1_match,
            "top_3": top_3_match,
            "top_5": top_5_match,
            "all_retrieved": titles
        })

    total_queries = len(benchmarks)
    top_1_acc = top_1_hits / total_queries
    top_3_acc = top_3_hits / total_queries
    top_5_acc = top_5_hits / total_queries

    print("=" * 80)
    print(f"Top-1 Accuracy: {top_1_acc * 100:.1f}% ({top_1_hits}/{total_queries})")
    print(f"Top-3 Accuracy: {top_3_acc * 100:.1f}% ({top_3_hits}/{total_queries})")
    print(f"Top-5 Accuracy: {top_5_acc * 100:.1f}% ({top_5_hits}/{total_queries})")
    print("=" * 80)

    # 4. Enforce Quality Assertions
    # We expect Top-1 accuracy to be high (e.g. at least 70% to pass)
    assert top_1_acc >= 0.77, f"Top-1 Accuracy dropped to {top_1_acc:.2%}"
    assert top_3_acc >= 0.88, f"Top-3 Accuracy dropped to {top_3_acc:.2%}"

    # Specifically test that spelling corrector behaves correctly with Chinese script
    corrector = SpellingCorrector(quality_db, tokenizer)
    
    # 5. Unicode Script Blindness specific check
    # Let's ensure '智能' is not corrected to any Latin script word (like 'in') if they are not script-compatible.
    # In our vocab, we have Latin words ('deep', 'researcher', 'sqlite', 'optimization', etc.) and CJK words ('智能搜索').
    # 'in' is not in this vocab, but if we query '智能', it has CJK script.
    # If we check correction for '智能', it must not yield any Latin suggestions even if they have small edit distance.
    # Actually, let's verify corrector returns None or a CJK word, and never a Latin word.
    suggestion = corrector.correct_query("智能")
    if suggestion is not None:
        suggestion_scripts = corrector._get_word_scripts(suggestion)
        token_scripts = corrector._get_word_scripts("智能")
        assert token_scripts & suggestion_scripts, f"Cross-script correction detected! '{suggestion}' suggested for '智能'"
