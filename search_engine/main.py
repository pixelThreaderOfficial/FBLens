import json
from pathlib import Path

# Import search engine components
from search_engine.storage.sqlite import SQLiteStorage
from search_engine.indexing.tokenizer import Tokenizer
from search_engine.indexing.prefix_indexer import PrefixIndexer
from search_engine.indexing.trigram_indexer import TrigramIndexer
from search_engine.retrieval.prefix import PrefixRetriever
from search_engine.retrieval.trigram import TrigramRetriever
from search_engine.retrieval.hybrid import HybridRetriever
from search_engine.ranking.weighted import WeightedRanker
from search_engine.core.pipeline import SearchPipeline

def main() -> None:
    print("=" * 60)
    print("INITIALIZING LOCAL-FIRST AUTOCOMPLETE SEARCH ENGINE")
    print("=" * 60)

    # 1. Setup a fresh SQLite database to ensure clean indexing demonstration
    db_path = Path("search.db")
    if db_path.exists():
        db_path.unlink()
        print("[System] Wiped existing search.db for a clean indexing run.")

    storage = SQLiteStorage(str(db_path))
    storage.init_db()
    print("[System] SQLite schema initialized successfully.")

    # 2. Instantiate indexers
    tokenizer = Tokenizer()
    prefix_indexer = PrefixIndexer(storage, tokenizer)
    trigram_indexer = TrigramIndexer(storage, tokenizer)

    # 3. Load sample document corpus
    docs_json_path = Path(__file__).parent / "data" / "documents.json"
    if not docs_json_path.exists():
        raise FileNotFoundError(f"Could not find documents.json at {docs_json_path}")

    with open(docs_json_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"[Indexer] Found {len(documents)} source documents. Starting indexing...")

    # 4. Populate Database & Index Tables
    for doc in documents:
        # Insert document to fetch auto-incremented primary key
        doc_id = storage.insert_document(
            title=doc["title"],
            content=doc["content"],
            doc_type=doc["type"],
            url=doc.get("url")
        )
        # Populate index mappings
        prefix_indexer.index_document(doc_id, doc["title"], doc["content"])
        trigram_indexer.index_document(doc_id, doc["title"], doc["content"])

    print("[Indexer] Indexing completed successfully.")
    print(f" - Documents: {len(documents)}")
    print(" - Index Tables: Populated prefix_index & trigram_index")
    print("-" * 60)

    # 5. Initialize search pipeline components
    prefix_retriever = PrefixRetriever(storage, tokenizer)
    trigram_retriever = TrigramRetriever(storage, tokenizer)
    hybrid_retriever = HybridRetriever(prefix_retriever, trigram_retriever)

    # Externalized configuration for weighted ranker
    RANKER_CONFIG = {
        "prefix_weight": 0.7,
        "trigram_weight": 0.3
    }
    ranker = WeightedRanker(RANKER_CONFIG)
    
    pipeline = SearchPipeline(storage, hybrid_retriever, ranker)
    print(f"[Search Engine] Pipeline constructed successfully.")
    print(f" - Configured Weights: Prefix={RANKER_CONFIG['prefix_weight']}, Trigram={RANKER_CONFIG['trigram_weight']}")
    print("-" * 60)

    # 6. Execute required test queries
    test_queries = [
        "deep re",
        "deep resercher",
        "agent",
        "research"
    ]

    for query in test_queries:
        print(f"\nQUERY: '{query}'")
        print("~" * 40)
        
        # Execute search (retrieve, rank, and hydrate Top 5)
        results = pipeline.search(query, limit=5)
        
        if not results:
            print("  No matching documents found.")
            continue

        for i, res in enumerate(results, start=1):
            doc = res.document
            print(f"  {i}. [{doc.type}] {doc.title} (Score: {res.score:.4f})")
            print(f"     URL: {doc.url or 'N/A'}")
            print(f"     Snippet: {doc.content[:100]}...")
            print()

    print("=" * 60)
    print("DEMONSTRATION RUN COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
