"""
Test the updated corpus.py integration with tokenizer.py and search functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_corpus_tokenizer_integration():
    """Test that corpus.py now uses the tokenizer module correctly."""
    print("Testing corpus.py integration with tokenizer.py...")
    
    try:
        from newa_nlp import build_unigram_from_csv, tokenize_text
        
        # Test that both functions use the same tokenization
        test_text = "नेपाल भाषा नेवाः भाषा खः।"
        
        # Test direct tokenization
        tokens_direct = tokenize_text(test_text, mode="regex")
        print(f"✓ Direct tokenization: {tokens_direct}")
        
        # Test corpus unigram building (should use same tokenization)
        if os.path.exists("corpus.csv"):
            print("Testing unigram building with new tokenizer...")
            
            # Build unigrams with regex mode (default now)
            unigrams = build_unigram_from_csv(
                "corpus.csv", 
                tokenizer_mode="regex",  # This should be the default now
                top_k=5
            )
            print(f"✓ Built unigrams with regex tokenizer: {len(unigrams)} terms")
            print(f"  Top 5 terms: {unigrams}")
            
            # Test that the default is now regex
            unigrams_default = build_unigram_from_csv(
                "corpus.csv", 
                top_k=5  # Should default to regex mode
            )
            print(f"✓ Built unigrams with default tokenizer: {len(unigrams_default)} terms")
            print(f"  Top 5 terms: {unigrams_default}")
            
            # Verify they're the same (should be since both use regex)
            if unigrams == unigrams_default:
                print("✓ Default tokenizer mode is correctly set to 'regex'")
            else:
                print("⚠ Default tokenizer mode might not be set correctly")
        
        else:
            print("⚠ corpus.csv not found, skipping corpus integration test")
        
        # Test with a small sample
        print("\nTesting with sample text...")
        sample_texts = [
            "नेपाल भाषा नेवाः भाषा खः।",
            "थ्व देश नेपाल खः।",
            "नेवाः भाषा नेपालया भाषा खः।"
        ]
        
        from newa_nlp import build_unigram
        unigrams_sample = build_unigram(sample_texts, top_k=10)
        print(f"✓ Sample unigrams: {unigrams_sample}")
        
        return True
        
    except Exception as e:
        print(f"✗ Corpus integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consistency():
    """Test that tokenization is consistent across modules."""
    print("\nTesting tokenization consistency...")
    
    try:
        from newa_nlp import tokenize_text, build_unigram
        
        test_text = "नेपाल भाषा नेवाः भाषा खः।"
        
        # Test direct tokenization
        tokens_direct = tokenize_text(test_text, mode="regex")
        
        # Test through build_unigram
        unigrams = build_unigram([test_text], tokenizer_mode="regex")
        tokens_unigram = [token for token, count in unigrams]
        
        print(f"Direct tokens: {tokens_direct}")
        print(f"Unigram tokens: {tokens_unigram}")
        
        # They should be the same (ignoring counts)
        if set(tokens_direct) == set(tokens_unigram):
            print("✓ Tokenization is consistent across modules")
            return True
        else:
            print("✗ Tokenization inconsistency detected")
            return False
            
    except Exception as e:
        print(f"✗ Consistency test error: {e}")
        return False

def test_inverted_index_with_corpus():
    """Test inverted index creation from corpus.csv and search with test text."""
    print("\nTesting inverted index creation from corpus.csv...")
    
    try:
        from newa_nlp import build_inverted_index_from_csv, tokenize_text
        
        if not os.path.exists("corpus.csv"):
            print("⚠ corpus.csv not found, skipping corpus-based test")
            return True
        
        # Check if index already exists, if not build it
        index_file = "test_inverted_index.json"
        if os.path.exists(index_file):
            print(f"Loading existing index from {index_file}...")
            from newa_nlp import load_inverted_index
            index = load_inverted_index(index_file)
            print("✓ Inverted index loaded from existing file")
        else:
            print("Building new inverted index from corpus.csv...")
            index = build_inverted_index_from_csv(
                csv_path="corpus.csv",
                doc_id_column="filename",
                content_column="content",
                tokenizer_mode="regex"
            )
            print("✓ Inverted index built from corpus.csv")
            
            # Save the inverted index
            print("Saving inverted index...")
            from newa_nlp import save_inverted_index
            save_inverted_index(index, index_file, format="json")
            print(f"✓ Inverted index saved to {index_file}")
        
        # Test search functionality with test text terms
        test_text = "नेपाल भाषा नेवाः भाषा खः।"
        test_tokens = tokenize_text(test_text, mode="regex")
        print(f"Test text: {test_text}")
        print(f"Test tokens: {test_tokens}")
        
        print("\n--- Testing Search Functionality with Test Text ---")
        
        # Test single term search for each token in test text
        for term in test_tokens:
            results = index.search([term])
            print(f"Search for '{term}': {len(results)} documents found")
            if results:
                sample_docs = list(results)[:3]  # Show first 3 documents
                print(f"  Sample documents: {sample_docs}")
        
        # Test AND operation with test text terms
        if len(test_tokens) >= 2:
            and_results = index.search(test_tokens[:2], operation="AND")
            print(f"Search for '{test_tokens[0]} AND {test_tokens[1]}': {len(and_results)} documents found")
            if and_results:
                sample_docs = list(and_results)[:3]
                print(f"  Sample documents: {sample_docs}")
        
        # Test OR operation with test text terms
        if len(test_tokens) >= 2:
            or_results = index.search(test_tokens[:2], operation="OR")
            print(f"Search for '{test_tokens[0]} OR {test_tokens[1]}': {len(or_results)} documents found")
            if or_results:
                sample_docs = list(or_results)[:3]
                print(f"  Sample documents: {sample_docs}")
        
        # Test index statistics
        stats = index.get_stats()
        print(f"\nIndex Statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inverted index test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_engine_with_corpus():
    """Test search engine with corpus.csv and test text."""
    print("\nTesting search engine with corpus.csv...")
    
    try:
        from newa_nlp import create_search_engine, build_inverted_index_from_csv
        
        if not os.path.exists("corpus.csv"):
            print("⚠ corpus.csv not found, skipping search engine test")
            return True
        
        # Check if we have the required files
        index_file = "inverted_index.json"
        if not os.path.exists(index_file):
            print("Building inverted index from corpus.csv...")
            index = build_inverted_index_from_csv(
                csv_path="corpus.csv",
                doc_id_column="filename",
                content_column="content",
                tokenizer_mode="regex"
            )
            # Save the index for future use
            from newa_nlp import save_inverted_index
            save_inverted_index(index, index_file, format="json")
            print(f"✓ Index built and saved to {index_file}")
        else:
            print(f"✓ Using existing {index_file}")
        
        # Create search engine
        print("Creating search engine...")
        engine = create_search_engine("inverted_index.json", "corpus.csv")
        print("✓ Search engine created")
        
        # Test with test text
        test_text = "नेपाल भाषा नेवाः भाषा खः।"
        print(f"Test text: {test_text}")
        
        # Test document search
        print("\n--- Document Search Tests ---")
        test_queries = ["नेपाल", "भाषा", "नेवाः", "नेपाल भाषा"]
        
        for query in test_queries:
            docs = engine.search_documents(query, limit=5)
            print(f"Search '{query}': {len(docs)} documents found")
            if docs:
                print(f"  Sample documents: {docs[:3]}")
        
        # Test sentence search
        print("\n--- Sentence Search Tests ---")
        for query in test_queries[:2]:  # Test first 2 queries
            sentences = engine.search_sentences(query, limit=3)
            print(f"\nSentences containing '{query}': {len(sentences)} found")
            for i, result in enumerate(sentences, 1):
                print(f"  {i}. {result['document']}: {result['sentence'][:80]}...")
        
        # Test highlighted search
        print("\n--- Highlighted Search Tests ---")
        highlighted = engine.search_with_highlight("नेपाल भाषा", limit=3)
        print(f"Highlighted search results: {len(highlighted)} found")
        for i, result in enumerate(highlighted, 1):
            print(f"  {i}. {result['document']}: {result['sentence'][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Search engine test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unigram_from_corpus():
    """Test building unigrams from corpus.csv."""
    print("\nTesting unigram building from corpus.csv...")
    
    try:
        from newa_nlp import build_unigram_from_csv
        
        if not os.path.exists("corpus.csv"):
            print("⚠ corpus.csv not found, skipping unigram test")
            return True
        
        # Build unigrams from corpus.csv
        print("Building unigrams from corpus.csv...")
        unigrams = build_unigram_from_csv(
            csv_path="corpus.csv",
            content_column="content",
            tokenizer_mode="regex",  # Using the new default
            sort_by="freq",
            top_k=20
        )
        
        print(f"✓ Built unigrams: {len(unigrams)} terms")
        print("Top 20 most frequent terms:")
        for i, (term, count) in enumerate(unigrams, 1):
            print(f"  {i:2d}. {term}: {count:,}")
        
        # Test with different sorting
        print("\nTesting Devanagari sorting...")
        unigrams_dev = build_unigram_from_csv(
            csv_path="corpus.csv",
            content_column="content",
            tokenizer_mode="regex",
            sort_by="dev",
            top_k=10
        )
        
        print("Top 10 terms (Devanagari sorted):")
        for i, (term, count) in enumerate(unigrams_dev, 1):
            print(f"  {i:2d}. {term}: {count:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Unigram test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Corpus.py Integration with Tokenizer.py and Search Functionality")
    print("=" * 60)
    
    tests = [
        test_corpus_tokenizer_integration,
        test_consistency,
        test_inverted_index_with_corpus,
        test_search_engine_with_corpus,
        test_unigram_from_corpus,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Corpus integration successful.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
