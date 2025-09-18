"""
Standalone script to build and save inverted index from corpus.csv
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def build_and_save_index():
    """Build and save inverted index from corpus.csv"""
    print("=" * 60)
    print("Building Inverted Index from Corpus")
    print("=" * 60)
    
    try:
        from newa_nlp import build_and_save_inverted_index
        
        # Check if corpus.csv exists
        if not os.path.exists("corpus.csv"):
            print("❌ Error: corpus.csv not found!")
            print("Please make sure corpus.csv is in the current directory.")
            return False
        
        print("📁 Found corpus.csv")
        print("🔨 Building inverted index...")
        
        # Build and save the inverted index
        index = build_and_save_inverted_index(
            csv_path="corpus.csv",
            output_path="inverted_index.json",
            doc_id_column="filename",
            content_column="content",
            tokenizer_mode="regex",
            output_format="json"
        )
        
        print("\n✅ Inverted index built and saved successfully!")
        
        # Display statistics
        stats = index.get_stats()
        print("\n📊 Index Statistics:")
        print(f"  Documents: {stats['document_count']:,}")
        print(f"  Unique terms: {stats['unique_terms']:,}")
        print(f"  Total term occurrences: {stats['total_terms']:,}")
        print(f"  Average terms per document: {stats['average_terms_per_document']:.1f}")
        
        # Test the index with a sample search
        print("\n🔍 Testing index with sample search...")
        test_terms = ["नेपाल", "भाषा", "नेवाः"]
        
        for term in test_terms:
            results = index.search([term])
            print(f"  '{term}': {len(results):,} documents")
        
        print(f"\n💾 Index saved to: inverted_index.json")
        print("🎉 Ready for search operations!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error building index: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_and_test_index():
    """Load existing index and test it"""
    print("\n" + "=" * 60)
    print("Testing Existing Index")
    print("=" * 60)
    
    try:
        from newa_nlp import load_inverted_index
        
        if not os.path.exists("inverted_index.json"):
            print("❌ Error: inverted_index.json not found!")
            print("Run build_and_save_index() first to create the index.")
            return False
        
        print("📁 Loading existing inverted_index.json...")
        index = load_inverted_index("inverted_index.json")
        
        # Display statistics
        stats = index.get_stats()
        print("\n📊 Index Statistics:")
        print(f"  Documents: {stats['document_count']:,}")
        print(f"  Unique terms: {stats['unique_terms']:,}")
        print(f"  Total term occurrences: {stats['total_terms']:,}")
        print(f"  Average terms per document: {stats['average_terms_per_document']:.1f}")
        
        # Test searches
        print("\n🔍 Testing searches...")
        test_queries = [
            ["नेपाल"],
            ["भाषा"],
            ["नेवाः"],
            ["नेपाल", "भाषा"],  # AND search
        ]
        
        for query in test_queries:
            if len(query) == 1:
                results = index.search(query)
                print(f"  '{query[0]}': {len(results):,} documents")
            else:
                results = index.search(query, operation="AND")
                print(f"  '{' AND '.join(query)}': {len(results):,} documents")
        
        print("✅ Index loaded and tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading index: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to build or test index"""
    print("Newa NLP - Inverted Index Builder")
    print("=" * 60)
    
    # Check if index already exists
    if os.path.exists("inverted_index.json"):
        print("📁 Found existing inverted_index.json")
        choice = input("Do you want to rebuild the index? (y/N): ").strip().lower()
        
        if choice in ['y', 'yes']:
            print("🔄 Rebuilding index...")
            success = build_and_save_index()
        else:
            print("📖 Testing existing index...")
            success = load_and_test_index()
    else:
        print("📁 No existing index found")
        print("🔨 Building new index...")
        success = build_and_save_index()
    
    if success:
        print("\n🎉 Operation completed successfully!")
    else:
        print("\n❌ Operation failed!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
