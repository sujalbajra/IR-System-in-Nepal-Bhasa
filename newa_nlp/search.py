"""
Search functionality for Newari text retrieval.
"""

import os
import csv
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from .invertedindex import InvertedIndex, load_inverted_index
from .tokenizer import tokenize_text, tokenize_sentences


class SearchEngine:
    """
    A search engine for Newari text using inverted index.
    """
    
    def __init__(self, index: Optional[InvertedIndex] = None, corpus_csv: Optional[str] = None):
        """
        Initialize the search engine.
        
        Args:
            index: Pre-built inverted index (optional)
            corpus_csv: Path to corpus CSV file for document retrieval (optional)
        """
        self.index = index
        self.corpus_csv = corpus_csv
        self._document_cache = {}
    
    def load_index(self, index_path: str, format: str = "json") -> None:
        """
        Load an inverted index from file.
        
        Args:
            index_path: Path to the index file
            format: File format ("json" or "pickle")
        """
        self.index = load_inverted_index(index_path, format)
    
    def set_corpus_csv(self, corpus_csv: str) -> None:
        """
        Set the corpus CSV file for document retrieval.
        
        Args:
            corpus_csv: Path to the corpus CSV file
        """
        self.corpus_csv = corpus_csv
    
    def search_documents(self, query: str, operation: str = "AND", limit: Optional[int] = None) -> List[str]:
        """
        Search for documents containing the query terms.
        
        Args:
            query: Search query string
            operation: "AND" or "OR" operation for combining terms
            limit: Maximum number of results to return
            
        Returns:
            List of document IDs matching the query
        """
        if not self.index:
            raise ValueError("No index loaded. Use load_index() or provide index during initialization.")
        
        # Tokenize the query
        query_terms = tokenize_text(query, mode="regex")
        
        if not query_terms:
            return []
        
        # Search the index
        results = self.index.search(query_terms, operation=operation)
        
        # Convert to list and limit results
        result_list = list(results)
        if limit:
            result_list = result_list[:limit]
        
        return result_list
    
    def search_sentences(self, query: str, operation: str = "AND", limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Search for sentences containing the query terms.
        
        Args:
            query: Search query string
            operation: "AND" or "OR" operation for combining terms
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with 'document', 'sentence', and 'context' keys
        """
        if not self.index or not self.corpus_csv:
            raise ValueError("Both index and corpus CSV must be available for sentence search.")
        
        # Get matching documents
        doc_ids = self.search_documents(query, operation=operation)
        
        if not doc_ids:
            return []
        
        # Tokenize query terms
        query_terms = tokenize_text(query, mode="regex")
        query_terms_lower = [term.lower() for term in query_terms]
        
        sentence_results = []
        
        # Read corpus and find matching sentences
        with open(self.corpus_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                doc_id = row.get('filename', '')
                if doc_id not in doc_ids:
                    continue
                
                content = row.get('content', '')
                if not content:
                    continue
                
                # Split into sentences
                sentences = tokenize_sentences(content)
                
                for sentence in sentences:
                    # Check if sentence contains query terms
                    sentence_tokens = tokenize_text(sentence, mode="regex")
                    sentence_tokens_lower = [token.lower() for token in sentence_tokens]
                    
                    if operation.upper() == "AND":
                        # All query terms must be present
                        if all(term in sentence_tokens_lower for term in query_terms_lower):
                            sentence_results.append({
                                'document': doc_id,
                                'sentence': sentence.strip(),
                                'context': self._get_sentence_context(content, sentence)
                            })
                    else:  # OR operation
                        # At least one query term must be present
                        if any(term in sentence_tokens_lower for term in query_terms_lower):
                            sentence_results.append({
                                'document': doc_id,
                                'sentence': sentence.strip(),
                                'context': self._get_sentence_context(content, sentence)
                            })
                    
                    # Limit results
                    if limit and len(sentence_results) >= limit:
                        return sentence_results[:limit]
        
        return sentence_results
    
    def _get_sentence_context(self, content: str, sentence: str, context_length: int = 100) -> str:
        """
        Get context around a sentence.
        
        Args:
            content: Full document content
            sentence: The sentence to get context for
            context_length: Length of context to include
            
        Returns:
            Context string around the sentence
        """
        sentence_start = content.find(sentence)
        if sentence_start == -1:
            return sentence
        
        # Get context before and after
        context_start = max(0, sentence_start - context_length)
        context_end = min(len(content), sentence_start + len(sentence) + context_length)
        
        context = content[context_start:context_end]
        
        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(content):
            context = context + "..."
        
        return context
    
    def get_document_content(self, doc_id: str) -> Optional[str]:
        """
        Get the full content of a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document content or None if not found
        """
        if not self.corpus_csv:
            return None
        
        # Check cache first
        if doc_id in self._document_cache:
            return self._document_cache[doc_id]
        
        # Read from CSV
        with open(self.corpus_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                if row.get('filename') == doc_id:
                    content = row.get('content', '')
                    self._document_cache[doc_id] = content
                    return content
        
        return None
    
    def get_index_stats(self) -> Dict[str, int]:
        """
        Get statistics about the loaded index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            return {}
        
        return self.index.get_stats()
    
    def search_with_highlight(self, query: str, operation: str = "AND", limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Search for sentences with highlighted query terms.
        
        Args:
            query: Search query string
            operation: "AND" or "OR" operation for combining terms
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with highlighted sentences
        """
        results = self.search_sentences(query, operation=operation, limit=limit)
        
        # Tokenize query terms for highlighting
        query_terms = tokenize_text(query, mode="regex")
        
        # Highlight query terms in results
        highlighted_results = []
        for result in results:
            highlighted_sentence = self._highlight_terms(result['sentence'], query_terms)
            highlighted_context = self._highlight_terms(result['context'], query_terms)
            
            highlighted_results.append({
                'document': result['document'],
                'sentence': highlighted_sentence,
                'context': highlighted_context,
                'original_sentence': result['sentence']
            })
        
        return highlighted_results
    
    def _highlight_terms(self, text: str, terms: List[str]) -> str:
        """
        Highlight query terms in text.
        
        Args:
            text: Text to highlight terms in
            terms: List of terms to highlight
            
        Returns:
            Text with highlighted terms
        """
        highlighted_text = text
        
        for term in terms:
            # Case-insensitive highlighting
            import re
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_text = pattern.sub(f"**{term}**", highlighted_text)
        
        return highlighted_text


def create_search_engine(index_path: str, corpus_csv: str, format: str = "json") -> SearchEngine:
    """
    Create a search engine with loaded index and corpus.
    
    Args:
        index_path: Path to the inverted index file
        corpus_csv: Path to the corpus CSV file
        format: Index file format ("json" or "pickle")
        
    Returns:
        Configured SearchEngine instance
    """
    engine = SearchEngine()
    engine.load_index(index_path, format)
    engine.set_corpus_csv(corpus_csv)
    return engine


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        # Create search engine
        engine = create_search_engine("inverted_index.json", "corpus.csv")
        
        # Search for documents
        print("Searching for documents containing 'नेपाल':")
        docs = engine.search_documents("नेपाल", limit=5)
        print(f"Found {len(docs)} documents")
        for doc in docs:
            print(f"  - {doc}")
        
        # Search for sentences
        print("\nSearching for sentences containing 'नेपाल':")
        sentences = engine.search_sentences("नेपाल", limit=3)
        for i, result in enumerate(sentences, 1):
            print(f"\n{i}. Document: {result['document']}")
            print(f"   Sentence: {result['sentence']}")
            print(f"   Context: {result['context'][:200]}...")
        
        # Search with highlighting
        print("\nSearching with highlighting:")
        highlighted = engine.search_with_highlight("नेपाल भाषा", limit=2)
        for i, result in enumerate(highlighted, 1):
            print(f"\n{i}. Document: {result['document']}")
            print(f"   Highlighted: {result['sentence']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
