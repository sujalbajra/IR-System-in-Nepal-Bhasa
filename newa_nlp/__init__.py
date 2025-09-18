"""
Newa NLP Library

A Python library for Newari Natural Language Processing tasks.
"""

__version__ = "0.1.0"
__author__ = "Newa NLP Team"

# Import corpus utilities
from .corpus import (
    on_progress,
    create_corpus_csv,
    get_corpus_stats,
    build_unigram,
    build_unigram_from_csv,
    save_unigram_to_csv,
)

# Import tokenizer utilities
from .tokenizer import (
    tokenize_text,
    tokenize_sentences,
    get_default_devanagari_pattern,
    is_devanagari_text,
    clean_text,
)

# Import inverted index
from .invertedindex import (
    InvertedIndex,
    build_inverted_index_from_csv,
    save_inverted_index,
    load_inverted_index,
    build_and_save_inverted_index,
)

# Import search engine
from .search import (
    SearchEngine,
    create_search_engine,
)

# Import embeddings utilities
from .embeddings import (
    SentenceEncoder,
    cosine_similarity,
    top_k_similar,
)

__all__ = [
    # Corpus utilities
    "on_progress",
    "create_corpus_csv",
    "get_corpus_stats",
    "build_unigram",
    "build_unigram_from_csv",
    "save_unigram_to_csv",
    
    # Tokenizer utilities
    "tokenize_text",
    "tokenize_sentences",
    "get_default_devanagari_pattern",
    "is_devanagari_text",
    "clean_text",
    
    # Inverted index
    "InvertedIndex",
    "build_inverted_index_from_csv",
    "save_inverted_index",
    "load_inverted_index",
    "build_and_save_inverted_index",
    
    # Search engine
    "SearchEngine",
    "create_search_engine",

    # Embeddings utilities
    "SentenceEncoder",
    "cosine_similarity",
    "top_k_similar",
]
