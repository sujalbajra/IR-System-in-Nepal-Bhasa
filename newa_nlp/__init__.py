"""
Newa NLP Library

A Python library for Newari Natural Language Processing tasks.
"""

__version__ = "0.1.0"
__author__ = "Newa NLP Team"

from .corpus import (
    on_progress,
    create_corpus_csv,
    get_corpus_stats,
    build_unigram,
    build_unigram_from_csv,
    save_unigram_to_csv,
)

__all__ = [
    "on_progress",
    "create_corpus_csv",
    "get_corpus_stats",
    "build_unigram",
    "build_unigram_from_csv",
    "save_unigram_to_csv",
]
