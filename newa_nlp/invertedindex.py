"""
Inverted Index implementation for Newari text retrieval.
"""

import os
import json
import pickle
import csv
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Union
from pathlib import Path

from .tokenizer import tokenize_text


class InvertedIndex:
    """
    A simple inverted index implementation for text retrieval.
    
    The inverted index maps terms to the documents (by filename) that contain them.
    """
    
    def __init__(self):
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self.document_count = 0
        self.total_terms = 0
    
    def add_document(self, doc_id: str, terms: List[str]) -> None:
        """
        Add a document to the inverted index.
        
        Args:
            doc_id: Unique identifier for the document (usually filename)
            terms: List of terms/tokens from the document
        """
        self.document_count += 1
        for term in terms:
            if term.strip():  # Skip empty terms
                self.index[term].add(doc_id)
                self.total_terms += 1
    
    def get_documents(self, term: str) -> Set[str]:
        """
        Get all documents containing a specific term.
        
        Args:
            term: The term to search for
            
        Returns:
            Set of document IDs containing the term
        """
        return self.index.get(term, set())
    
    def search(self, query_terms: List[str], operation: str = "AND") -> Set[str]:
        """
        Search for documents containing query terms.
        
        Args:
            query_terms: List of terms to search for
            operation: "AND" or "OR" operation for combining results
            
        Returns:
            Set of document IDs matching the query
        """
        if not query_terms:
            return set()
        
        results = self.get_documents(query_terms[0])
        
        for term in query_terms[1:]:
            term_docs = self.get_documents(term)
            if operation.upper() == "AND":
                results = results.intersection(term_docs)
            elif operation.upper() == "OR":
                results = results.union(term_docs)
            else:
                raise ValueError("Operation must be 'AND' or 'OR'")
        
        return results
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the inverted index.
        
        Returns:
            Dictionary with index statistics
        """
        unique_terms = len(self.index)
        avg_terms_per_doc = self.total_terms / self.document_count if self.document_count > 0 else 0
        
        return {
            "document_count": self.document_count,
            "unique_terms": unique_terms,
            "total_terms": self.total_terms,
            "average_terms_per_document": avg_terms_per_doc
        }
    
    def to_dict(self) -> Dict[str, List[str]]:
        """
        Convert the index to a dictionary with lists instead of sets for serialization.
        
        Returns:
            Dictionary representation of the index
        """
        return {term: list(docs) for term, docs in self.index.items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, List[str]], doc_count: int = 0) -> 'InvertedIndex':
        """
        Create an InvertedIndex from a dictionary representation.
        
        Args:
            data: Dictionary with terms as keys and document lists as values
            doc_count: Number of documents (if known)
            
        Returns:
            InvertedIndex instance
        """
        index = cls()
        index.document_count = doc_count
        for term, docs in data.items():
            index.index[term] = set(docs)
            index.total_terms += len(docs)
        return index


def build_inverted_index_from_csv(
    csv_path: str,
    doc_id_column: str = "filename",
    content_column: str = "content",
    tokenizer_mode: str = "regex",
    regex_pattern: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> InvertedIndex:
    """
    Build an inverted index from a CSV file containing documents.
    
    Args:
        csv_path: Path to the CSV file
        doc_id_column: Name of the column containing document IDs
        content_column: Name of the column containing text content
        tokenizer_mode: Tokenization mode ("space" or "regex")
        regex_pattern: Custom regex pattern for tokenization
        progress_callback: Optional callback function for progress updates
        
    Returns:
        InvertedIndex object
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    index = InvertedIndex()
    
    # Read CSV file
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Validate columns
        if doc_id_column not in reader.fieldnames:
            raise ValueError(f"Column '{doc_id_column}' not found. Available columns: {reader.fieldnames}")
        if content_column not in reader.fieldnames:
            raise ValueError(f"Column '{content_column}' not found. Available columns: {reader.fieldnames}")
        
        # Count total rows for progress tracking
        csvfile.seek(0)
        total_rows = sum(1 for _ in csv.DictReader(csvfile)) - 1  # Subtract header
        csvfile.seek(0)
        reader = csv.DictReader(csvfile)
        
        processed = 0
        
        for row in reader:
            try:
                doc_id = row[doc_id_column]
                content = row[content_column]
                
                # Tokenize the content
                terms = tokenize_text(content, mode=tokenizer_mode, pattern=regex_pattern)
                
                # Add to index
                index.add_document(doc_id, terms)
                
                processed += 1
                
                # Progress callback
                if processed % 1000 == 0 or processed == total_rows:
                    if progress_callback:
                        progress_callback(processed, total_rows, f"Processed {processed} documents")
                    else:
                        print(f"[{processed}/{total_rows}] Processed {processed} documents")
                        
            except Exception as e:
                print(f"Error processing document {row.get(doc_id_column, 'unknown')}: {str(e)}")
                continue
    
    print(f"Inverted index built successfully!")
    print(f"Documents processed: {index.document_count}")
    print(f"Unique terms: {len(index.index)}")
    print(f"Total term occurrences: {index.total_terms}")
    
    return index


def save_inverted_index(
    index: InvertedIndex,
    output_path: str,
    format: str = "json"
) -> None:
    """
    Save an inverted index to a file.
    
    Args:
        index: InvertedIndex object to save
        output_path: Path where to save the index
        format: Output format - "json" or "pickle"
    """
    if format not in ["json", "pickle"]:
        raise ValueError("format must be 'json' or 'pickle'")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Convert sets to lists for JSON serialization
        data = {
            "index": index.to_dict(),
            "document_count": index.document_count,
            "total_terms": index.total_terms
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    elif format == "pickle":
        with open(output_path, 'wb') as f:
            pickle.dump(index, f)
    
    print(f"Inverted index saved to: {output_path}")


def load_inverted_index(
    input_path: str,
    format: str = "json"
) -> InvertedIndex:
    """
    Load an inverted index from a file.
    
    Args:
        input_path: Path to the saved index file
        format: File format - "json" or "pickle"
        
    Returns:
        InvertedIndex object
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Index file not found: {input_path}")
    
    if format not in ["json", "pickle"]:
        raise ValueError("format must be 'json' or 'pickle'")
    
    if format == "json":
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        index = InvertedIndex.from_dict(
            data["index"],
            doc_count=data.get("document_count", 0)
        )
        index.total_terms = data.get("total_terms", 0)
    
    elif format == "pickle":
        with open(input_path, 'rb') as f:
            index = pickle.load(f)
    
    print(f"Inverted index loaded from: {input_path}")
    return index


def build_and_save_inverted_index(
    csv_path: str,
    output_path: str,
    doc_id_column: str = "filename",
    content_column: str = "content",
    tokenizer_mode: str = "regex",
    regex_pattern: Optional[str] = None,
    output_format: str = "json",
    progress_callback: Optional[callable] = None
) -> InvertedIndex:
    """
    Build an inverted index from a CSV file and save it to disk.
    
    This is a convenience function that combines building and saving the index.
    
    Args:
        csv_path: Path to the input CSV file
        output_path: Path where to save the index
        doc_id_column: Name of the column containing document IDs
        content_column: Name of the column containing text content
        tokenizer_mode: Tokenization mode ("space" or "regex")
        regex_pattern: Custom regex pattern for tokenization
        output_format: Output format - "json" or "pickle"
        progress_callback: Optional callback function for progress updates
        
    Returns:
        InvertedIndex object
    """
    print(f"Building inverted index from: {csv_path}")
    
    # Build the index
    index = build_inverted_index_from_csv(
        csv_path=csv_path,
        doc_id_column=doc_id_column,
        content_column=content_column,
        tokenizer_mode=tokenizer_mode,
        regex_pattern=regex_pattern,
        progress_callback=progress_callback
    )
    
    # Save the index
    save_inverted_index(index, output_path, format=output_format)
    
    return index
