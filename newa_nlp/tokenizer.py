"""
Tokenizer module for Newari text processing.
"""

import re
from typing import List, Optional


def tokenize_text(text: str, mode: str = "regex", pattern: Optional[str] = None) -> List[str]:
    """
    Tokenize text for Newari/Devanagari text processing.
    
    Args:
        text: Input text to tokenize
        mode: Tokenization mode - "space" for whitespace splitting, "regex" for regex pattern
        pattern: Custom regex pattern when mode="regex"
        
    Returns:
        List of tokens
    """
    if not text or not text.strip():
        return []
    
    if mode == "space":
        # Simple whitespace tokenization
        tokens = [t.strip() for t in text.split() if t.strip()]
    elif mode == "regex":
        # Regex-based tokenization (default for Devanagari text)
        default_pattern = r"[\u0900-\u0963\u0966-\u097F]+"  # Devanagari characters
        regex_pattern = pattern or default_pattern
        tokens = re.findall(regex_pattern, text)
    else:
        raise ValueError("mode must be 'space' or 'regex'")
    
    # Filter out empty tokens and convert to lowercase for consistency
    return [token.lower() for token in tokens if token.strip()]


def tokenize_sentences(text: str, sentence_delimiters: str = r"[।॥!?]") -> List[str]:
    """
    Split text into sentences using Devanagari punctuation.
    
    Args:
        text: Input text to split into sentences
        sentence_delimiters: Regex pattern for sentence delimiters
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    # Split by sentence delimiters and clean up
    sentences = re.split(sentence_delimiters, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def get_default_devanagari_pattern() -> str:
    """
    Get the default Devanagari regex pattern.
    
    Returns:
        Default regex pattern for Devanagari characters
    """
    return r"[\u0900-\u0963\u0966-\u097F]+"


def is_devanagari_text(text: str) -> bool:
    """
    Check if text contains Devanagari characters.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains Devanagari characters
    """
    devanagari_pattern = get_default_devanagari_pattern()
    return bool(re.search(devanagari_pattern, text))


def clean_text(text: str, remove_punctuation: bool = False) -> str:
    """
    Clean text by removing extra whitespace and optionally punctuation.
    
    Args:
        text: Input text to clean
        remove_punctuation: Whether to remove punctuation marks
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    if remove_punctuation:
        # Remove common punctuation marks
        cleaned = re.sub(r'[।॥!?.,;:()\[\]{}"\'-]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned
