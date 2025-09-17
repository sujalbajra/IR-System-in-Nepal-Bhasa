"""
Corpus processing utilities for Newari text data.
"""

import os
import csv
import glob
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Callable, Iterable, List, Tuple


def on_progress(current: int, total: int, message: str) -> None:
    """
    Default progress reporter: prints a concise progress line.
    """
    print(f"[{current}/{total}] {message}")


def create_corpus_csv(corpus_dir: str, output_csv: str, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> None:
    """
    Create a CSV file with filename and content columns from all text files in the corpus folder.
    
    Args:
        corpus_dir (str): Directory containing .txt files
        output_csv (str): Path to the output CSV file
        progress_callback (callable, optional): Function to call for progress updates
        
    Raises:
        FileNotFoundError: If corpus_dir doesn't exist
        ValueError: If no .txt files found in corpus_dir
    """
    # Validate input directory
    if not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    # Get all .txt files in the corpus directory
    txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    if not txt_files:
        raise ValueError(f"No .txt files found in directory: {corpus_dir}")
    
    total = len(txt_files)
    print(f"Found {total} text files to process...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Process each text file
        for i, file_path in enumerate(txt_files):
            try:
                # Get just the filename without path
                filename = os.path.basename(file_path)
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as txtfile:
                    content = txtfile.read().strip()
                
                # Write to CSV
                writer.writerow({
                    'filename': filename,
                    'content': content
                })
                
                # Progress indicator
                if (i + 1) % 1000 == 0:
                    progress_msg = f"Processed {i + 1} files..."
                    print(progress_msg)
                    if progress_callback:
                        progress_callback(i + 1, total, progress_msg)
                    else:
                        on_progress(i + 1, total, progress_msg)
                    
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                print(error_msg)
                if progress_callback:
                    progress_callback(i + 1, total, error_msg)
                else:
                    on_progress(i + 1, total, error_msg)
                continue
    
    print(f"CSV file created successfully: {output_csv}")
    print(f"Total files processed: {total}")


def get_corpus_stats(corpus_dir: str) -> dict:
    """
    Get statistics about the corpus directory.
    
    Args:
        corpus_dir (str): Directory containing .txt files
        
    Returns:
        dict: Statistics including file count, total size, etc.
    """
    if not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    if not txt_files:
        return {"file_count": 0, "total_size": 0, "files": []}
    
    total_size = 0
    file_stats = []
    
    for file_path in txt_files:
        try:
            size = os.path.getsize(file_path)
            total_size += size
            file_stats.append({
                "filename": os.path.basename(file_path),
                "size": size
            })
        except OSError:
            continue
    
    return {
        "file_count": len(txt_files),
        "total_size": total_size,
        "average_size": total_size / len(txt_files) if txt_files else 0,
        "files": file_stats
    }


# -------------------- Unigram utilities --------------------

def _tokenize(text: str, mode: str = "space", pattern: Optional[str] = None) -> List[str]:
    """
    Tokenize text either by spaces or using a regex pattern.
    - mode="space": split on whitespace
    - mode="regex": use provided pattern or a default word pattern
    """
    if mode not in {"space", "regex"}:
        raise ValueError("mode must be 'space' or 'regex'")
    if mode == "space":
        return [t for t in text.split() if t]
    # default regex: Devanagari characters excluding danda (।) and other punctuation
    # This matches your notebook pattern: [\u0900-\u0963\u0965-\u097F]+
    default_pattern = r"[\u0900-\u0963\u0965-\u097F]+"
    compiled = re.compile(pattern or default_pattern)
    return compiled.findall(text)


def _devanagari_sort_key(token: str) -> Tuple:
    """
    Sort key that prioritizes Devanagari codepoints, then unicode codepoints for stability.
    This approximates alphabetical order for Devanagari.
    """
    return tuple(ord(ch) for ch in token)


def build_unigram(
    texts: Iterable[str],
    tokenizer_mode: str = "space",
    regex_pattern: Optional[str] = None,
    sort_by: str = "freq",
    top_k: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Build unigram frequencies from an iterable of texts.
    
    Args:
        texts: Iterable of input strings
        tokenizer_mode: 'space' or 'regex'
        regex_pattern: custom regex when tokenizer_mode='regex'
        sort_by: 'freq' or 'dev'
        top_k: if provided, return only top_k items after sorting
    
    Returns:
        List of (token, count) tuples
    """
    counter: Counter = Counter()
    for text in texts:
        if not text or not text.strip():
            continue
        tokens = _tokenize(text, mode=tokenizer_mode, pattern=regex_pattern)
        # Filter out empty tokens
        tokens = [t for t in tokens if t and t.strip()]
        if not tokens:
            continue
        counter.update(tokens)
    
    items = list(counter.items())
    if sort_by == "freq":
        items.sort(key=lambda x: (-x[1], x[0]))
    elif sort_by == "dev":
        items.sort(key=lambda x: _devanagari_sort_key(x[0]))
    else:
        raise ValueError("sort_by must be 'freq' or 'dev'")
    
    if top_k is not None:
        items = items[:top_k]
    return items


def build_unigram_from_csv(
    csv_path: str,
    content_column: str = "content",
    tokenizer_mode: str = "space",
    regex_pattern: Optional[str] = None,
    sort_by: str = "freq",
    top_k: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Build unigram frequencies from a CSV file containing text content.
    
    Args:
        csv_path: Path to CSV file
        content_column: Name of the column containing text content
        tokenizer_mode: 'space' or 'regex'
        regex_pattern: custom regex when tokenizer_mode='regex'
        sort_by: 'freq' or 'dev'
        top_k: if provided, return only top_k items after sorting
    
    Returns:
        List of (token, count) tuples
    """
    import pandas as pd
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    if content_column not in df.columns:
        raise ValueError(f"Column '{content_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Get texts from the specified column
    texts = df[content_column].astype(str).tolist()
    
    return build_unigram(
        texts=texts,
        tokenizer_mode=tokenizer_mode,
        regex_pattern=regex_pattern,
        sort_by=sort_by,
        top_k=top_k,
    )


def save_unigram_to_csv(unigrams: List[Tuple[str, int]], output_csv: str) -> None:
    """Save unigram (token,count) pairs to a CSV file with headers token,count."""
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count"])
        for token, count in unigrams:
            writer.writerow([token, count])
