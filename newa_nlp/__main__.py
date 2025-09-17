"""
Command-line interface for newa_nlp library.
"""

import argparse
import sys
from .corpus import (
    create_corpus_csv,
    get_corpus_stats,
    build_unigram,
    build_unigram_from_csv,
    save_unigram_to_csv,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Newa NLP - Newari Natural Language Processing tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Corpus CSV creation command
    csv_parser = subparsers.add_parser('create-csv', help='Create CSV from corpus directory')
    csv_parser.add_argument('input_dir', help='Input directory containing .txt files')
    csv_parser.add_argument('output_csv', help='Output CSV file path')
    
    # Corpus stats command
    stats_parser = subparsers.add_parser('stats', help='Get corpus statistics')
    stats_parser.add_argument('corpus_dir', help='Corpus directory to analyze')

    # Unigram command
    uni_parser = subparsers.add_parser('unigram', help='Build unigram from text files')
    uni_parser.add_argument('input_dir', help='Input directory containing .txt files')
    uni_parser.add_argument('output_csv', help='Output CSV file (token,count)')
    uni_parser.add_argument('--mode', choices=['space', 'regex'], default='space', help='Tokenizer mode')
    uni_parser.add_argument('--pattern', type=str, default=None, help='Custom regex when mode=regex')
    uni_parser.add_argument('--sort-by', choices=['freq', 'dev'], default='freq', help='Sort order')
    uni_parser.add_argument('--top-k', type=int, default=None, help='Limit number of rows')
    
    # Unigram from CSV command
    uni_csv_parser = subparsers.add_parser('unigram-csv', help='Build unigram from CSV file')
    uni_csv_parser.add_argument('input_csv', help='Input CSV file')
    uni_csv_parser.add_argument('output_csv', help='Output CSV file (token,count)')
    uni_csv_parser.add_argument('--content-column', default='content', help='Name of content column')
    uni_csv_parser.add_argument('--mode', choices=['space', 'regex'], default='space', help='Tokenizer mode')
    uni_csv_parser.add_argument('--pattern', type=str, default=None, help='Custom regex when mode=regex')
    uni_csv_parser.add_argument('--sort-by', choices=['freq', 'dev'], default='freq', help='Sort order')
    uni_csv_parser.add_argument('--top-k', type=int, default=None, help='Limit number of rows')
    
    args = parser.parse_args()
    
    if args.command == 'create-csv':
        try:
            create_corpus_csv(args.input_dir, args.output_csv)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == 'stats':
        try:
            stats = get_corpus_stats(args.corpus_dir)
            print(f"File count: {stats['file_count']}")
            print(f"Total size: {stats['total_size']:,} bytes")
            print(f"Average file size: {stats['average_size']:.2f} bytes")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == 'unigram':
        try:
            # Read texts from input_dir
            import glob, os
            txt_files = glob.glob(os.path.join(args.input_dir, '*.txt'))
            texts = []
            for fp in txt_files:
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                except Exception:
                    continue
            unis = build_unigram(
                texts,
                tokenizer_mode=args.mode,
                regex_pattern=args.pattern,
                sort_by=args.sort_by,
                top_k=args.top_k,
            )
            save_unigram_to_csv(unis, args.output_csv)
            print(f"Unigram saved to {args.output_csv}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == 'unigram-csv':
        try:
            unis = build_unigram_from_csv(
                csv_path=args.input_csv,
                content_column=args.content_column,
                tokenizer_mode=args.mode,
                regex_pattern=args.pattern,
                sort_by=args.sort_by,
                top_k=args.top_k,
            )
            save_unigram_to_csv(unis, args.output_csv)
            print(f"Unigram saved to {args.output_csv}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
