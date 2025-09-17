from newa_nlp import build_unigram_from_csv, save_unigram_to_csv

# Build unigram from CSV file
unigrams = build_unigram_from_csv(
    csv_path="corpus.csv",
    content_column="content",  # default is "content"
    tokenizer_mode="regex",    # or "space"
    regex_pattern=r"[\u0900-\u0963\u0965-\u097F]+",  # Devanagari pattern
    sort_by="dev",            # or "dev" for Devanagari alphabetical
)

# Save to CSV
save_unigram_to_csv(unigrams, "unigram_re.csv")