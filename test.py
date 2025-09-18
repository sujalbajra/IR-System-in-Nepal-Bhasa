from newa_nlp import build_unigram_from_csv, save_unigram_to_csv

# Build unigram from CSV file
unigrams = build_unigram_from_csv(
    csv_path="corpus.csv",
    content_column="content",  # default is "content"
    tokenizer_mode="regex",    # or "space"
    # regex_pattern=r"[\u0900-\u0963\u0966-\u097F]+",  all digits 
    regex_pattern = r"[\u0900-\u0963\u0971-\u097F]+",  # no digits
  # Devanagari pattern
    sort_by="freq",            # or "dev" for Devanagari alphabetical
)

unigrams_dev = build_unigram_from_csv(
    csv_path="corpus.csv",
    content_column="content",  # default is "content"
    tokenizer_mode="regex",    # or "space"
    # regex_pattern=r"[\u0900-\u0963\u0966-\u097F]+",  all digits 
    regex_pattern = r"[\u0900-\u0963\u0971-\u097F]+",  # no digits
  # Devanagari pattern
    sort_by="dev",            # or "dev" for Devanagari alphabetical
)


# Save to CSV
save_unigram_to_csv(unigrams, "unigram_nodigits.csv")
save_unigram_to_csv(unigrams_dev, "unigram_nodigits_sorted.csv")