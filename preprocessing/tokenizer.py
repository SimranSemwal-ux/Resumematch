"""
Tokenizer Module for ResumeMatch
Splits cleaned text into tokens, preserves multi-word skills, and generates n-grams.
"""

import re
from typing import List, Set

# Standard English stop words to filter out during general token similarity,
# intentionally omitting words that double as technical terms (like 'c', 'r', 'it', 'go')
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """
    Tokenizes text into individual words and preserved technical tokens.

    Args:
        text: Preprocessed text string.
        remove_stopwords: If True, filters out standard non-technical stopwords.

    Returns:
        List[str]: List of token strings.
    """
    if not text:
        return []

    # Match alphanumeric words, including tech terms with ++, #, ., -
    token_pattern = r"[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+|\+\+|#|\-[a-zA-Z0-9]+|\/[a-zA-Z0-9]+)?"
    tokens = re.findall(token_pattern, text)

    if remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in STOP_WORDS]

    return tokens


def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    """
    Generates contiguous n-grams from a list of tokens.

    Args:
        tokens: Token list.
        n: Window size (e.g. 2 for bigrams, 3 for trigrams).

    Returns:
        List[str]: Space-separated n-gram strings.
    """
    if not tokens or n <= 0 or len(tokens) < n:
        return []

    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
