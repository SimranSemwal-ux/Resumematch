"""
Similarity Calculation Module for ResumeMatch
Computes sequence similarity between Resume and Job Description using our
manually implemented LCS and Edit Distance algorithms.
"""

from typing import List, Tuple, Dict, Any
from algorithms.lcs import lcs_length_optimized, lcs_similarity
from algorithms.edit_distance import edit_distance_optimized, edit_distance_similarity
from preprocessing.text_cleaner import clean_text
from preprocessing.tokenizer import tokenize


def compute_sequence_similarities(
    resume_text: str,
    jd_text: str,
    max_tokens: int = 1200
) -> Dict[str, Any]:
    """
    Computes both LCS and Edit Distance similarities on token sequences.

    To ensure responsive performance while preserving full context:
    - Filters out high-frequency non-technical stopwords.
    - Limits sequence length to `max_tokens` (resumes & JDs rarely exceed 1000 salient tokens).

    Args:
        resume_text: Cleaned or raw resume text.
        jd_text: Cleaned or raw job description text.
        max_tokens: Ceiling for token sequence length.

    Returns:
        Dict[str, Any]: Detailed similarity metrics including:
            - lcs_length: Absolute LCS token count.
            - lcs_score: Normalized LCS percentage [0.0, 100.0].
            - edit_distance: Absolute token edit distance.
            - edit_distance_score: Normalized Edit Distance percentage [0.0, 100.0].
            - resume_token_count: Number of tokens evaluated.
            - jd_token_count: Number of tokens evaluated.
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    # Tokenize with stop words removed to focus on technical and meaningful content
    resume_tokens = tokenize(cleaned_resume, remove_stopwords=True)[:max_tokens]
    jd_tokens = tokenize(cleaned_jd, remove_stopwords=True)[:max_tokens]

    r_len = len(resume_tokens)
    j_len = len(jd_tokens)

    if r_len == 0 or j_len == 0:
        return {
            "lcs_length": 0,
            "lcs_score": 0.0,
            "edit_distance": max(r_len, j_len),
            "edit_distance_score": 0.0,
            "resume_token_count": r_len,
            "jd_token_count": j_len,
            "resume_tokens": resume_tokens,
            "jd_tokens": jd_tokens,
        }

    # 1. LCS Token Similarity
    lcs_len = lcs_length_optimized(resume_tokens, jd_tokens)
    lcs_score = round((2.0 * lcs_len / (r_len + j_len)) * 100.0, 2)
    lcs_score = min(100.0, max(0.0, lcs_score))

    # 2. Edit Distance Token Similarity
    edit_dist = edit_distance_optimized(resume_tokens, jd_tokens)
    max_len = max(r_len, j_len)
    edit_score = round((1.0 - (edit_dist / max_len)) * 100.0, 2)
    edit_score = min(100.0, max(0.0, edit_score))

    return {
        "lcs_length": lcs_len,
        "lcs_score": lcs_score,
        "edit_distance": edit_dist,
        "edit_distance_score": edit_score,
        "resume_token_count": r_len,
        "jd_token_count": j_len,
        "resume_tokens": resume_tokens,
        "jd_tokens": jd_tokens,
    }
