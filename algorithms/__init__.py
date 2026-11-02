"""
Algorithms package for ResumeMatch DAA Project.
Exports core implementations: LCS, Edit Distance, KMP, and Naive matching.
"""

from .lcs import lcs_full, lcs_length_optimized, lcs_similarity
from .edit_distance import (
    edit_distance_full,
    edit_distance_optimized,
    edit_distance_similarity,
)
from .kmp import compute_lps, compute_lps_with_trace, kmp_search, kmp_search_with_trace
from .naive_matching import naive_search, naive_search_with_trace
from .optimized_algorithms import lcs_single_row, edit_distance_single_row

__all__ = [
    "lcs_full",
    "lcs_length_optimized",
    "lcs_similarity",
    "edit_distance_full",
    "edit_distance_optimized",
    "edit_distance_similarity",
    "compute_lps",
    "compute_lps_with_trace",
    "kmp_search",
    "kmp_search_with_trace",
    "naive_search",
    "naive_search_with_trace",
    "lcs_single_row",
    "edit_distance_single_row",
]
