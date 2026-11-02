"""
Optimized DAA Algorithms and Space-Time Trade-off Implementations
Academic DAA Module for ResumeMatch

Theoretical Analysis of Space Optimization:
------------------------------------------
In classical 2D Dynamic Programming (e.g., LCS and Levenshtein Distance), computing
cell dp[i][j] only requires access to values from the current row (i) and the
immediately preceding row (i-1).

1. Space Reduction:
   - Full 2D Table: Stores all (m+1) x (n+1) entries -> Space Complexity = O(m * n).
   - Rolling Row Optimization: Only stores two rows of size (min(m, n) + 1) -> Space Complexity = O(min(m, n)).
   - Single Row with Variable: Further reduces storage to 1 array of size min(m, n) + 1 and 1 integer variable for the diagonal.

2. Does Space Optimization reduce Time Complexity?
   - NO. The theoretical time complexity remains O(m * n) because every subproblem (i, j)
     must still be evaluated at least once.
   - HOWEVER, empirical runtime is frequently 2x-4x faster due to:
     a) CPU Cache Locality: An array of size O(min(m, n)) fits entirely in L1/L2 CPU cache,
        eliminating costly RAM cache misses.
     b) Garbage Collection & Allocation Overhead: Eliminates allocating millions of inner
        list pointers in Python.

3. When to use which:
   - Full DP Table: MANDATORY when reconstructing the alignment, subsequence, or edit trace.
   - Space-Optimized: PREFERRED for production scoring, benchmark comparisons, and large input checks.
"""

from typing import Union, Sequence, Any, List


def lcs_single_row(
    seq1: Union[str, Sequence[Any]],
    seq2: Union[str, Sequence[Any]]
) -> int:
    """
    Computes LCS length using a single 1D array of size min(m, n) + 1.
    Uses a temporary variable `prev_diag` to preserve dp[i-1][j-1].

    Complexity:
        - Time: O(m * n)
        - Auxiliary Space: O(min(m, n)) - exactly 1 array.
    """
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    m, n = len(seq1), len(seq2)
    if n == 0:
        return 0

    dp: List[int] = [0] * (n + 1)

    for i in range(1, m + 1):
        prev_diag = 0  # Represents dp[i-1][j-1]
        for j in range(1, n + 1):
            temp = dp[j]  # dp[i-1][j] before it gets overwritten
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = 1 + prev_diag
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev_diag = temp

    return dp[n]


def edit_distance_single_row(
    seq1: Union[str, Sequence[Any]],
    seq2: Union[str, Sequence[Any]]
) -> int:
    """
    Computes Levenshtein edit distance using a single 1D array of size min(m, n) + 1.
    Maintains `prev_diag` for the diagonal substitution cost dp[i-1][j-1].

    Complexity:
        - Time: O(m * n)
        - Auxiliary Space: O(min(m, n)) - exactly 1 array.
    """
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    m, n = len(seq1), len(seq2)
    if n == 0:
        return m

    dp: List[int] = list(range(n + 1))

    for i in range(1, m + 1):
        prev_diag = i - 1  # Base case for column 0
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]  # dp[i-1][j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev_diag
            else:
                dp[j] = 1 + min(
                    dp[j],        # Deletion: dp[i-1][j]
                    dp[j - 1],    # Insertion: dp[i][j-1]
                    prev_diag     # Substitution: dp[i-1][j-1]
                )
            prev_diag = temp

    return dp[n]
