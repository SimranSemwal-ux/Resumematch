"""
Longest Common Subsequence (LCS) Algorithm Implementation
Academic DAA Module for ResumeMatch

Theoretical Background:
-----------------------
The Longest Common Subsequence problem determines the longest subsequence common
to all sequences in a set of sequences (often two). A subsequence is a sequence
derived from another sequence by deleting some or no elements without changing the
relative order of the remaining elements.

Recurrence Relation:
--------------------
Let X = [x_1, x_2, ..., x_m] and Y = [y_1, y_2, ..., y_n].
Let dp[i][j] represent the length of the LCS of prefixes X[0..i-1] and Y[0..j-1].

Base Cases:
    dp[0][j] = 0   for all 0 <= j <= n  (empty prefix of X)
    dp[i][0] = 0   for all 0 <= i <= m  (empty prefix of Y)

Recursive Step:
    dp[i][j] = 1 + dp[i-1][j-1]                if X[i-1] == Y[j-1]
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])      if X[i-1] != Y[j-1]

Complexity:
-----------
Full DP Table:
    - Time Complexity: O(m * n) - each cell of the (m+1) x (n+1) grid is computed in O(1).
    - Auxiliary Space Complexity: O(m * n) - to store the full table and reconstruct the sequence.
Space-Optimized Length:
    - Time Complexity: O(m * n) - requires evaluating all transitions.
    - Auxiliary Space Complexity: O(min(m, n)) - only previous and current row needed.
"""

from typing import List, Tuple, Union, Sequence, Any


def lcs_full(
    seq1: Union[str, Sequence[Any]],
    seq2: Union[str, Sequence[Any]]
) -> Tuple[int, List[List[int]], Union[str, List[Any]]]:
    """
    Computes the LCS using a full 2D DP table and reconstructs the common subsequence.

    Args:
        seq1: First sequence (string or list of tokens).
        seq2: Second sequence (string or list of tokens).

    Returns:
        A tuple of:
            - lcs_length (int): Length of the longest common subsequence.
            - dp_table (List[List[int]]): Complete (m+1) x (n+1) DP grid.
            - reconstructed_lcs (str or List[Any]): The actual reconstructed LCS.
    """
    m = len(seq1)
    n = len(seq2)

    # Step 1: Construct the (m+1) x (n+1) DP table initialized with zeros
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    # Step 2: Fill the DP table iteratively using bottom-up dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    # Step 3: Backtrack through the DP table to reconstruct the actual subsequence
    reconstructed: List[Any] = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            reconstructed.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            # Came from top cell (deletion/skip from seq1)
            i -= 1
        else:
            # Came from left cell (insertion/skip from seq2)
            j -= 1

    # Elements were collected in reverse order during backtracking
    reconstructed.reverse()

    # Preserve input representation: return string if input is string
    if isinstance(seq1, str) and isinstance(seq2, str):
        return lcs_length, dp, "".join(reconstructed)
    return lcs_length, dp, reconstructed


def lcs_length_optimized(
    seq1: Union[str, Sequence[Any]],
    seq2: Union[str, Sequence[Any]]
) -> int:
    """
    Computes the length of the LCS using space-optimized Dynamic Programming.
    Reduces space complexity from O(m * n) to O(min(m, n)) by maintaining only
    two rows of the DP table.

    Args:
        seq1: First sequence.
        seq2: Second sequence.

    Returns:
        int: Length of the longest common subsequence.
    """
    # Ensure seq2 is the shorter sequence to minimize auxiliary space
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    m, n = len(seq1), len(seq2)
    if n == 0:
        return 0

    # Allocate only two rows of size (n + 1)
    prev_row: List[int] = [0] * (n + 1)
    curr_row: List[int] = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr_row[j] = 1 + prev_row[j - 1]
            else:
                curr_row[j] = max(prev_row[j], curr_row[j - 1])
        # Move current row to previous for next iteration
        prev_row = curr_row[:]

    return prev_row[n]


def lcs_similarity(
    seq1: Union[str, Sequence[Any]],
    seq2: Union[str, Sequence[Any]]
) -> float:
    """
    Calculates the normalized LCS similarity percentage between two sequences:
        Similarity = (2 * LCS_length / (len(seq1) + len(seq2))) * 100

    Edge Cases:
        - If both sequences are empty: returns 100.0 (identical empty documents).
        - If one sequence is empty and the other is not: returns 0.0.

    Args:
        seq1: First sequence (string or token sequence).
        seq2: Second sequence (string or token sequence).

    Returns:
        float: Similarity percentage in range [0.0, 100.0].
    """
    m, n = len(seq1), len(seq2)
    if m == 0 and n == 0:
        return 100.0
    if m == 0 or n == 0:
        return 0.0

    length = lcs_length_optimized(seq1, seq2)
    similarity = (2.0 * length / (m + n)) * 100.0
    return min(100.0, max(0.0, round(similarity, 2)))
