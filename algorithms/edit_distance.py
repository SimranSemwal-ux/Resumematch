"""
Edit Distance (Levenshtein Distance) Algorithm Implementation
Academic DAA Module for ResumeMatch

Theoretical Background:
-----------------------
The Levenshtein Edit Distance quantifies how dissimilar two strings (or sequences)
are by counting the minimum number of single-character operations required to
transform string X into string Y.

Permitted Operations (Unit Cost = 1):
    1. Insertion: Add a character to X.
    2. Deletion: Remove a character from X.
    3. Substitution: Replace a character in X with a character from Y.
    (Match: No operation required, cost = 0).

Recurrence Relation:
--------------------
Let X = [x_1, x_2, ..., x_m] and Y = [y_1, y_2, ..., y_n].
Let dp[i][j] represent the minimum edit distance between prefixes X[0..i-1] and Y[0..j-1].

Base Cases:
    dp[i][0] = i   for all 0 <= i <= m  (i deletions required to convert X[0..i-1] to empty string)
    dp[0][j] = j   for all 0 <= j <= n  (j insertions required to convert empty string to Y[0..j-1])

Recursive Step:
    If X[i-1] == Y[j-1]:
        dp[i][j] = dp[i-1][j-1]
    Else:
        dp[i][j] = 1 + min(
            dp[i-1][j],    # Deletion of X[i-1]
            dp[i][j-1],    # Insertion of Y[j-1]
            dp[i-1][j-1]   # Substitution of X[i-1] with Y[j-1]
        )

Complexity:
-----------
Full DP Table:
    - Time Complexity: O(m * n) - grid of size (m+1) x (n+1) with O(1) transitions.
    - Auxiliary Space Complexity: O(m * n) - required for DP table and backtracking operations.
Space-Optimized Distance:
    - Time Complexity: O(m * n).
    - Auxiliary Space Complexity: O(min(m, n)) - using two rows (previous and current).
"""

from typing import List, Tuple, Union, Sequence, Any, Dict


def edit_distance_full(
    seq1: Union[str, Sequence[Any]],
    seq2: Union[str, Sequence[Any]]
) -> Tuple[int, List[List[int]], List[Dict[str, Any]]]:
    """
    Computes Levenshtein edit distance using a full 2D DP table and reconstructs
    the optimal sequence of edit operations.

    Args:
        seq1: Source sequence (string or token sequence).
        seq2: Target sequence (string or token sequence).

    Returns:
        A tuple of:
            - min_distance (int): Minimum edit distance.
            - dp_table (List[List[int]]): Complete (m+1) x (n+1) DP grid.
            - operations (List[Dict[str, Any]]): List of step-by-step edit operations.
              Each operation is a dict with keys: 'operation', 'char_src', 'char_tgt', 'cost'.
    """
    m = len(seq1)
    n = len(seq2)

    # Step 1: Initialize (m+1) x (n+1) DP table
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    # Step 2: Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Step 3: Populate DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion from seq1
                    dp[i][j - 1],      # Insertion from seq2
                    dp[i - 1][j - 1]   # Substitution
                )

    min_distance = dp[m][n]

    # Step 4: Backtrack to reconstruct the optimal sequence of edit operations
    operations: List[Dict[str, Any]] = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i - 1] == seq2[j - 1]:
            # Exact match: no operation needed
            operations.append({
                "operation": "match",
                "char_src": seq1[i - 1],
                "char_tgt": seq2[j - 1],
                "cost": 0,
                "description": f"Keep '{seq1[i - 1]}'"
            })
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            operations.append({
                "operation": "substitute",
                "char_src": seq1[i - 1],
                "char_tgt": seq2[j - 1],
                "cost": 1,
                "description": f"Substitute '{seq1[i - 1]}' with '{seq2[j - 1]}'"
            })
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion
            operations.append({
                "operation": "delete",
                "char_src": seq1[i - 1],
                "char_tgt": None,
                "cost": 1,
                "description": f"Delete '{seq1[i - 1]}'"
            })
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # Insertion
            operations.append({
                "operation": "insert",
                "char_src": None,
                "char_tgt": seq2[j - 1],
                "cost": 1,
                "description": f"Insert '{seq2[j - 1]}'"
            })
            j -= 1
        else:
            # Safety break to prevent infinite loop
            break

    # Reversing operations gives step-by-step transformation from seq1 to seq2
    operations.reverse()

    return min_distance, dp, operations


def edit_distance_optimized(
    seq1: Union[str, Sequence[Any]],
    seq2: Union[str, Sequence[Any]]
) -> int:
    """
    Computes Levenshtein edit distance using space-optimized DP.
    Auxiliary space is reduced from O(m * n) to O(min(m, n)).

    Args:
        seq1: Source sequence.
        seq2: Target sequence.

    Returns:
        int: Minimum edit distance.
    """
    # Ensure seq2 is the shorter sequence to minimize space
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    m, n = len(seq1), len(seq2)
    if n == 0:
        return m

    # Allocate two rows of size (n + 1)
    prev_row: List[int] = list(range(n + 1))
    curr_row: List[int] = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(
                    prev_row[j],      # Deletion
                    curr_row[j - 1],  # Insertion
                    prev_row[j - 1]   # Substitution
                )
        prev_row = curr_row[:]

    return prev_row[n]


def edit_distance_similarity(
    seq1: Union[str, Sequence[Any]],
    seq2: Union[str, Sequence[Any]]
) -> float:
    """
    Calculates normalized Edit Distance similarity percentage:
        Similarity = (1.0 - (distance / max(len(seq1), len(seq2)))) * 100.0

    Edge Cases:
        - Both sequences empty: returns 100.0.
        - One sequence empty and one non-empty: returns 0.0.

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

    max_len = max(m, n)
    distance = edit_distance_optimized(seq1, seq2)
    similarity = (1.0 - (distance / max_len)) * 100.0
    return min(100.0, max(0.0, round(similarity, 2)))
