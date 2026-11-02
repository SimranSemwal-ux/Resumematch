"""
Naive (Brute-Force) String Matching Algorithm Implementation
Academic DAA Baseline Module for ResumeMatch

Theoretical Background:
-----------------------
The Naive string matching algorithm slides the pattern P of length m over the
text T of length n, checking one character at a time at every possible alignment
index from 0 to n - m.

Upon encountering a mismatch at P[j], the algorithm shifts the pattern forward
by only 1 position, discarding any knowledge gained about matching characters,
and starts comparing again from P[0].

Complexity:
-----------
- Best-Case Time: O(n) (e.g., first character mismatches at every window)
- Worst-Case Time: O(n * m) (e.g., Text = "AAAAAA...A", Pattern = "AAAB")
- Auxiliary Space: O(1) excluding returned matches list.

Purpose in ResumeMatch:
-----------------------
Serves as an empirical and theoretical baseline against which the linear-time
Knuth-Morris-Pratt (KMP) algorithm is evaluated in the Algorithm Lab and
Benchmarking Suite.
"""

from typing import List, Tuple, Dict, Any


def naive_search(text: str, pattern: str) -> Tuple[List[int], int]:
    """
    Finds all occurrences of pattern in text using brute-force comparison.

    Args:
        text: Target text string (length n).
        pattern: Pattern string (length m).

    Returns:
        Tuple of:
            - matches (List[int]): 0-indexed starting positions of all matches.
            - comparisons (int): Total character comparisons performed.
    """
    n = len(text)
    m = len(pattern)

    if m == 0 or n == 0 or m > n:
        return [], 0

    matches: List[int] = []
    comparisons = 0

    for i in range(n - m + 1):
        j = 0
        while j < m:
            comparisons += 1
            if text[i + j] != pattern[j]:
                break
            j += 1

        if j == m:
            matches.append(i)

    return matches, comparisons


def naive_search_with_trace(
    text: str,
    pattern: str,
    max_steps: int = 150
) -> Tuple[List[int], int, List[Dict[str, Any]]]:
    """
    Executes Naive search while recording visual trace steps for comparative visualization
    against KMP in the Algorithm Lab.

    Args:
        text: Target text string.
        pattern: Pattern string.
        max_steps: Step ceiling to prevent UI freezing on long texts.

    Returns:
        Tuple of (matches, comparisons, trace_steps).
    """
    n = len(text)
    m = len(pattern)

    if m == 0 or n == 0 or m > n:
        return [], 0, []

    matches: List[int] = []
    trace: List[Dict[str, Any]] = []
    comparisons = 0
    step = 0

    for i in range(n - m + 1):
        if step >= max_steps:
            break
        for j in range(m):
            step += 1
            comparisons += 1
            match_char = (text[i + j] == pattern[j])

            trace.append({
                "step": step,
                "window_start": i,
                "text_idx": i + j,
                "pattern_idx": j,
                "text_char": text[i + j],
                "pattern_char": pattern[j],
                "is_char_match": match_char,
                "action": f"Compare text[{i+j}]='{text[i+j]}' with pattern[{j}]='{pattern[j]}'"
            })

            if not match_char:
                break

        if j == m - 1 and text[i + m - 1] == pattern[m - 1]:
            matches.append(i)
            trace[-1]["match_found"] = True
            trace[-1]["match_pos"] = i

    # Finish any remaining comparisons if max_steps reached
    if step >= max_steps:
        for i_rem in range(i + 1, n - m + 1):
            j = 0
            while j < m:
                comparisons += 1
                if text[i_rem + j] != pattern[j]:
                    break
                j += 1
            if j == m:
                matches.append(i_rem)

    return matches, comparisons, trace
