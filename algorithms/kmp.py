"""
Knuth-Morris-Pratt (KMP) String Matching Algorithm Implementation
Academic DAA Module for ResumeMatch

Theoretical Background:
-----------------------
The Knuth-Morris-Pratt (KMP) algorithm is a linear-time string searching algorithm
that searches for occurrences of a pattern P of length m within a text T of length n.
Unlike the naive algorithm, which shifts the pattern by exactly 1 position on mismatch
and re-examines previously matched characters, KMP precomputes an auxiliary table:
the Longest Proper Prefix which is also a Suffix (LPS array).

LPS Array Definition:
---------------------
LPS[i] stores the length of the longest proper prefix of P[0..i] that is also a suffix of P[0..i].
A proper prefix is a prefix that is not the entire string itself.

Matching Invariant:
-------------------
When a mismatch occurs at P[j] after matching j characters (T[i-j .. i-1] == P[0 .. j-1]),
the next possible matching position without re-evaluating already matched text characters
is determined by resetting j to LPS[j-1]. The text pointer i is NEVER decremented.

Complexity:
-----------
- LPS Preprocessing Time: O(m)
- Text Search Time: O(n)
- Overall Time Complexity: O(n + m)
- Auxiliary Space: O(m) to store the LPS array.
"""

from typing import List, Tuple, Dict, Any


def compute_lps(pattern: str) -> List[int]:
    """
    Constructs the Longest Proper Prefix which is also a Suffix (LPS) array for pattern P.

    Args:
        pattern: The pattern string of length m.

    Returns:
        List[int]: The LPS array of size m, where lps[i] is the length of the longest
                   proper prefix of pattern[0..i] that is also a suffix of pattern[0..i].
    """
    m = len(pattern)
    if m == 0:
        return []

    lps = [0] * m
    length = 0  # Length of the previous longest prefix suffix
    i = 1

    # Loop computes lps[i] for i = 1 to m - 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                # Fall back to the previous known longest prefix suffix
                length = lps[length - 1]
                # Note: i is NOT incremented here
            else:
                lps[i] = 0
                i += 1

    return lps


def compute_lps_with_trace(pattern: str) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Computes LPS array while recording step-by-step trace for educational visualization
    in the Streamlit Algorithm Lab.

    Args:
        pattern: Pattern string.

    Returns:
        Tuple of (lps_array, trace_steps).
    """
    m = len(pattern)
    if m == 0:
        return [], []

    lps = [0] * m
    length = 0
    i = 1
    trace: List[Dict[str, Any]] = [
        {
            "step": 0,
            "i": 0,
            "char_i": pattern[0],
            "length": 0,
            "char_len": "",
            "action": "Base case: lps[0] = 0 (proper prefix cannot be full string)",
            "lps_state": [0] + [0] * (m - 1)
        }
    ]

    step_count = 1
    while i < m:
        curr_step = {
            "step": step_count,
            "i": i,
            "char_i": pattern[i],
            "length": length,
            "char_len": pattern[length] if length < m else "",
        }
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            curr_step["action"] = f"Match ('{pattern[i]}' == '{pattern[length-1]}'): increment length to {length}, lps[{i}] = {length}, advance i"
            i += 1
        else:
            if length != 0:
                old_len = length
                length = lps[length - 1]
                curr_step["action"] = f"Mismatch ('{pattern[i]}' != '{pattern[old_len]}'): backtrack length to lps[{old_len - 1}] = {length} (keep i={i})"
            else:
                lps[i] = 0
                curr_step["action"] = f"Mismatch with length=0: lps[{i}] = 0, advance i"
                i += 1

        curr_step["lps_state"] = lps[:]
        trace.append(curr_step)
        step_count += 1

    return lps, trace


def kmp_search(text: str, pattern: str) -> Tuple[List[int], int]:
    """
    Performs Knuth-Morris-Pratt string matching to find all occurrences of pattern in text.

    Features:
        - Detects all match positions (0-indexed).
        - Handles overlapping matches correctly.
        - Returns total character comparisons made for empirical comparison with Naive.
        - Gracefully handles empty pattern or empty text.

    Args:
        text: Target text string (length n).
        pattern: Search pattern string (length m).

    Returns:
        Tuple of:
            - matches (List[int]): 0-indexed starting positions of all occurrences.
            - comparisons (int): Total character comparisons performed.
    """
    n = len(text)
    m = len(pattern)

    # Base / edge cases
    if m == 0 or n == 0 or m > n:
        return [], 0

    lps = compute_lps(pattern)
    matches: List[int] = []
    comparisons = 0

    i = 0  # Index for text
    j = 0  # Index for pattern

    while i < n:
        comparisons += 1
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            # Complete match found! Record starting position
            match_start = i - j
            matches.append(match_start)
            # Use LPS to find the next possible start position (handles overlaps)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            # Mismatch after j matches
            if j != 0:
                # Do NOT increment i; skip redundant prefix comparisons
                j = lps[j - 1]
            else:
                i += 1

    return matches, comparisons


def kmp_search_with_trace(text: str, pattern: str, max_steps: int = 150) -> Tuple[List[int], int, List[Dict[str, Any]]]:
    """
    Performs KMP search while recording visual trace steps for the interactive Algorithm Lab.

    Args:
        text: Text to search within.
        pattern: Pattern to search for.
        max_steps: Maximum steps to record to prevent oversized UI payload.

    Returns:
        Tuple of (matches, comparisons, trace_steps).
    """
    n = len(text)
    m = len(pattern)

    if m == 0 or n == 0 or m > n:
        return [], 0, []

    lps = compute_lps(pattern)
    matches: List[int] = []
    trace: List[Dict[str, Any]] = []
    comparisons = 0

    i = 0
    j = 0
    step = 0

    while i < n and step < max_steps:
        step += 1
        comparisons += 1
        match_char = (pattern[j] == text[i])

        trace_item = {
            "step": step,
            "text_idx": i,
            "pattern_idx": j,
            "text_char": text[i],
            "pattern_char": pattern[j],
            "is_char_match": match_char,
        }

        if match_char:
            trace_item["action"] = f"Match '{text[i]}' == '{pattern[j]}'. Increment i and j."
            i += 1
            j += 1
        else:
            if j != 0:
                trace_item["action"] = f"Mismatch! Skip using lps[{j-1}] = {lps[j-1]}. Keep i={i}, new j={lps[j-1]}."
                j = lps[j - 1]
            else:
                trace_item["action"] = f"Mismatch with j=0. Advance text pointer i to {i+1}."
                i += 1

        if j == m:
            match_pos = i - j
            matches.append(match_pos)
            trace_item["match_found"] = True
            trace_item["match_pos"] = match_pos
            trace_item["action"] += f" FULL MATCH at index {match_pos}! Shift j to lps[{j-1}] = {lps[j-1]}."
            j = lps[j - 1]
        else:
            trace_item["match_found"] = False

        trace.append(trace_item)

    # Finish any remaining search without trace if max_steps reached
    while i < n:
        comparisons += 1
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return matches, comparisons, trace
