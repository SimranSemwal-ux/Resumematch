"""
Unit Tests for Knuth-Morris-Pratt (KMP) String Matching Algorithm
"""

import pytest
from algorithms.kmp import compute_lps, compute_lps_with_trace, kmp_search, kmp_search_with_trace
from algorithms.naive_matching import naive_search


class TestKMP:
    def test_lps_computation_known_patterns(self):
        # Textbook pattern 1: "ABCDABD"
        # Prefixes: A, AB, ABC, ABCD, ABCDA, ABCDAB, ABCDABD
        # Suffixes: D, BD, ABD, DABD, CDABD, BCDABD
        lps1 = compute_lps("ABCDABD")
        assert lps1 == [0, 0, 0, 0, 1, 2, 0]

        # Monotonically increasing pattern: "AAAA"
        lps2 = compute_lps("AAAA")
        assert lps2 == [0, 1, 2, 3]

        # Complex pattern: "AABAACAABAA"
        lps3 = compute_lps("AABAACAABAA")
        assert lps3 == [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]

        # Empty string
        assert compute_lps("") == []

        # Single character
        assert compute_lps("X") == [0]

    def test_lps_with_trace(self):
        lps, trace = compute_lps_with_trace("ABABC")
        assert lps == [0, 0, 1, 2, 0]
        assert len(trace) > 0
        assert "lps_state" in trace[0]

    def test_pattern_at_beginning(self):
        text = "PYTHON is an interpreted programming language"
        pattern = "PYTHON"
        matches, _ = kmp_search(text, pattern)
        assert matches == [0]

    def test_pattern_at_end(self):
        text = "we build distributed microservices using DOCKER"
        pattern = "DOCKER"
        matches, _ = kmp_search(text, pattern)
        assert matches == [len(text) - len(pattern)]

    def test_pattern_absent(self):
        text = "Frontend engineering with React and TypeScript"
        pattern = "KUBERNETES"
        matches, _ = kmp_search(text, pattern)
        assert matches == []

    def test_multiple_occurrences(self):
        text = "the quick brown fox jumps over the lazy dog in the park"
        pattern = "the"
        matches, _ = kmp_search(text, pattern)
        assert matches == [0, 31, 47]

    def test_overlapping_matches(self):
        # Overlapping occurrences: "AAA" in "AAAAA"
        # Indices: 0 ("AAA"..), 1 (. "AAA".), 2 (.."AAA")
        text = "AAAAA"
        pattern = "AAA"
        matches, _ = kmp_search(text, pattern)
        assert matches == [0, 1, 2]

        # Another overlap: "ABA" in "ABABA" -> [0, 2]
        matches_aba, _ = kmp_search("ABABA", "ABA")
        assert matches_aba == [0, 2]

    def test_empty_inputs(self):
        # Empty pattern
        matches, comps = kmp_search("Valid text", "")
        assert matches == []
        assert comps == 0

        # Empty text
        matches, comps = kmp_search("", "Pattern")
        assert matches == []
        assert comps == 0

        # Pattern longer than text
        matches, comps = kmp_search("short", "longer_pattern")
        assert matches == []
        assert comps == 0

    def test_kmp_matches_naive_baseline(self):
        """
        Verify that KMP and Naive search return identical match indices
        across diverse inputs.
        """
        test_cases = [
            ("bacbababaabcbab", "ab"),
            ("mississippi", "issi"),
            ("abcde fghij klmno pqrst uvwxyz", "klm"),
            ("110101110101101", "101"),
            ("resume screening with automated algorithms", "algo"),
        ]
        for text, pattern in test_cases:
            kmp_m, _ = kmp_search(text, pattern)
            naive_m, _ = naive_search(text, pattern)
            assert kmp_m == naive_m, f"Failed on text='{text}', pattern='{pattern}'"

    def test_kmp_with_trace(self):
        text = "ABABABC"
        pattern = "ABABC"
        matches, comps, trace = kmp_search_with_trace(text, pattern)
        assert matches == [2]
        assert len(trace) > 0
        assert trace[-1]["match_found"] is True
