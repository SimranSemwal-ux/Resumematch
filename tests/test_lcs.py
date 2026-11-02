"""
Unit Tests for Longest Common Subsequence (LCS) Implementation
"""

import pytest
from algorithms.lcs import lcs_full, lcs_length_optimized, lcs_similarity
from algorithms.optimized_algorithms import lcs_single_row


class TestLCS:
    def test_identical_strings(self):
        s1 = "ALGORITHM"
        s2 = "ALGORITHM"
        length, dp, seq = lcs_full(s1, s2)
        assert length == 9
        assert seq == "ALGORITHM"
        assert lcs_length_optimized(s1, s2) == 9
        assert lcs_single_row(s1, s2) == 9
        assert lcs_similarity(s1, s2) == 100.0

    def test_completely_different_strings(self):
        s1 = "ABC"
        s2 = "XYZ"
        length, dp, seq = lcs_full(s1, s2)
        assert length == 0
        assert seq == ""
        assert lcs_length_optimized(s1, s2) == 0
        assert lcs_single_row(s1, s2) == 0
        assert lcs_similarity(s1, s2) == 0.0

    def test_empty_strings(self):
        # Both empty
        length, dp, seq = lcs_full("", "")
        assert length == 0
        assert seq == ""
        assert lcs_similarity("", "") == 100.0

        # One empty
        length, dp, seq = lcs_full("PYTHON", "")
        assert length == 0
        assert seq == ""
        assert lcs_similarity("PYTHON", "") == 0.0

        length, dp, seq = lcs_full("", "JAVA")
        assert length == 0
        assert seq == ""
        assert lcs_similarity("", "JAVA") == 0.0

    def test_repeated_characters(self):
        s1 = "AAAA"
        s2 = "AA"
        length, dp, seq = lcs_full(s1, s2)
        assert length == 2
        assert seq == "AA"
        assert lcs_length_optimized(s1, s2) == 2
        assert lcs_single_row(s1, s2) == 2

    def test_known_classical_lcs(self):
        # Classical textbook example: "ABCBDAB" and "BDCABA"
        # One valid LCS is "BCBA" or "BDAB", length is 4
        s1 = "ABCBDAB"
        s2 = "BDCABA"
        length, dp, seq = lcs_full(s1, s2)
        assert length == 4
        assert len(seq) == 4
        # Verify reconstructed subsequence is indeed a subsequence of both
        i = 0
        for ch in s1:
            if i < len(seq) and ch == seq[i]:
                i += 1
        assert i == 4

        j = 0
        for ch in s2:
            if j < len(seq) and ch == seq[j]:
                j += 1
        assert j == 4

    def test_token_sequences(self):
        seq1 = ["python", "docker", "kubernetes", "aws", "react"]
        seq2 = ["c++", "docker", "aws", "react", "graphql"]
        length, dp, reconstructed = lcs_full(seq1, seq2)
        assert length == 3
        assert reconstructed == ["docker", "aws", "react"]
        assert lcs_length_optimized(seq1, seq2) == 3
        assert lcs_single_row(seq1, seq2) == 3

    def test_space_optimized_equivalence(self):
        pairs = [
            ("DYNAMIC", "PROGRAMMING"),
            ("SUBSEQUENCE", "SUBSTRING"),
            ("DATASTRUCTURES", "ALGORITHMS"),
            ("SOFTWARE", "ENGINEER"),
        ]
        for s1, s2 in pairs:
            full_len, _, _ = lcs_full(s1, s2)
            opt_len = lcs_length_optimized(s1, s2)
            single_len = lcs_single_row(s1, s2)
            assert full_len == opt_len == single_len
