"""
Unit Tests for Edit Distance (Levenshtein Distance) Implementation
"""

import pytest
from algorithms.edit_distance import (
    edit_distance_full,
    edit_distance_optimized,
    edit_distance_similarity,
)
from algorithms.optimized_algorithms import edit_distance_single_row


class TestEditDistance:
    def test_identical_strings(self):
        s1 = "DEVELOPER"
        s2 = "DEVELOPER"
        dist, dp, ops = edit_distance_full(s1, s2)
        assert dist == 0
        assert edit_distance_optimized(s1, s2) == 0
        assert edit_distance_single_row(s1, s2) == 0
        assert edit_distance_similarity(s1, s2) == 100.0
        # All operations should be matches
        assert all(op["operation"] == "match" for op in ops)

    def test_empty_strings(self):
        # Both empty
        assert edit_distance_optimized("", "") == 0
        assert edit_distance_similarity("", "") == 100.0

        # One empty
        assert edit_distance_optimized("PYTHON", "") == 6
        assert edit_distance_optimized("", "PYTHON") == 6
        assert edit_distance_similarity("PYTHON", "") == 0.0

    def test_single_operations(self):
        # 1 insertion
        dist, _, ops = edit_distance_full("cat", "cats")
        assert dist == 1
        assert any(op["operation"] == "insert" for op in ops)

        # 1 deletion
        dist, _, ops = edit_distance_full("cats", "cat")
        assert dist == 1
        assert any(op["operation"] == "delete" for op in ops)

        # 1 substitution
        dist, _, ops = edit_distance_full("cat", "bat")
        assert dist == 1
        assert any(op["operation"] == "substitute" for op in ops)

    def test_known_classical_examples(self):
        # Classical textbook example: "kitten" -> "sitting" = 3
        # (k->s, e->i, +g)
        dist, dp, ops = edit_distance_full("kitten", "sitting")
        assert dist == 3
        assert edit_distance_optimized("kitten", "sitting") == 3
        assert edit_distance_single_row("kitten", "sitting") == 3

        # Classical: "saturday" -> "sunday" = 3
        assert edit_distance_optimized("saturday", "sunday") == 3

    def test_reconstructed_operations_transformation(self):
        """
        Verify that applying the reconstructed sequence of edit operations to s1
        accurately transforms it into s2.
        """
        s1 = "algorithm"
        s2 = "logarithm"
        dist, dp, ops = edit_distance_full(s1, s2)

        # Reconstruct transformed string using the ops
        transformed = []
        for op in ops:
            if op["operation"] == "match":
                transformed.append(op["char_src"])
            elif op["operation"] == "substitute":
                transformed.append(op["char_tgt"])
            elif op["operation"] == "insert":
                transformed.append(op["char_tgt"])
            elif op["operation"] == "delete":
                pass  # character omitted

        assert "".join(transformed) == s2

    def test_space_optimized_equivalence(self):
        pairs = [
            ("javascript", "typescript"),
            ("database", "dataset"),
            ("engineering", "engineer"),
            ("machinelearning", "deeplearning")
        ]
        for s1, s2 in pairs:
            full_dist, _, _ = edit_distance_full(s1, s2)
            opt_dist = edit_distance_optimized(s1, s2)
            single_dist = edit_distance_single_row(s1, s2)
            assert full_dist == opt_dist == single_dist
