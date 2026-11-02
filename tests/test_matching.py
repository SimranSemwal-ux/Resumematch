"""
Unit Tests for Skill Matching and Transparent Scoring Engine
"""

import pytest
from matching.skill_matcher import extract_skills_from_text, analyze_skill_matching
from matching.scoring import compute_matching_score


class TestMatching:
    def test_perfect_skill_match(self):
        jd = "We need an engineer experienced with Python, Docker, and PostgreSQL."
        resume = "Proficient in Python programming, containerization with Docker, and relational databases like PostgreSQL."
        analysis = analyze_skill_matching(resume, jd)

        assert analysis["total_required_count"] == 3
        assert analysis["matched_count"] == 3
        assert analysis["missing_count"] == 0
        assert analysis["skill_coverage_pct"] == 100.0
        assert "python" in analysis["matched_skills"]
        assert "docker" in analysis["matched_skills"]
        assert "postgresql" in analysis["matched_skills"]

    def test_partial_skill_match(self):
        jd = "Required: Python, React, AWS, Docker."
        resume = "Hands-on experience developing web applications with Python and React."
        analysis = analyze_skill_matching(resume, jd)

        assert analysis["total_required_count"] == 4
        assert analysis["matched_count"] == 2
        assert analysis["missing_count"] == 2
        assert analysis["skill_coverage_pct"] == 50.0
        assert "python" in analysis["matched_skills"]
        assert "react" in analysis["matched_skills"]
        assert "aws" in analysis["missing_skills"]
        assert "docker" in analysis["missing_skills"]

    def test_zero_skill_match(self):
        jd = "Seeking C++, Rust, and Embedded Linux specialists."
        resume = "Marketing specialist with expertise in SEO, Social Media, and Copywriting."
        analysis = analyze_skill_matching(resume, jd)

        assert analysis["matched_count"] == 0
        assert analysis["skill_coverage_pct"] == 0.0

    def test_case_insensitivity(self):
        jd = "Requirements: pYtHoN, dOcKeR, kUbErNeTeS"
        resume = "Skilled in PYTHON, DOCKER, and KUBERNETES."
        analysis = analyze_skill_matching(resume, jd)
        assert analysis["skill_coverage_pct"] == 100.0

    def test_alias_resolution(self):
        jd = "Must know JavaScript, Machine Learning, and Kubernetes."
        # Candidate uses aliases: JS, ML, and K8s
        resume = "Experienced with JS frontend, ML pipelines, and K8s orchestration."
        analysis = analyze_skill_matching(resume, jd)

        assert "javascript" in analysis["matched_skills"]
        assert "machine learning" in analysis["matched_skills"]
        assert "kubernetes" in analysis["matched_skills"]
        assert analysis["skill_coverage_pct"] == 100.0

    def test_false_positive_boundary_protection(self):
        """
        Verify that sub-word substrings are NOT falsely identified as skills:
        - 'smart' should NOT match 'art'
        - 'cloud' should NOT match language 'c'
        - 'good' should NOT match 'go'
        - 'program' should NOT match 'r'
        """
        text = "A smart engineer designing good cloud software using modern programming methods."
        extracted = extract_skills_from_text(text, enable_fuzzy=False)

        assert "c" not in extracted, "False positive: 'c' matched in word 'cloud'"
        assert "go" not in extracted, "False positive: 'go' matched in word 'good'"
        assert "r" not in extracted, "False positive: 'r' matched in word 'program'"

    def test_controlled_fuzzy_matching(self):
        """
        Verify that minor typos in long technical keywords are matched via Levenshtein
        while avoiding false positives.
        """
        # "kubernets" (typo for "kubernetes")
        # "postgre" (typo for "postgresql")
        resume = "Deploying microservices to kubernets clusters with postgre database backend."
        extracted = extract_skills_from_text(resume, enable_fuzzy=True)

        assert "kubernetes" in extracted
        assert extracted["kubernetes"]["match_type"] == "fuzzy"

    def test_scoring_formula(self):
        jd = "Looking for a Python and SQL developer."
        resume = "Experienced Python and SQL developer with strong problem-solving skills."

        score_res = compute_matching_score(resume, jd)
        assert score_res["status"] == "success"
        assert 0.0 <= score_res["final_score"] <= 100.0

        comps = score_res["components"]
        assert comps["skill_coverage"]["weight"] == 0.50
        assert comps["lcs"]["weight"] == 0.30
        assert comps["edit_distance"]["weight"] == 0.20

        # Verify sum of contributions equals final_score within rounding tolerance
        calculated_sum = (
            comps["skill_coverage"]["contribution"]
            + comps["lcs"]["contribution"]
            + comps["edit_distance"]["contribution"]
        )
        assert abs(score_res["final_score"] - calculated_sum) <= 0.05

    def test_empty_inputs_scoring(self):
        res1 = compute_matching_score("", "Valid JD")
        assert res1["status"] == "empty_input"
        assert res1["final_score"] == 0.0

        res2 = compute_matching_score("Valid Resume", "")
        assert res2["status"] == "empty_input"
        assert res2["final_score"] == 0.0
