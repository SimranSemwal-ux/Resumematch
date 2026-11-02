"""
Scoring Engine for ResumeMatch
Implements the transparent, explainable DAA weighted scoring model.

Mathematical Formula:
---------------------
Final Score = (0.50 * Skill Coverage) + (0.30 * LCS Score) + (0.20 * Edit Distance Score)

Component Definitions:
----------------------
1. Skill Coverage (Weight: 50%):
   Measures proportion of mandatory job description skills possessed by the candidate:
   Skill Coverage = (matched_required_skills / total_required_skills) * 100

2. LCS Token Similarity Score (Weight: 30%):
   Measures structural and contextual sequence alignment between resume and JD tokens:
   LCS Score = (2 * LCS_length / (resume_tokens + jd_tokens)) * 100

3. Edit Distance Token Similarity Score (Weight: 20%):
   Quantifies overall lexical transformation closeness:
   Edit Distance Score = (1 - edit_distance / max(resume_tokens, jd_tokens)) * 100

4. Explainable KMP Keyword Evidence:
   Knuth-Morris-Pratt (KMP) runs independently across the raw resume text to locate
   exact matches, occurrences, and character offsets for all required keywords,
   providing auditability without double-counting.
"""

from typing import Dict, Any, List
from matching.skill_matcher import analyze_skill_matching
from matching.similarity import compute_sequence_similarities
from algorithms.kmp import kmp_search
from preprocessing.text_cleaner import clean_text


def compute_matching_score(
    resume_text: str,
    jd_text: str,
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive matching analysis, component scores, and final transparent score.

    Args:
        resume_text: Raw or extracted resume text.
        jd_text: Raw or extracted job description text.
        weights: Optional dictionary of weights for components. Default:
                 {"skill_coverage": 0.50, "lcs": 0.30, "edit_distance": 0.20}

    Returns:
        Dict[str, Any]: Complete analysis dictionary containing:
            - final_score: Weighted composite score [0.0, 100.0].
            - components: Breakdown of scores, weights, and contributions.
            - skill_analysis: Detailed skill match results.
            - sequence_analysis: LCS and Edit Distance metrics.
            - kmp_evidence: Keyword search evidence with occurrences & positions.
            - status: 'success' | 'empty_input'.
    """
    if weights is None:
        weights = {
            "skill_coverage": 0.50,
            "lcs": 0.30,
            "edit_distance": 0.20
        }

    # Normalize weights to ensure sum == 1.0
    weight_sum = sum(weights.values())
    w_skill = weights.get("skill_coverage", 0.50) / weight_sum
    w_lcs = weights.get("lcs", 0.30) / weight_sum
    w_edit = weights.get("edit_distance", 0.20) / weight_sum

    if not resume_text or not resume_text.strip() or not jd_text or not jd_text.strip():
        return {
            "status": "empty_input",
            "final_score": 0.0,
            "components": {
                "skill_coverage": {"score": 0.0, "weight": w_skill, "contribution": 0.0},
                "lcs": {"score": 0.0, "weight": w_lcs, "contribution": 0.0},
                "edit_distance": {"score": 0.0, "weight": w_edit, "contribution": 0.0},
            },
            "skill_analysis": {
                "matched_skills": {},
                "missing_skills": {},
                "additional_skills": {},
                "skill_coverage_pct": 0.0,
                "matched_count": 0,
                "total_required_count": 0,
            },
            "sequence_analysis": {
                "lcs_length": 0,
                "lcs_score": 0.0,
                "edit_distance": 0,
                "edit_distance_score": 0.0,
            },
            "kmp_evidence": {},
            "error_message": "One or both input texts are empty."
        }

    # 1. Skill Extraction and Matching Analysis
    skill_analysis = analyze_skill_matching(resume_text, jd_text)
    skill_cov_score = float(skill_analysis["skill_coverage_pct"])

    # 2. Sequence Similarity Analysis (LCS and Levenshtein)
    seq_analysis = compute_sequence_similarities(resume_text, jd_text)
    lcs_score = float(seq_analysis["lcs_score"])
    edit_score = float(seq_analysis["edit_distance_score"])

    # 3. KMP Keyword Verification Evidence
    # Run KMP search over cleaned resume text for all required skills
    cleaned_resume = clean_text(resume_text)
    kmp_evidence: Dict[str, Dict[str, Any]] = {}

    for skill in skill_analysis["required_skills"]:
        # Search for skill pattern
        matches, comps = kmp_search(cleaned_resume, skill)
        kmp_evidence[skill] = {
            "occurrences": len(matches),
            "match_positions": matches[:10],  # cap to first 10 positions
            "comparisons": comps,
            "status": "Found in Resume" if len(matches) > 0 else "Missing in Resume"
        }

    # 4. Final Weighted Composite Score Calculation
    skill_contrib = round(skill_cov_score * w_skill, 2)
    lcs_contrib = round(lcs_score * w_lcs, 2)
    edit_contrib = round(edit_score * w_edit, 2)

    final_score = round(skill_contrib + lcs_contrib + edit_contrib, 2)
    final_score = min(100.0, max(0.0, final_score))

    return {
        "status": "success",
        "final_score": final_score,
        "components": {
            "skill_coverage": {
                "score": skill_cov_score,
                "weight": round(w_skill, 2),
                "contribution": skill_contrib
            },
            "lcs": {
                "score": lcs_score,
                "weight": round(w_lcs, 2),
                "contribution": lcs_contrib
            },
            "edit_distance": {
                "score": edit_score,
                "weight": round(w_edit, 2),
                "contribution": edit_contrib
            }
        },
        "skill_analysis": skill_analysis,
        "sequence_analysis": seq_analysis,
        "kmp_evidence": kmp_evidence
    }
