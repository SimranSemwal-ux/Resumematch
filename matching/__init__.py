"""
Matching package for ResumeMatch.
Exports skill extractor, matcher, scoring engine, and recommendation engine.
"""

from .skill_matcher import extract_skills_from_text, analyze_skill_matching
from .similarity import compute_sequence_similarities
from .scoring import compute_matching_score
from .recommendations import generate_recommendations, analyze_resume_structure

__all__ = [
    "extract_skills_from_text",
    "analyze_skill_matching",
    "compute_sequence_similarities",
    "compute_matching_score",
    "generate_recommendations",
    "analyze_resume_structure",
]
