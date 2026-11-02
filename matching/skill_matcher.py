"""
Skill Matcher Module for ResumeMatch
Extracts technical skills using exact boundary matching, alias resolution,
and controlled fuzzy matching using our DAA Levenshtein Edit Distance.

Design Considerations for False Positive Prevention:
----------------------------------------------------
1. Strict Word Boundaries:
   - Short terms like 'c', 'r', 'go', 'git', 'sql', 'js', 'ml' MUST use strict word
     boundaries (?<![a-zA-Z0-9]) and (?![a-zA-Z0-9]).
   - This prevents 'c' from matching in 'cloud', 'r' in 'program', 'art' in 'smart',
     or 'go' in 'good'.

2. Controlled Fuzzy Matching:
   - Uses our own algorithms.edit_distance_optimized implementation.
   - ONLY applies to candidate tokens of length >= 5 characters.
   - Maximum allowed edit distance is 1 (or 2 for words >= 8 characters) with distance/length <= 0.2.
   - Never applies fuzzy matching to short acronyms (e.g., 'aws', 'dsa', 'sql').
"""

import re
from typing import Set, Dict, List, Any, Tuple
from algorithms.edit_distance import edit_distance_optimized
from preprocessing import load_skills_dictionary, clean_text, tokenize


def build_canonical_skill_map(skills_dict: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str], Set[str]]:
    """
    Constructs lookup maps for canonical skills, aliases, and categories.

    Returns:
        Tuple of:
            - canonical_skills (Dict[str, str]): skill -> category.
            - alias_map (Dict[str, str]): alias -> canonical_skill.
            - strict_boundary_skills (Set[str]): skills requiring strict single-word boundaries.
    """
    canonical_skills: Dict[str, str] = {}
    for category, skill_list in skills_dict.get("categories", {}).items():
        for skill in skill_list:
            canonical_skills[skill.lower()] = category

    alias_map: Dict[str, str] = {
        alias.lower(): target.lower()
        for alias, target in skills_dict.get("aliases", {}).items()
    }

    strict_boundary_skills = set(
        s.lower() for s in skills_dict.get("single_word_boundary_strict", [])
    )

    return canonical_skills, alias_map, strict_boundary_skills


def extract_skills_from_text(
    raw_text: str,
    enable_fuzzy: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts all recognized technical skills from text using boundary matching,
    alias normalization, and controlled DAA fuzzy matching.

    Args:
        raw_text: Input text (resume or job description).
        enable_fuzzy: If True, uses controlled Levenshtein matching for minor typos.

    Returns:
        Dict[str, Dict[str, Any]]: Map of canonical_skill -> {
            'category': str,
            'match_type': 'exact' | 'alias' | 'fuzzy',
            'matched_term': str
        }
    """
    if not raw_text or not isinstance(raw_text, str):
        return {}

    skills_dict = load_skills_dictionary()
    canonical_skills, alias_map, strict_boundary_skills = build_canonical_skill_map(skills_dict)
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned, remove_stopwords=False)
    token_set = set(tokens)

    extracted: Dict[str, Dict[str, Any]] = {}

    # Helper to check boundary match in cleaned text
    def has_boundary_match(phrase: str) -> bool:
        # For technical phrases with punctuation like c++, c#, .net, node.js
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(phrase) + r"(?![a-zA-Z0-9])"
        return bool(re.search(pattern, cleaned, re.IGNORECASE))

    # Phase 1: Match multi-word canonical skills first (e.g. "machine learning", "rest api")
    multi_word_skills = [s for s in canonical_skills if " " in s]
    for skill in multi_word_skills:
        if has_boundary_match(skill):
            extracted[skill] = {
                "category": canonical_skills[skill],
                "match_type": "exact",
                "matched_term": skill
            }

    # Phase 2: Match multi-word aliases (e.g. "restful api" -> "rest api")
    multi_word_aliases = [a for a in alias_map if " " in a]
    for alias in multi_word_aliases:
        canonical = alias_map[alias]
        if canonical not in extracted and has_boundary_match(alias):
            extracted[canonical] = {
                "category": canonical_skills.get(canonical, "Technical Skills"),
                "match_type": "alias",
                "matched_term": alias
            }

    # Phase 3: Match single-word canonical skills
    single_word_skills = [s for s in canonical_skills if " " not in s]
    for skill in single_word_skills:
        if skill in extracted:
            continue
        # Use boundary search for safety
        if has_boundary_match(skill):
            extracted[skill] = {
                "category": canonical_skills[skill],
                "match_type": "exact",
                "matched_term": skill
            }

    # Phase 4: Match single-word aliases (e.g. "js" -> "javascript", "ml" -> "machine learning")
    single_word_aliases = [a for a in alias_map if " " not in a]
    for alias in single_word_aliases:
        canonical = alias_map[alias]
        if canonical in extracted:
            continue
        if has_boundary_match(alias):
            extracted[canonical] = {
                "category": canonical_skills.get(canonical, "Technical Skills"),
                "match_type": "alias",
                "matched_term": alias
            }

    # Phase 5: Controlled Fuzzy Matching using DAA Levenshtein Distance
    # Only runs for unmatched single-word canonical skills of length >= 5
    if enable_fuzzy:
        for skill in single_word_skills:
            if skill in extracted or len(skill) < 5 or skill in strict_boundary_skills:
                continue

            max_allowed_dist = 2 if len(skill) >= 8 else 1

            for token in token_set:
                if len(token) < 4:
                    continue
                # Length difference optimization check
                if abs(len(token) - len(skill)) > max_allowed_dist:
                    continue

                dist = edit_distance_optimized(token, skill)
                if dist <= max_allowed_dist and (dist / len(skill)) <= 0.25:
                    extracted[skill] = {
                        "category": canonical_skills[skill],
                        "match_type": "fuzzy",
                        "matched_term": f"{token} (typo for {skill})"
                    }
                    break

    return extracted


def analyze_skill_matching(
    resume_text: str,
    jd_text: str
) -> Dict[str, Any]:
    """
    Performs comprehensive comparative skill analysis between Resume and Job Description.

    Args:
        resume_text: Resume document text.
        jd_text: Job description text.

    Returns:
        Dict[str, Any] containing:
            - required_skills: Dict of skills required by JD.
            - candidate_skills: Dict of skills found in Resume.
            - matched_skills: Dict of skills present in both.
            - missing_skills: Dict of skills required by JD but missing in Resume.
            - additional_skills: Dict of candidate skills not requested by JD.
            - skill_coverage_pct: Percentage of required skills met [0.0, 100.0].
    """
    required = extract_skills_from_text(jd_text, enable_fuzzy=False)
    candidate = extract_skills_from_text(resume_text, enable_fuzzy=True)

    required_keys = set(required.keys())
    candidate_keys = set(candidate.keys())

    matched_keys = required_keys.intersection(candidate_keys)
    missing_keys = required_keys.difference(candidate_keys)
    additional_keys = candidate_keys.difference(required_keys)

    total_required = len(required_keys)
    matched_count = len(matched_keys)

    if total_required == 0:
        # If JD has no technical skills identified, coverage is neutrally 100% or 0%
        # If candidate has some skills, 100.0, else 100.0
        coverage = 100.0
    else:
        coverage = round((matched_count / total_required) * 100.0, 2)

    return {
        "required_skills": {k: required[k] for k in sorted(required_keys)},
        "candidate_skills": {k: candidate[k] for k in sorted(candidate_keys)},
        "matched_skills": {k: candidate[k] for k in sorted(matched_keys)},
        "missing_skills": {k: required[k] for k in sorted(missing_keys)},
        "additional_skills": {k: candidate[k] for k in sorted(additional_keys)},
        "total_required_count": total_required,
        "matched_count": matched_count,
        "missing_count": len(missing_keys),
        "additional_count": len(additional_keys),
        "skill_coverage_pct": coverage
    }
