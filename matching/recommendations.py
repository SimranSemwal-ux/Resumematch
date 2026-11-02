"""
Recommendation Engine for ResumeMatch
Generates actionable, ethical, and explainable recommendations based on
algorithmic gap analysis between the resume and the target job description.

Ethical Guidelines:
-------------------
All recommendations adhere strictly to truthful representation. Candidates are
explicitly cautioned NEVER to fabricate experience, coursework, or credentials.
Suggestions focus exclusively on formatting, highlighting existing verified
proficiencies, and identifying areas for genuine self-study or upskilling.
"""

import re
from typing import Dict, List, Any


# Standard headings commonly expected in professional resumes
EXPECTED_SECTIONS: Dict[str, List[str]] = {
    "Work Experience": ["experience", "employment", "work history", "professional experience"],
    "Projects": ["projects", "personal projects", "academic projects", "key projects"],
    "Education": ["education", "academic background", "qualifications", "academics"],
    "Technical Skills": ["skills", "technical skills", "technologies", "proficiencies", "core competencies"],
    "Certifications": ["certifications", "certificates", "credentials", "courses", "licenses"]
}


def analyze_resume_structure(resume_text: str) -> Dict[str, bool]:
    """
    Checks for the presence of standard professional resume sections.

    Args:
        resume_text: Raw resume text.

    Returns:
        Dict[str, bool]: Section name -> True if detected, False otherwise.
    """
    lower = resume_text.lower()
    sections_found: Dict[str, bool] = {}

    for section_name, keywords in EXPECTED_SECTIONS.items():
        found = False
        for kw in keywords:
            # Look for section header pattern (standalone line or clear boundary)
            pattern = r"(?:^|\n)\s*" + re.escape(kw) + r"\s*(?::|\n|\r|$)"
            if re.search(pattern, lower, re.MULTILINE):
                found = True
                break
        sections_found[section_name] = found

    return sections_found


def generate_recommendations(
    scoring_result: Dict[str, Any],
    resume_text: str
) -> Dict[str, Any]:
    """
    Generates structured, rule-based recommendations based on skill gaps,
    sequence alignment, and resume structure.

    Args:
        scoring_result: Output dictionary from compute_matching_score.
        resume_text: Full raw resume text.

    Returns:
        Dict[str, Any] containing:
            - high_priority: Immediate actionable recommendations.
            - missing_skills_by_category: Grouped missing skills.
            - structural_suggestions: Section-level resume improvement suggestions.
            - keyword_alignment_tips: Suggestions regarding keyword alignment.
            - ethical_notice: Explicit disclaimer on academic honesty.
    """
    skill_analysis = scoring_result.get("skill_analysis", {})
    missing_skills = skill_analysis.get("missing_skills", {})
    matched_skills = skill_analysis.get("matched_skills", {})
    coverage_pct = skill_analysis.get("skill_coverage_pct", 0.0)

    seq_analysis = scoring_result.get("sequence_analysis", {})
    lcs_score = seq_analysis.get("lcs_score", 0.0)

    structure_audit = analyze_resume_structure(resume_text)

    high_priority: List[str] = []
    structural_suggestions: List[str] = []
    keyword_tips: List[str] = []

    # 1. Structural Audit Recommendations
    for section, detected in structure_audit.items():
        if not detected:
            if section in ["Projects", "Work Experience"]:
                structural_suggestions.append(
                    f"Consider adding a dedicated **{section}** section. Demonstrating practical application of skills through concrete projects significantly improves recruiter comprehension."
                )
            elif section == "Certifications":
                structural_suggestions.append(
                    f"A **Certifications** section was not explicitly detected. If you hold industry-recognized credentials, creating this section provides strong third-party verification."
                )
            else:
                structural_suggestions.append(
                    f"Ensure your **{section}** section has a clear, standardized heading so that automated parsers can identify your qualifications without ambiguity."
                )

    # 2. Missing Skills Categorization
    missing_by_cat: Dict[str, List[str]] = {}
    for skill_name, info in missing_skills.items():
        cat = info.get("category", "General Technical Skills")
        missing_by_cat.setdefault(cat, []).append(skill_name.upper() if len(skill_name) <= 3 else skill_name.title())

    # 3. Priority Recommendations based on Skill Coverage
    if coverage_pct < 40.0:
        high_priority.append(
            f"**Critical Skill Coverage Gap ({coverage_pct}% match)**: The resume lacks more than half of the explicitly stated technical requirements. Review the job description carefully to assess if this position aligns with your current technical profile."
        )
    elif coverage_pct < 70.0:
        high_priority.append(
            f"**Moderate Skill Coverage ({coverage_pct}% match)**: You meet several core requirements, but key competencies are missing. If you have practical experience with the missing skills, make sure they are prominently featured."
        )
    else:
        high_priority.append(
            f"**Strong Skill Coverage ({coverage_pct}% match)**: Your profile aligns well with the stated core technical requirements. Fine-tuning terminology and project impact metrics will maximize ATS resonance."
        )

    # Category-specific advice
    if "Programming Languages" in missing_by_cat:
        langs = ", ".join(missing_by_cat["Programming Languages"])
        high_priority.append(
            f"**Missing Core Languages ({langs})**: If you have worked with any of these languages in academic labs, personal projects, or previous roles, explicitly document them along with the version or framework used."
        )

    if "Databases" in missing_by_cat:
        dbs = ", ".join(missing_by_cat["Databases"])
        high_priority.append(
            f"**Database Architecture Gap ({dbs})**: If you have built relational schemas or utilized NoSQL stores, describe the specific queries, indexing, or ORM integration you implemented."
        )

    if "Cloud & DevOps" in missing_by_cat:
        cloud = ", ".join(missing_by_cat["Cloud & DevOps"])
        keyword_tips.append(
            f"The job description emphasizes DevOps & Cloud tooling ({cloud}). Mentioning containerization (Docker) or deployment workflows strengthens your engineering profile."
        )

    # 4. Sequence & Phrasing (LCS) Feedback
    if lcs_score < 15.0:
        keyword_tips.append(
            "**Terminology Divergence (Low LCS Alignment)**: Your resume uses distinct phraseology compared to the job description. Without misrepresenting your experience, consider adopting the standard industry terminology used in the posting (e.g., using 'REST API integration' instead of 'backend web service connectors')."
        )
    else:
        keyword_tips.append(
            "**Good Vocabulary Alignment**: Your document shares substantial common phraseology and technical idioms with the target job description."
        )

    ethical_notice = (
        "**Ethical Resume Optimization Notice**: All recommendations generated by ResumeMatch are designed "
        "to help you communicate your authentic background clearly. Never fabricate experience, claim technologies "
        "you have not practiced, or falsify credentials. Authenticity and demonstrable depth during technical interviews "
        "remain the ultimate benchmark for engineering success."
    )

    return {
        "high_priority": high_priority,
        "missing_skills_by_category": missing_by_cat,
        "structural_suggestions": structural_suggestions,
        "keyword_alignment_tips": keyword_tips,
        "ethical_notice": ethical_notice
    }
