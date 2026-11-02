"""
Text Cleaner Module for ResumeMatch
Preprocesses and normalizes raw text while preserving critical technical syntax.

Key Design Decisions:
---------------------
1. Technical Tokens Preservation:
   - Symbols like '++', '#', '.', '/', and '-' are crucial in technical contexts:
     e.g., 'C++', 'C#', '.NET', 'Node.js', 'Next.js', 'CI/CD', 'scikit-learn'.
   - Naive punctuation removal (e.g., string.punctuation) destroys these terms.
   - We replace protected technical terms with temporary tokens, clean punctuation,
     and restore them, or use a regex designed specifically for technical vocabularies.

2. Unicode Normalization:
   - Replaces fancy typographic characters (bullet points, en-dash, em-dash, smart quotes)
     with standard ASCII equivalents.

3. Whitespace Normalization:
   - Collapses multiple consecutive spaces, tabs, and line breaks into clean single spaces.
"""

import re
import unicodedata
from typing import Dict

# Mapping of protected technical phrases to safe temporary placeholders
PROTECTED_TERMS: Dict[str, str] = {
    "c++": "__TECH_CPP__",
    "c#": "__TECH_CSHARP__",
    ".net": "__TECH_DOTNET__",
    "node.js": "__TECH_NODEJS__",
    "next.js": "__TECH_NEXTJS__",
    "vue.js": "__TECH_VUEJS__",
    "react.js": "__TECH_REACTJS__",
    "express.js": "__TECH_EXPRESSJS__",
    "ci/cd": "__TECH_CICD__",
    "ci / cd": "__TECH_CICD__",
    "scikit-learn": "__TECH_SKLEARN__",
    "rest-api": "__TECH_RESTAPI__",
    "rest api": "__TECH_RESTAPI__",
    "power bi": "__TECH_POWERBI__",
    "spring boot": "__TECH_SPRINGBOOT__",
}


def clean_text(raw_text: str) -> str:
    """
    Cleans raw resume or job description text.

    Args:
        raw_text: Unprocessed input text.

    Returns:
        str: Lowercased, sanitized, and normalized text.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # Step 1: Normalize unicode characters (NFKD decomposes ligatures and accents)
    text = unicodedata.normalize("NFKD", raw_text)

    # Step 2: Convert to lowercase
    text = text.lower()

    # Step 3: Replace non-standard bullets and dashes
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\u00B7\u25AA\u25CF]", " ", text)  # Bullets
    text = re.sub(r"[\u2013\u2014\u2015]", "-", text)  # En-dash, Em-dash
    text = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", text)  # Smart single quotes
    text = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', text)  # Smart double quotes

    # Step 4: Protect critical technical terms
    # Sort terms by length descending to match multi-word phrases before single words
    for term, placeholder in sorted(PROTECTED_TERMS.items(), key=lambda x: -len(x[0])):
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(term) + r"(?![a-zA-Z0-9])"
        text = re.sub(pattern, f" {placeholder} ", text)

    # Step 5: Replace unwanted punctuation with space, keeping alphanumeric characters
    # Retain standard hyphens inside words, but strip isolated symbols
    text = re.sub(r"[^\w\s\-]", " ", text)

    # Step 6: Restore protected technical terms
    for term, placeholder in PROTECTED_TERMS.items():
        text = text.replace(placeholder, term)
        text = text.replace(placeholder.lower(), term)

    # Step 7: Normalize whitespace (collapse multiple spaces, tabs, newlines)
    text = re.sub(r"\s+", " ", text).strip()

    return text
