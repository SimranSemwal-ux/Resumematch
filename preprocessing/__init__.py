"""
Preprocessing package for ResumeMatch.
Provides extractors, cleaners, tokenizers, and skills dictionary loader.
"""

import json
import os
from typing import Dict, Any

from .pdf_extractor import extract_text_from_pdf
from .docx_extractor import extract_text_from_docx
from .text_cleaner import clean_text
from .tokenizer import tokenize, generate_ngrams


def load_skills_dictionary() -> Dict[str, Any]:
    """
    Loads the curated skills dictionary JSON file.

    Returns:
        Dict[str, Any]: Parsed skills dictionary containing categories, aliases,
                        and boundary strict rules.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "skills_dictionary.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_from_file(file_obj_or_path, filename: str = "") -> str:
    """
    Unified dispatcher to extract text from PDF, DOCX, or plain text sources.

    Args:
        file_obj_or_path: File path or file-like object.
        filename: Optional filename hint to determine file type.

    Returns:
        str: Extracted raw text.
    """
    name = ""
    if isinstance(file_obj_or_path, str):
        name = file_obj_or_path.lower()
    elif filename:
        name = filename.lower()
    elif hasattr(file_obj_or_path, "name"):
        name = getattr(file_obj_or_path, "name", "").lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_obj_or_path)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_obj_or_path)
    elif name.endswith(".txt"):
        if isinstance(file_obj_or_path, str):
            with open(file_obj_or_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        elif hasattr(file_obj_or_path, "read"):
            content = file_obj_or_path.read()
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="replace")
            return content
        elif isinstance(file_obj_or_path, bytes):
            return file_obj_or_path.decode("utf-8", errors="replace")
        raise ValueError("Invalid source for TXT file")
    else:
        raise ValueError(
            f"Unsupported file format '{name}'. Please provide a .pdf, .docx, or .txt file."
        )


__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_docx",
    "extract_text_from_file",
    "clean_text",
    "tokenize",
    "generate_ngrams",
    "load_skills_dictionary",
]
