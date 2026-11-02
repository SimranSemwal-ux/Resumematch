"""
DOCX Text Extractor Module for ResumeMatch
Extracts text from DOCX documents (paragraphs and tables) using python-docx.
"""

from typing import Union, BinaryIO
import io
import os


def extract_text_from_docx(source: Union[str, bytes, BinaryIO]) -> str:
    """
    Extracts text content from a DOCX document, including paragraphs and tables.

    Args:
        source: File path (str), raw bytes (bytes), or file-like stream (BytesIO).

    Returns:
        str: Extracted text content.

    Raises:
        ValueError: If file is empty, corrupted, or unsupported.
        FileNotFoundError: If a file path is provided but does not exist.
    """
    try:
        import docx
    except ImportError as e:
        raise ImportError(
            "python-docx is required for DOCX extraction. Install with 'pip install python-docx'."
        ) from e

    try:
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"DOCX file not found at path: {source}")
            if os.path.getsize(source) == 0:
                raise ValueError("The provided DOCX file is empty (0 bytes).")
            doc = docx.Document(source)
        elif isinstance(source, bytes):
            if len(source) == 0:
                raise ValueError("Provided DOCX byte buffer is empty.")
            doc = docx.Document(io.BytesIO(source))
        elif hasattr(source, "read"):
            content = source.read()
            if len(content) == 0:
                raise ValueError("Uploaded DOCX stream is empty.")
            doc = docx.Document(io.BytesIO(content))
        else:
            raise TypeError(f"Unsupported source type for DOCX extraction: {type(source)}")

        extracted_lines = []

        # 1. Extract from standard paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                extracted_lines.append(text)

        # 2. Extract from tables (resumes commonly format skills and experience inside tables)
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    # De-duplicate cell text in case of merged horizontal cells
                    seen = set()
                    unique_cells = []
                    for c in row_cells:
                        if c not in seen:
                            seen.add(c)
                            unique_cells.append(c)
                    extracted_lines.append(" | ".join(unique_cells))

        full_text = "\n\n".join(extracted_lines).strip()
        if not full_text:
            raise ValueError("No extractable text found in DOCX file.")

        return full_text

    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError, TypeError)):
            raise
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}") from e
