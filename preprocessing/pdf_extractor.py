"""
PDF Text Extractor Module for ResumeMatch
Extracts text from PDF documents using PyMuPDF (fitz) with robust error handling.
"""

from typing import Union, BinaryIO
import io
import os


def extract_text_from_pdf(source: Union[str, bytes, BinaryIO]) -> str:
    """
    Extracts text content from a PDF document.

    Args:
        source: File path (str), raw bytes (bytes), or file-like stream (BytesIO).

    Returns:
        str: Extracted text content across all pages.

    Raises:
        ValueError: If file is empty, corrupted, or unsupported.
        FileNotFoundError: If a file path is provided but does not exist.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is required for PDF extraction. Please install it with 'pip install pymupdf'."
            ) from e

    doc = None
    try:
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"PDF file not found at path: {source}")
            if os.path.getsize(source) == 0:
                raise ValueError("The provided PDF file is empty (0 bytes).")
            doc = fitz.open(source)
        elif isinstance(source, bytes):
            if len(source) == 0:
                raise ValueError("Provided PDF byte buffer is empty.")
            doc = fitz.open(stream=source, filetype="pdf")
        elif hasattr(source, "read"):
            # File-like object (e.g., Streamlit UploadedFile)
            content = source.read()
            if len(content) == 0:
                raise ValueError("Uploaded PDF stream is empty.")
            doc = fitz.open(stream=content, filetype="pdf")
        else:
            raise TypeError(f"Unsupported source type for PDF extraction: {type(source)}")

        if doc.page_count == 0:
            raise ValueError("The PDF document contains no pages.")

        extracted_pages = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text:
                extracted_pages.append(page_text.strip())

        full_text = "\n\n".join(extracted_pages).strip()
        if not full_text:
            raise ValueError(
                "No extractable text found in PDF. The document may be scanned or image-based."
            )

        return full_text

    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError, TypeError)):
            raise
        raise ValueError(f"Failed to extract text from PDF: {str(e)}") from e
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
