"""
Unit Tests for Preprocessing Module
"""

import pytest
from preprocessing.text_cleaner import clean_text
from preprocessing.tokenizer import tokenize, generate_ngrams
from preprocessing import load_skills_dictionary
from preprocessing.pdf_extractor import extract_text_from_pdf
from preprocessing.docx_extractor import extract_text_from_docx


class TestPreprocessing:
    def test_clean_text_technical_preservation(self):
        raw = "Expert in C++, C#, .NET, Node.js, and CI/CD pipelines."
        cleaned = clean_text(raw)

        assert "c++" in cleaned
        assert "c#" in cleaned
        assert ".net" in cleaned
        assert "node.js" in cleaned
        assert "ci/cd" in cleaned

    def test_clean_text_unicode_normalization(self):
        # Text with smart quotes, em-dash, and bullet points
        raw = "• Experienced Engineer — working with ‘Python’ & “Docker”"
        cleaned = clean_text(raw)

        # Bullets should be converted to space, quotes sanitized, em-dash normalized
        assert "•" not in cleaned
        assert "—" not in cleaned
        assert "experienced engineer" in cleaned
        assert "python" in cleaned
        assert "docker" in cleaned

    def test_clean_text_whitespace(self):
        raw = "   Lots    of \t  irregular  \n\n  spaces   here.  "
        cleaned = clean_text(raw)
        assert cleaned == "lots of irregular spaces here"

    def test_tokenizer_basic_and_tech_terms(self):
        text = "built microservices in c++ and node.js with docker"
        tokens = tokenize(text)

        assert "built" in tokens
        assert "c++" in tokens
        assert "node.js" in tokens
        assert "docker" in tokens

    def test_tokenizer_stopwords_removal(self):
        text = "the candidate has experience with python and in sql"
        tokens_all = tokenize(text, remove_stopwords=False)
        tokens_filtered = tokenize(text, remove_stopwords=True)

        assert "the" in tokens_all
        assert "the" not in tokens_filtered
        assert "python" in tokens_filtered
        assert "sql" in tokens_filtered

    def test_generate_ngrams(self):
        tokens = ["deep", "learning", "neural", "network"]
        bigrams = generate_ngrams(tokens, 2)
        assert bigrams == [
            "deep learning",
            "learning neural",
            "neural network"
        ]

        trigrams = generate_ngrams(tokens, 3)
        assert trigrams == [
            "deep learning neural",
            "learning neural network"
        ]

        assert generate_ngrams(tokens, 5) == []

    def test_skills_dictionary_structure(self):
        data = load_skills_dictionary()
        assert "categories" in data
        assert "aliases" in data
        assert "single_word_boundary_strict" in data

        categories = data["categories"]
        assert "Programming Languages" in categories
        assert "Frameworks & Libraries" in categories
        assert "Databases" in categories
        assert "Cloud & DevOps" in categories
        assert "AI & Data Science" in categories

        # Verify key skills present
        all_skills = []
        for cat_list in categories.values():
            all_skills.extend(cat_list)

        for expected in ["python", "c++", "docker", "machine learning", "sql"]:
            assert expected in all_skills

    def test_extractor_empty_validation(self):
        with pytest.raises(ValueError):
            extract_text_from_pdf(b"")

        with pytest.raises(ValueError):
            extract_text_from_docx(b"")

    def test_extract_sample_files(self):
        import os
        from preprocessing import extract_text_from_file

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resumes_dir = os.path.join(base_dir, "sample_data", "resumes")

        pdf_path = os.path.join(resumes_dir, "alex_morgan_software_engineer.pdf")
        docx_path = os.path.join(resumes_dir, "alex_morgan_software_engineer.docx")
        txt_path = os.path.join(resumes_dir, "alex_morgan_software_engineer.txt")

        if os.path.exists(pdf_path):
            text_pdf = extract_text_from_file(pdf_path)
            assert "Alex Morgan" in text_pdf
            assert "Python" in text_pdf

        if os.path.exists(docx_path):
            text_docx = extract_text_from_file(docx_path)
            assert "Alex Morgan" in text_docx
            assert "Python" in text_docx

        if os.path.exists(txt_path):
            text_txt = extract_text_from_file(txt_path)
            assert "Alex Morgan" in text_txt
            assert "Python" in text_txt
