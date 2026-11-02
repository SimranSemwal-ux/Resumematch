# Resume–Job Description Matching System (ResumeMatch)

> **Design and Analysis of Algorithms (DAA) Academic Capstone Project**  
> Developed by **[Simran Semwal](https://github.com/SimranSemwal-ux)**  
> Built strictly with custom, scratch-implemented algorithms: **Longest Common Subsequence (LCS)**, **Edit Distance (Levenshtein Distance)**, and **Knuth-Morris-Pratt (KMP) String Matching**.

---

## 📌 1. Project Overview

**ResumeMatch** is a transparent, explainable, and mathematically grounded resume screening platform. Unlike modern commercial Applicant Tracking Systems (ATS) that operate as opaque, proprietary machine learning black boxes, ResumeMatch evaluates candidate alignment strictly using **foundational computer science algorithms implemented from scratch in pure Python**.

The system allows users to upload resumes in **PDF**, **DOCX**, or **TXT** format alongside a target Job Description. It extracts technical competencies using strict boundary matching and alias resolution, calculates structural and lexical sequence alignments, renders 2D dynamic programming matrices, generates actionable ethical recommendations, and provides empirical performance benchmarking against theoretical complexities.

---

## 🎯 2. Problem Statement & Academic Objectives

### Problem Statement
Standard ATS platforms rely on third-party semantic embeddings or opaque LLM prompts that introduce unpredictable hallucinations, high operational costs, and zero auditability for candidates and recruiters. Furthermore, superficial vocabulary differences often cause qualified candidates to be filtered out without clear explanation.

### Key Objectives
1. **Zero External AI / LLM Dependency**: Implement all core string matching, subsequence discovery, and distance metrics manually in pure Python without using `difflib`, `fuzzywuzzy`, `rapidfuzz`, or external AI APIs.
2. **Transparent Mathematical Scoring**: Formulate a composite heuristic based on skill coverage (50%), token LCS alignment (30%), and token Levenshtein closeness (20%).
3. **Interactive DAA Laboratory**: Provide interactive visualizers for 2D DP grids, backtracking trajectories, and LPS table transitions for academic demonstration.
4. **Empirical Benchmarking**: Profile runtime and memory utilization using `time.perf_counter` and `tracemalloc`, validating empirical behavior against theoretical $O(m \cdot n)$ and $O(n + m)$ complexities.

---

## 🧬 3. Core DAA Algorithms & Mathematical Formulations

### A. Longest Common Subsequence (LCS)
Calculates the longest sequence of words/characters present in both documents in the same relative order without requiring contiguity.

#### Recurrence Relation:
Let $X = [x_1, \dots, x_m]$ and $Y = [y_1, \dots, y_n]$.
$$
dp[i][j] = \begin{cases}
0 & \text{if } i=0 \lor j=0 \\
1 + dp[i-1][j-1] & \text{if } X[i-1] = Y[j-1] \\
\max(dp[i-1][j], dp[i][j-1]) & \text{if } X[i-1] \neq Y[j-1]
\end{cases}
$$

#### Complexity Analysis:
- **Full DP Table**: Time $O(m \cdot n)$, Auxiliary Space $O(m \cdot n)$ (permits backtracking optimal subsequence).
- **Space-Optimized Length**: Time $O(m \cdot n)$, Auxiliary Space $O(\min(m, n))$ (utilizes two rolling rows).

#### Normalized LCS Similarity:
$$
\text{LCS Similarity} = \begin{cases}
100.0\% & \text{if } m = 0 \land n = 0 \\
0.0\% & \text{if } m = 0 \lor n = 0 \\
\frac{2 \cdot \text{LCS Length}}{m + n} \times 100\% & \text{otherwise}
\end{cases}
$$

---

### B. Edit Distance (Levenshtein Distance)
Determines the minimal number of character/token edit operations (**Insertion**, **Deletion**, **Substitution**) required to transform the source sequence into the target.

#### Recurrence Relation:
$$
dp[i][j] = \begin{cases}
i & \text{if } j=0 \\
j & \text{if } i=0 \\
dp[i-1][j-1] & \text{if } X[i-1] = Y[j-1] \\
1 + \min \begin{pmatrix} dp[i-1][j] & \text{(Deletion)} \\ dp[i][j-1] & \text{(Insertion)} \\ dp[i-1][j-1] & \text{(Substitution)} \end{pmatrix} & \text{if } X[i-1] \neq Y[j-1]
\end{cases}
$$

#### Complexity Analysis:
- **Full DP Table**: Time $O(m \cdot n)$, Auxiliary Space $O(m \cdot n)$ (supports reconstruction of optimal edit operations).
- **Space-Optimized Distance**: Time $O(m \cdot n)$, Auxiliary Space $O(\min(m, n))$.

#### Normalized Edit Distance Similarity:
$$
\text{Edit Similarity} = \begin{cases}
100.0\% & \text{if } m = 0 \land n = 0 \\
\left(1 - \frac{\text{Distance}}{\max(m, n)}\right) \times 100\% & \text{otherwise}
\end{cases}
$$

---

### C. Knuth-Morris-Pratt (KMP) String Matching
Searches for occurrences of job-required technical terms within the raw resume in linear time without backtracking in the input text.

#### Preprocessing (LPS Array):
Constructs the **Longest Proper Prefix which is also a Suffix (LPS)** array of size $m$.
- LPS Preprocessing: Time $O(m)$, Space $O(m)$.
- Text Search: Time $O(n)$.
- **Total Complexity**: Time $O(n + m)$, Space $O(m)$.

#### Invariant:
When a mismatch occurs after matching $j$ characters, the text pointer $i$ is never decremented. The next prospective match alignment begins by resetting $j = LPS[j-1]$.

---

### D. Naive String Matching Baseline
Serves as an empirical baseline against KMP.
- Compares pattern characters across all $n - m + 1$ window positions.
- **Worst-Case Time**: $O(n \cdot m)$ (e.g., $A \dots AB$ inside $A \dots B$).
- **Auxiliary Space**: $O(1)$.

---

## 📊 4. Complexity Comparison Matrix

| Algorithm | Best Time | Average Time | Worst Time | Auxiliary Space | Reconstructed Output |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LCS (Full DP)** | $O(m \cdot n)$ | $O(m \cdot n)$ | $O(m \cdot n)$ | $O(m \cdot n)$ | Optimal Common Subsequence |
| **LCS (Optimized)** | $O(m \cdot n)$ | $O(m \cdot n)$ | $O(m \cdot n)$ | $O(\min(m, n))$ | Length only |
| **Edit Distance (Full)** | $O(m \cdot n)$ | $O(m \cdot n)$ | $O(m \cdot n)$ | $O(m \cdot n)$ | Sequence of Edit Operations |
| **Edit Distance (Opt)** | $O(m \cdot n)$ | $O(m \cdot n)$ | $O(m \cdot n)$ | $O(\min(m, n))$ | Distance only |
| **KMP String Matching** | $O(n)$ | $O(n + m)$ | $O(n + m)$ | $O(m)$ | Exact match 0-indices |
| **Naive String Matching** | $O(n)$ | $O(n)$ | $O(n \cdot m)$ | $O(1)$ | Exact match 0-indices |

---

## 🧮 5. Transparent Scoring Model

$$
\text{Skill Coverage} = \left(\frac{|\text{Matched Required Skills}|}{|\text{Total Required Skills}|}\right) \times 100\%
$$

$$
\text{LCS Score} = \left(\frac{2 \cdot \text{LCS Length}}{\text{Resume Salient Tokens} + \text{JD Salient Tokens}}\right) \times 100\%
$$

$$
\text{Edit Distance Score} = \left(1 - \frac{\text{Token Edit Distance}}{\max(\text{Resume Tokens}, \text{JD Tokens})}\right) \times 100\%
$$

$$
\mathbf{\text{Final Composite Score}} = 0.50 \cdot (\text{Skill Coverage}) + 0.30 \cdot (\text{LCS Score}) + 0.20 \cdot (\text{Edit Distance Score})
$$

All individual components and the composite score are clamped to $[0.0, 100.0]$.

---

## 🏗️ 6. System Architecture & Directory Structure

```
ResumeMatch/
│
├── algorithms/
│   ├── __init__.py
│   ├── lcs.py                    # Full DP, backtracking, space-optimized & similarity
│   ├── edit_distance.py          # Full DP, op trace, space-optimized & similarity
│   ├── kmp.py                    # LPS computation, trace & linear search
│   ├── naive_matching.py         # O(n*m) baseline & comparison counter
│   └── optimized_algorithms.py   # Single-row cache-optimized DP implementations
│
├── preprocessing/
│   ├── __init__.py
│   ├── pdf_extractor.py          # PyMuPDF extractor with byte-stream support
│   ├── docx_extractor.py         # python-docx table and paragraph extractor
│   ├── text_cleaner.py           # Sanitizer preserving C++, C#, Node.js, CI/CD
│   ├── tokenizer.py              # Word, N-gram, and technical phrase tokenizer
│   └── skills_dictionary.json   # 150+ categorized skills, aliases & strict boundaries
│
├── matching/
│   ├── __init__.py
│   ├── skill_matcher.py          # Boundary-safe extraction & DAA fuzzy matching
│   ├── similarity.py             # Token-level sequence wrappers
│   ├── scoring.py                # 50/30/20 transparent weighted model & KMP audit
│   └── recommendations.py        # Rule-based ethical improvement engine
│
├── benchmarking/
│   ├── __init__.py
│   ├── performance.py            # tracemalloc memory and time.perf_counter profiler
│   ├── benchmark.py              # Automated empirical benchmark suite
│   ├── generate_graphs.py        # Interactive Plotly figures & HTML exporter
│   └── benchmark_results.csv     # Recorded empirical measurements
│
├── tests/
│   ├── test_lcs.py               # Unit tests for LCS
│   ├── test_edit_distance.py     # Unit tests for Levenshtein distance
│   ├── test_kmp.py               # Unit tests for KMP and LPS table
│   ├── test_matching.py          # Unit tests for skill extraction and scoring
│   └── test_preprocessing.py     # Unit tests for cleaners, extractors, dictionaries
│
├── sample_data/
│   ├── resumes/                  # Fictional resumes in PDF, DOCX, and TXT
│   │   ├── alex_morgan_software_engineer.{pdf,docx,txt}
│   │   ├── priya_sharma_data_scientist.{pdf,docx,txt}
│   │   └── jordan_lee_frontend_developer.{pdf,docx,txt}
│   ├── job_descriptions/         # Target job descriptions
│   │   ├── fullstack_python_jd.txt
│   │   ├── data_scientist_jd.txt
│   │   └── frontend_react_jd.txt
│   └── generate_samples.py       # Programmatic sample generator
│
├── assets/
│   └── graphs/                   # Exported interactive Plotly HTML figures
│
├── app.py                        # Streamlit multi-module SaaS dashboard
├── requirements.txt              # Pinned dependencies
├── README.md                     # Comprehensive academic documentation
├── .gitignore                    # Standard Python/IDE ignores
└── LICENSE                       # MIT Open Source License
```

---

## 💻 7. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/SimranSemwal-ux/ResumeMatch.git
cd ResumeMatch
```

### 2. Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 8. Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🧪 9. Running the Automated Test Suite

ResumeMatch includes a comprehensive suite of **41 automated tests** covering DP recurrences, backtracking accuracy, boundary safety, and document extractors:
```bash
pytest -v
```

Expected output:
```
tests/test_edit_distance.py::TestEditDistance::... PASSED
tests/test_kmp.py::TestKMP::... PASSED
tests/test_lcs.py::TestLCS::... PASSED
tests/test_matching.py::TestMatching::... PASSED
tests/test_preprocessing.py::TestPreprocessing::... PASSED

============================= 41 passed in 2.61s ==============================
```

---

## 📈 10. Running Empirical Benchmarks

Execute the automated empirical benchmark suite across synthetic token sequences and string lengths:
```bash
python -m benchmarking.benchmark
```

This runs repetitions of each algorithm, measures median runtimes using `time.perf_counter`, measures peak memory using `tracemalloc`, and exports the dataset to `benchmarking/benchmark_results.csv`.

To export interactive HTML charts to `assets/graphs/`:
```bash
python -m benchmarking.generate_graphs
```

### Key Empirical Findings:
1. **String Matching (Worst-Case Repetitive $A \dots AB$ in $A \dots B$)**:
   - At $N = 25,000$ characters: **Naive took 81.86 ms** vs. **KMP took 3.25 ms** (**>25x speedup**), confirming KMP's theoretical $O(n + m)$ linear bound.
2. **Space-Time Trade-off in LCS and Edit Distance**:
   - Full 2D Table at $800 \times 800$ tokens consumed **21.99 MB** of memory.
   - Space-Optimized 2-Row implementation consumed only **68.45 KB** of memory (**>320x memory footprint reduction**).

---

## ⚖️ 11. Ethical Guidelines & Limitations

- **Ethical Resume Optimization**: Suggestions generated by ResumeMatch are designed to format and showcase *genuine* competencies. Candidates are explicitly cautioned never to fabricate credentials or coursework.
- **Heuristic Nature**: The final matching score represents an algorithmic degree-of-overlap heuristic and should not be construed as an absolute predictor of professional hiring success.
- **OCR Limitation**: Scanned, non-searchable PDF documents without an OCR layer require pre-rasterization.

---

## 👤 12. Author & Project Metadata

- **Author**: [Simran Semwal](https://github.com/SimranSemwal-ux)
- **Academic Subject**: Design and Analysis of Algorithms (DAA) Laboratory
- **GitHub Profile**: [https://github.com/SimranSemwal-ux](https://github.com/SimranSemwal-ux)
- **License**: [MIT License](LICENSE)
