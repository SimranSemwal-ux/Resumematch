"""
ResumeMatch - Resume–Job Description Matching System
Advanced Design and Analysis of Algorithms (DAA) Academic Project
Developed by Simran Semwal (https://github.com/SimranSemwal-ux)

Core DAA Algorithms Implemented from Scratch:
1. Longest Common Subsequence (LCS) - Full 2D DP Table, Backtracking & Space-Optimized
2. Edit Distance (Levenshtein Distance) - Full DP Table, Op Reconstitution & Space-Optimized
3. Knuth-Morris-Pratt (KMP) - LPS Precomputation & O(N+M) Pattern Search
4. Naive Pattern Matching - O(N*M) Comparison Baseline
"""

import os
import io
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import core modules
from algorithms.lcs import lcs_full, lcs_length_optimized, lcs_similarity
from algorithms.edit_distance import (
    edit_distance_full,
    edit_distance_optimized,
    edit_distance_similarity,
)
from algorithms.kmp import (
    compute_lps,
    compute_lps_with_trace,
    kmp_search,
    kmp_search_with_trace,
)
from algorithms.naive_matching import naive_search, naive_search_with_trace
from algorithms.optimized_algorithms import lcs_single_row, edit_distance_single_row
from preprocessing import (
    clean_text,
    tokenize,
    extract_text_from_file,
    load_skills_dictionary,
)
from matching.scoring import compute_matching_score
from matching.recommendations import generate_recommendations
from benchmarking.generate_graphs import (
    get_benchmark_dataframe,
    plot_string_matching_comparison,
    plot_dp_runtime_comparison,
    plot_dp_memory_comparison,
)
from benchmarking.benchmark import run_all_benchmarks
from benchmarking.performance import measure_execution_and_memory

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeMatch | DAA Algorithm System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS (DARK SAAS THEME)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container styling */
    .stApp {
        background-color: #0E1117;
        color: #E6EDF3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Card design */
    .metric-card {
        background: linear-gradient(135deg, #161B22 0%, #1F242C 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58A6FF;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
    }
    .badge-success {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3FB950;
        border: 1px solid rgba(46, 160, 67, 0.4);
    }
    .badge-danger {
        background-color: rgba(248, 81, 73, 0.15);
        color: #F85149;
        border: 1px solid rgba(248, 81, 73, 0.4);
    }
    .badge-info {
        background-color: rgba(88, 166, 255, 0.15);
        color: #58A6FF;
        border: 1px solid rgba(88, 166, 255, 0.4);
    }
    .badge-purple {
        background-color: rgba(163, 113, 247, 0.15);
        color: #BC8CFF;
        border: 1px solid rgba(163, 113, 247, 0.4);
    }
    
    /* Code and pre tags */
    code, pre {
        background-color: #161B22 !important;
        color: #79C0FF !important;
        border: 1px solid #30363D !important;
        border-radius: 6px;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #58A6FF;
        margin-bottom: 8px;
    }
    
    /* Formula callout */
    .formula-box {
        background-color: #161B22;
        border-left: 4px solid #A371F7;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 12px 0;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HELPER DATA LOADERS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RESUME_DIR = os.path.join(BASE_DIR, "sample_data", "resumes")
SAMPLE_JD_DIR = os.path.join(BASE_DIR, "sample_data", "job_descriptions")


@st.cache_data
def load_sample_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **ResumeMatch**")
    st.caption("Algorithm-Driven Resume Screening | DAA Project")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home / Overview",
            "📄 Resume Analysis",
            "🔬 Algorithm Lab",
            "📊 Performance Analysis",
            "📚 About & Methodology",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Core Algorithms**")
    st.markdown("- 🧬 **LCS**: $O(m \\cdot n)$ DP")
    st.markdown("- 📏 **Levenshtein**: $O(m \\cdot n)$ DP")
    st.markdown("- 🔍 **KMP**: $O(n + m)$ Search")
    st.markdown("- ⏱️ **Naive**: $O(n \\cdot m)$ Baseline")

    st.markdown("---")
    st.caption("Crafted by [Simran Semwal](https://github.com/SimranSemwal-ux)")

# =============================================================================
# PAGE 1: HOME / OVERVIEW
# =============================================================================
if page == "🏠 Home / Overview":
    st.title("⚡ Resume–Job Description Matching System")
    st.subheader("Industry-Grade Academic DAA Project | Transparent, Explainable & Algorithmic")

    st.markdown(
        """
        **ResumeMatch** is an automated document screening platform engineered entirely from foundational
        computer science principles. Rather than relying on opaque third-party neural APIs or black-box embeddings,
        ResumeMatch uses **manually implemented Dynamic Programming and String Matching algorithms** to deliver
        explainable, mathematically verifiable scoring.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#3FB950; margin:0 0 10px 0;">🧬 Longest Common Subsequence</h4>
                <p style="font-size:0.9rem; color:#8B949E;">Computes structural token alignment between resume and JD sequences.</p>
                <b>Time:</b> <code>O(m · n)</code><br>
                <b>Space:</b> <code>O(m · n)</code> full | <code>O(min(m, n))</code> opt
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#58A6FF; margin:0 0 10px 0;">📏 Edit Distance (Levenshtein)</h4>
                <p style="font-size:0.9rem; color:#8B949E;">Measures minimum insertions, deletions, and substitutions to quantify divergence.</p>
                <b>Time:</b> <code>O(m · n)</code><br>
                <b>Space:</b> <code>O(m · n)</code> full | <code>O(min(m, n))</code> opt
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color:#BC8CFF; margin:0 0 10px 0;">🔍 Knuth-Morris-Pratt (KMP)</h4>
                <p style="font-size:0.9rem; color:#8B949E;">Locates required technical competencies in linear time using LPS prefix table.</p>
                <b>Time:</b> <code>O(n + m)</code><br>
                <b>Space:</b> <code>O(m)</code> auxiliary LPS table
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🛠️ System Architecture Pipeline")
    st.markdown(
        """
        ```
        [ Resume PDF/DOCX/TXT ] ────> [ PyMuPDF / python-docx Extractor ]
                                                  │
                                                  ▼
        [ Job Description Text ] ───> [ Technical Text Cleaner (Preserves C++, Node.js) ]
                                                  │
                                                  ▼
                                      [ N-Gram Tokenizer & Vocabulary ]
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
        [ Skill Extraction ]            [ Token LCS Alignment ]           [ Edit Distance Engine ]
        - Boundary Safe                 - 2D DP Matrix                    - Levenshtein Matrix
        - DAA Levenshtein Fuzzy         - Reconstruct Subsequence         - Op Transformation Trace
        - Alias Mapping                 - Normalized Score (30%)          - Normalized Score (20%)
        - Skill Coverage (50%)
                │                                 │                                 │
                └─────────────────────────────────┼─────────────────────────────────┘
                                                  ▼
                                    [ Transparent Scoring Engine ]
                      Score = 0.50*(Skill Cov) + 0.30*(LCS) + 0.20*(Edit Distance)
                                                  │
                                                  ▼
                              [ Explainable Recommendations & Audit ]
        ```
        """
    )

    st.info("💡 **Getting Started**: Navigate to the **Resume Analysis** tab in the sidebar or test individual algorithms in the **Algorithm Lab**.")

# =============================================================================
# PAGE 2: RESUME ANALYSIS
# =============================================================================
elif page == "📄 Resume Analysis":
    st.title("📄 Resume vs. Job Description Analysis")
    st.caption("Upload candidate documents or choose preloaded samples to inspect DAA matching scores.")

    # Two columns for inputs
    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        st.markdown("#### 1. Candidate Resume")
        sample_resume_choice = st.selectbox(
            "Load Preloaded Sample Resume (Optional):",
            [
                "-- None (Upload your own) --",
                "Alex Morgan (Full-Stack Python Engineer)",
                "Priya Sharma (Data Scientist / ML Engineer)",
                "Jordan Lee (Frontend React Engineer)",
            ],
        )

        uploaded_file = st.file_uploader(
            "Or Upload Resume (PDF, DOCX, TXT):",
            type=["pdf", "docx", "txt"],
            help="Extracts clean text page-by-page using PyMuPDF or python-docx.",
        )

        resume_text = ""
        if uploaded_file is not None:
            try:
                resume_text = extract_text_from_file(uploaded_file, uploaded_file.name)
                st.success(f"Successfully parsed `{uploaded_file.name}` ({len(resume_text):,} chars).")
            except Exception as e:
                st.error(f"Error parsing document: {str(e)}")
        elif sample_resume_choice != "-- None (Upload your own) --":
            sample_map = {
                "Alex Morgan (Full-Stack Python Engineer)": "alex_morgan_software_engineer.txt",
                "Priya Sharma (Data Scientist / ML Engineer)": "priya_sharma_data_scientist.txt",
                "Jordan Lee (Frontend React Engineer)": "jordan_lee_frontend_developer.txt",
            }
            sample_path = os.path.join(SAMPLE_RESUME_DIR, sample_map[sample_resume_choice])
            resume_text = load_sample_file(sample_path)
            st.info(f"Loaded sample resume for **{sample_resume_choice}**.")

        resume_display = st.text_area(
            "Resume Text (Review or Edit Below):",
            value=resume_text,
            height=280,
            placeholder="Paste raw resume text here if not uploading a file...",
        )

    with col_input2:
        st.markdown("#### 2. Job Description")
        sample_jd_choice = st.selectbox(
            "Load Preloaded Sample Job Description (Optional):",
            [
                "-- None (Paste your own) --",
                "Senior Full-Stack Python Engineer JD",
                "Machine Learning & Data Scientist JD",
                "Senior Frontend React Engineer JD",
            ],
        )

        jd_text = ""
        if sample_jd_choice != "-- None (Paste your own) --":
            jd_map = {
                "Senior Full-Stack Python Engineer JD": "fullstack_python_jd.txt",
                "Machine Learning & Data Scientist JD": "data_scientist_jd.txt",
                "Senior Frontend React Engineer JD": "frontend_react_jd.txt",
            }
            jd_path = os.path.join(SAMPLE_JD_DIR, jd_map[sample_jd_choice])
            jd_text = load_sample_file(jd_path)
            st.info(f"Loaded job description for **{sample_jd_choice}**.")

        jd_display = st.text_area(
            "Job Description Text:",
            value=jd_text,
            height=345,
            placeholder="Paste job posting text here...",
        )

    st.markdown("---")
    analyze_btn = st.button("🚀 Run Algorithmic Matching Analysis", type="primary", use_container_width=True)

    if analyze_btn or (resume_display and jd_display and "last_score" in st.session_state):
        if not resume_display.strip() or not jd_display.strip():
            st.warning("⚠️ Please provide both a Resume and a Job Description to proceed.")
        else:
            with st.spinner("Executing DAA matching pipeline (LCS, Edit Distance, KMP)..."):
                start_clock = time.perf_counter()
                scoring_result = compute_matching_score(resume_display, jd_display)
                total_calc_ms = round((time.perf_counter() - start_clock) * 1000.0, 2)
                st.session_state["last_score"] = scoring_result

            final_score = scoring_result["final_score"]
            components = scoring_result["components"]
            skill_info = scoring_result["skill_analysis"]
            seq_info = scoring_result["sequence_analysis"]
            kmp_info = scoring_result["kmp_evidence"]

            # Top Score Banner
            st.markdown("### 🎯 Match Score Summary")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            # Score color
            score_color = "#3FB950" if final_score >= 75 else "#E3B341" if final_score >= 50 else "#F85149"

            with metric_col1:
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align:center;">
                        <span style="font-size:0.85rem; color:#8B949E; text-transform:uppercase; font-weight:700;">Final Composite Score</span>
                        <h1 style="color:{score_color}; margin:6px 0 0 0; font-size:2.8rem;">{final_score}%</h1>
                        <span style="font-size:0.75rem; color:#8B949E;">Processed in {total_calc_ms} ms</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with metric_col2:
                cov = components["skill_coverage"]
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align:center;">
                        <span style="font-size:0.85rem; color:#8B949E; text-transform:uppercase; font-weight:700;">Skill Coverage (50%)</span>
                        <h2 style="color:#58A6FF; margin:6px 0 0 0;">{cov['score']}%</h2>
                        <span style="font-size:0.75rem; color:#8B949E;">Contribution: +{cov['contribution']} pts</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with metric_col3:
                lcs_c = components["lcs"]
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align:center;">
                        <span style="font-size:0.85rem; color:#8B949E; text-transform:uppercase; font-weight:700;">LCS Similarity (30%)</span>
                        <h2 style="color:#BC8CFF; margin:6px 0 0 0;">{lcs_c['score']}%</h2>
                        <span style="font-size:0.75rem; color:#8B949E;">Tokens: {seq_info['lcs_length']} / {seq_info['resume_token_count'] + seq_info['jd_token_count']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with metric_col4:
                ed_c = components["edit_distance"]
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align:center;">
                        <span style="font-size:0.85rem; color:#8B949E; text-transform:uppercase; font-weight:700;">Edit Distance (20%)</span>
                        <h2 style="color:#39D353; margin:6px 0 0 0;">{ed_c['score']}%</h2>
                        <span style="font-size:0.75rem; color:#8B949E;">Distance: {seq_info['edit_distance']} ops</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Detail Tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏷️ Skill Breakdown",
                "🔍 KMP Keyword Evidence",
                "🧮 Mathematical Scoring Decomposition",
                "💡 Actionable Recommendations",
            ])

            with tab1:
                st.markdown("#### Technical Competencies Analysis")
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(f"**Matched Skills ({len(skill_info['matched_skills'])}):**")
                    if skill_info["matched_skills"]:
                        badges = "".join(
                            f"<span class='badge badge-success'>✓ {k.upper()}</span>"
                            for k in skill_info["matched_skills"]
                        )
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.caption("No matching technical skills identified.")

                with c2:
                    st.markdown(f"**Missing Required Skills ({len(skill_info['missing_skills'])}):**")
                    if skill_info["missing_skills"]:
                        badges = "".join(
                            f"<span class='badge badge-danger'>✗ {k.upper()}</span>"
                            for k in skill_info["missing_skills"]
                        )
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.success("All required technical skills matched!")

                with c3:
                    st.markdown(f"**Additional Skills in Resume ({len(skill_info['additional_skills'])}):**")
                    if skill_info["additional_skills"]:
                        badges = "".join(
                            f"<span class='badge badge-info'>+ {k.upper()}</span>"
                            for k in skill_info["additional_skills"]
                        )
                        st.markdown(badges, unsafe_allow_html=True)
                    else:
                        st.caption("No extra recognized skills detected.")

            with tab2:
                st.markdown("#### Knuth-Morris-Pratt (KMP) Keyword Audit")
                st.caption(
                    "Each JD requirement is searched via KMP in linear O(N+M) time. Starting character indices are recorded below."
                )

                kmp_rows = []
                for skill, data in kmp_info.items():
                    kmp_rows.append({
                        "Skill / Keyword": skill.upper(),
                        "Status": data["status"],
                        "Occurrences in Resume": data["occurrences"],
                        "Match Offsets (0-indexed)": str(data["match_positions"]),
                        "KMP Char Comparisons": data["comparisons"],
                    })

                if kmp_rows:
                    st.dataframe(pd.DataFrame(kmp_rows), use_container_width=True)
                else:
                    st.info("No technical keywords available to verify.")

            with tab3:
                st.markdown("#### Mathematical Scoring Breakdown")
                st.markdown(
                    """
                    <div class="formula-box">
                    Final Score = (0.50 × Skill Coverage) + (0.30 × LCS Score) + (0.20 × Edit Distance Score)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                decomp_df = pd.DataFrame([
                    {
                        "Component Metric": "Skill Coverage (Matched / Required)",
                        "Raw Value": f"{skill_info['matched_count']} / {skill_info['total_required_count']} skills",
                        "Component Score": f"{components['skill_coverage']['score']}%",
                        "Assigned Weight": "50% (0.50)",
                        "Weighted Contribution": f"+{components['skill_coverage']['contribution']} pts",
                    },
                    {
                        "Component Metric": "Token LCS Sequence Alignment",
                        "Raw Value": f"LCS Length: {seq_info['lcs_length']} tokens",
                        "Component Score": f"{components['lcs']['score']}%",
                        "Assigned Weight": "30% (0.30)",
                        "Weighted Contribution": f"+{components['lcs']['contribution']} pts",
                    },
                    {
                        "Component Metric": "Token Levenshtein Closeness",
                        "Raw Value": f"Edit Distance: {seq_info['edit_distance']} ops",
                        "Component Score": f"{components['edit_distance']['score']}%",
                        "Assigned Weight": "20% (0.20)",
                        "Weighted Contribution": f"+{components['edit_distance']['contribution']} pts",
                    },
                    {
                        "Component Metric": "FINAL COMPOSITE SCORE",
                        "Raw Value": "Weighted Linear Combination",
                        "Component Score": f"{final_score}%",
                        "Assigned Weight": "100% (1.00)",
                        "Weighted Contribution": f"{final_score} / 100",
                    },
                ])
                st.dataframe(decomp_df, use_container_width=True)

            with tab4:
                st.markdown("#### Explainable & Ethical Recommendations")
                recs = generate_recommendations(scoring_result, resume_display)

                st.markdown("##### 🚨 Priority Actions")
                for act in recs["high_priority"]:
                    st.markdown(f"- {act}")

                if recs["structural_suggestions"]:
                    st.markdown("##### 📐 Document Structure Audit")
                    for s in recs["structural_suggestions"]:
                        st.markdown(f"- {s}")

                if recs["keyword_alignment_tips"]:
                    st.markdown("##### 🔤 Vocabulary & Phrase Alignment")
                    for tip in recs["keyword_alignment_tips"]:
                        st.markdown(f"- {tip}")

                st.markdown("---")
                st.info(recs["ethical_notice"])

# =============================================================================
# PAGE 3: ALGORITHM LAB
# =============================================================================
elif page == "🔬 Algorithm Lab":
    st.title("🔬 Interactive DAA Algorithm Laboratory")
    st.caption("Inspect Dynamic Programming tables, backtracking trajectories, and pattern matching execution step-by-step.")

    lab_mode = st.radio(
        "Select Algorithm to Demonstrate:",
        [
            "🧬 Longest Common Subsequence (LCS)",
            "📏 Edit Distance (Levenshtein Distance)",
            "🔍 Knuth-Morris-Pratt (KMP) String Matching",
            "⚔️ Naive vs. KMP Head-to-Head Comparison",
        ],
        horizontal=True,
    )

    # 1. LCS VISUALIZER
    if "LCS" in lab_mode:
        st.markdown("### 🧬 Longest Common Subsequence Visualizer")
        st.markdown(
            """
            **Recurrence**:
            - If $X[i-1] == Y[j-1]$: $dp[i][j] = 1 + dp[i-1][j-1]$
            - Else: $dp[i][j] = \\max(dp[i-1][j], dp[i][j-1])$
            """
        )

        c1, c2 = st.columns(2)
        with c1:
            lcs_s1 = st.text_input("Sequence A (String or Words):", value="ABCBDAB")
        with c2:
            lcs_s2 = st.text_input("Sequence B (String or Words):", value="BDCABA")

        token_mode = st.checkbox("Treat as whitespace-separated tokens instead of characters", value=False)

        seq1 = lcs_s1.split() if token_mode else lcs_s1
        seq2 = lcs_s2.split() if token_mode else lcs_s2

        if seq1 and seq2:
            length, dp_table, recon = lcs_full(seq1, seq2)
            opt_len = lcs_length_optimized(seq1, seq2)

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("LCS Length", length)
            res_col2.metric("Space-Optimized Length", opt_len)
            res_col3.metric("Reconstructed Subsequence", str(recon))

            st.markdown("#### 2D Dynamic Programming Table (Grid)")
            # Build DataFrame representation of DP Table
            m, n = len(seq1), len(seq2)
            cols = ["Ø"] + [str(x) for x in seq2]
            rows = ["Ø"] + [str(x) for x in seq1]

            dp_df = pd.DataFrame(dp_table, index=rows, columns=cols)
            st.dataframe(
                dp_df.style.background_gradient(cmap="Blues").format("{:.0f}"),
                use_container_width=True,
            )

    # 2. EDIT DISTANCE VISUALIZER
    elif "Edit Distance" in lab_mode:
        st.markdown("### 📏 Edit Distance (Levenshtein Distance) Visualizer")
        st.markdown(
            """
            **Recurrence**:
            - If $X[i-1] == Y[j-1]$: $dp[i][j] = dp[i-1][j-1]$
            - Else: $dp[i][j] = 1 + \\min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])$
            """
        )

        c1, c2 = st.columns(2)
        with c1:
            ed_s1 = st.text_input("Source Word / Sequence (s1):", value="kitten")
        with c2:
            ed_s2 = st.text_input("Target Word / Sequence (s2):", value="sitting")

        if ed_s1 and ed_s2:
            dist, dp_table, ops = edit_distance_full(ed_s1, ed_s2)
            opt_dist = edit_distance_optimized(ed_s1, ed_s2)
            sim = edit_distance_similarity(ed_s1, ed_s2)

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Minimum Edit Distance", dist)
            res_col2.metric("Space-Optimized Distance", opt_dist)
            res_col3.metric("Normalized Similarity", f"{sim}%")

            st.markdown("#### Step-by-Step Optimal Edit Operations")
            ops_df = pd.DataFrame([
                {
                    "Step": idx + 1,
                    "Operation": op["operation"].upper(),
                    "Source Char": op["char_src"] if op["char_src"] else "—",
                    "Target Char": op["char_tgt"] if op["char_tgt"] else "—",
                    "Operation Cost": op["cost"],
                    "Action Description": op["description"],
                }
                for idx, op in enumerate(ops)
            ])
            st.dataframe(ops_df, use_container_width=True)

            st.markdown("#### Full Levenshtein DP Matrix")
            cols = ["Ø"] + list(ed_s2)
            rows = ["Ø"] + list(ed_s1)
            dp_df = pd.DataFrame(dp_table, index=rows, columns=cols)
            st.dataframe(
                dp_df.style.background_gradient(cmap="Purples").format("{:.0f}"),
                use_container_width=True,
            )

    # 3. KMP VISUALIZER
    elif "KMP" in lab_mode and "Naive" not in lab_mode:
        st.markdown("### 🔍 Knuth-Morris-Pratt (KMP) Visualizer")
        st.markdown(
            """
            KMP leverages the **Longest Proper Prefix which is also Suffix (LPS)** array to avoid
            backtracking in the target text.
            - Preprocessing Time: $O(m)$
            - Search Time: $O(n)$
            - Total Time: $O(n + m)$
            """
        )

        c1, c2 = st.columns(2)
        with c1:
            kmp_p = st.text_input("Search Pattern (P):", value="AABAACAABAA")
        with c2:
            kmp_t = st.text_input("Target Text (T):", value="AABAACAADAABAACAABAA")

        if kmp_p:
            lps, lps_trace = compute_lps_with_trace(kmp_p)

            st.markdown("#### 1. Longest Proper Prefix-Suffix (LPS) Table")
            lps_df = pd.DataFrame({
                "Index (i)": list(range(len(kmp_p))),
                "Character P[i]": list(kmp_p),
                "LPS[i] Value": lps,
            })
            st.dataframe(lps_df.T, use_container_width=True)

            if kmp_t:
                matches, comps, search_trace = kmp_search_with_trace(kmp_t, kmp_p)

                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Matches Found at 0-Indices", str(matches))
                m_col2.metric("Total Character Comparisons", comps)

                with st.expander("Show LPS Preprocessing Execution Trace"):
                    st.dataframe(pd.DataFrame(lps_trace), use_container_width=True)

                with st.expander("Show KMP Step-by-Step Search Trace"):
                    st.dataframe(pd.DataFrame(search_trace), use_container_width=True)

    # 4. NAIVE VS KMP RACE
    elif "Naive vs. KMP" in lab_mode:
        st.markdown("### ⚔️ Naive vs. KMP Head-to-Head Comparison")
        st.markdown(
            """
            Compare brute-force $O(n \\cdot m)$ character comparisons against linear-time $O(n + m)$ KMP
            on custom or worst-case input strings.
            """
        )

        preset = st.selectbox(
            "Select Benchmark Scenario Preset:",
            [
                "Worst-Case: Repetitive String (A...AB in A...B)",
                "Average Case: Real-world English resume sentence",
                "Custom String Input",
            ],
        )

        if "Worst-Case" in preset:
            race_text = "A" * 80 + "B"
            race_pattern = "A" * 10 + "B"
        elif "Average" in preset:
            race_text = "senior software engineer specialized in distributed cloud systems and python"
            race_pattern = "distributed"
        else:
            race_text = "ABCABCDABABCDABCDABDE"
            race_pattern = "ABCDABD"

        c1, c2 = st.columns(2)
        with c1:
            t_input = st.text_input("Text String:", value=race_text)
        with c2:
            p_input = st.text_input("Pattern String:", value=race_pattern)

        if t_input and p_input:
            kmp_m, kmp_c = kmp_search(t_input, p_input)
            naive_m, naive_c = naive_search(t_input, p_input)

            col_race1, col_race2, col_race3 = st.columns(3)
            col_race1.metric("Matches Identified", str(kmp_m))
            col_race2.metric("KMP Comparisons [O(n+m)]", kmp_c)
            col_race3.metric("Naive Comparisons [O(n*m)]", naive_c)

            comp_reduction = round((1.0 - (kmp_c / naive_c)) * 100.0, 2) if naive_c > 0 else 0.0

            st.markdown(
                f"""
                <div class="metric-card" style="border-left:4px solid #3FB950;">
                    <h4 style="margin:0; color:#3FB950;">Empirical Efficiency Gain: {comp_reduction}% Fewer Comparisons</h4>
                    <p style="margin:4px 0 0 0; color:#8B949E;">
                    KMP eliminated redundant checks by shifting according to the precomputed LPS table without rewinding the text pointer.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =============================================================================
# PAGE 4: PERFORMANCE ANALYSIS
# =============================================================================
elif page == "📊 Performance Analysis":
    st.title("📊 Empirical Performance & Complexity Analysis")
    st.caption("Empirical measurements across input dimensions recorded using time.perf_counter and tracemalloc.")

    try:
        bench_df = get_benchmark_dataframe()
    except FileNotFoundError:
        with st.spinner("Initial benchmark results not detected. Running benchmark suite now..."):
            run_all_benchmarks(save_csv=True)
            bench_df = get_benchmark_dataframe()

    tab_p1, tab_p2, tab_p3 = st.tabs([
        "🔍 String Matching: Naive vs. KMP",
        "🧬 LCS: Full vs. Space-Optimized DP",
        "📏 Edit Distance: Full vs. Space-Optimized DP",
    ])

    with tab_p1:
        st.plotly_chart(plot_string_matching_comparison(bench_df), use_container_width=True)
        st.markdown(
            """
            > [!TIP]
            > **Observation**: In worst-case repetitive text ($A...AB$), the Naive algorithm degrades towards
            > quadratic time $O(n \\cdot m)$ as it repeats matching prefixes at every index. In contrast,
            > KMP achieves strictly linear time $O(n + m)$, producing a massive speedup.
            """
        )

    with tab_p2:
        col_lcs1, col_lcs2 = st.columns(2)
        with col_lcs1:
            st.plotly_chart(plot_dp_runtime_comparison(bench_df, "LCS"), use_container_width=True)
        with col_lcs2:
            st.plotly_chart(plot_dp_memory_comparison(bench_df, "LCS"), use_container_width=True)

        st.markdown(
            """
            > [!NOTE]
            > **Space-Time Trade-off**: Space optimization reduces memory consumption from $O(m \\cdot n)$ to $O(\\min(m, n))$,
            > yielding up to a **260x reduction in peak allocated RAM**. Furthermore, enhanced CPU cache locality
            > also lowers execution time compared to allocating full 2D grids in Python.
            """
        )

    with tab_p3:
        col_ed1, col_ed2 = st.columns(2)
        with col_ed1:
            st.plotly_chart(plot_dp_runtime_comparison(bench_df, "Edit Distance"), use_container_width=True)
        with col_ed2:
            st.plotly_chart(plot_dp_memory_comparison(bench_df, "Edit Distance"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Raw Empirical Benchmark Data")
    st.dataframe(bench_df, use_container_width=True)

    csv_data = bench_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Benchmark Dataset (CSV)",
        data=csv_data,
        file_name="resume_match_benchmark_results.csv",
        mime="text/csv",
    )

# =============================================================================
# PAGE 5: ABOUT & METHODOLOGY
# =============================================================================
elif page == "📚 About & Methodology":
    st.title("📚 DAA Project Methodology & Academic Documentation")
    st.caption("Design, recurrence derivations, time-space complexities, and implementation details.")

    st.markdown(
        """
        ### 🎯 Problem Statement
        In modern Applicant Tracking Systems (ATS), candidate resumes are frequently judged using proprietary,
        opaque machine learning black boxes. These systems lack mathematical explainability and often misjudge
        relevant technical candidates due to superficial vocabulary discrepancies.

        **ResumeMatch** addresses this challenge by establishing an algorithm-driven, explainable matching framework
        powered exclusively by foundational algorithms designed and implemented from scratch in pure Python.
        """
    )

    st.markdown("### 📐 Formal Algorithm Specifications")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### 1. Longest Common Subsequence (LCS)")
        st.markdown(
            r"""
            **Mathematical Recurrence**:
            $$
            dp[i][j] = \begin{cases}
            0 & \text{if } i=0 \lor j=0 \\
            1 + dp[i-1][j-1] & \text{if } X[i-1] = Y[j-1] \\
            \max(dp[i-1][j], dp[i][j-1]) & \text{if } X[i-1] \neq Y[j-1]
            \end{cases}
            $$

            - **Time Complexity**: $O(m \cdot n)$ - All $(m+1) \times (n+1)$ subproblems are computed in $O(1)$.
            - **Space Complexity**:
              - Full 2D Table: $O(m \cdot n)$ (Required to backtrack and reconstruct the exact subsequence).
              - Rolling Row Optimization: $O(\min(m, n))$ (Retains only two rows for length computation).
            - **Normalized Metric**:
              $$
              \text{Similarity} = \frac{2 \cdot \text{LCS Length}}{m + n} \times 100\%
              $$
            """
        )

    with col_t2:
        st.markdown("#### 2. Edit Distance (Levenshtein Distance)")
        st.markdown(
            r"""
            **Mathematical Recurrence**:
            $$
            dp[i][j] = \begin{cases}
            i & \text{if } j=0 \\
            j & \text{if } i=0 \\
            dp[i-1][j-1] & \text{if } X[i-1] = Y[j-1] \\
            1 + \min \begin{pmatrix} dp[i-1][j] \\ dp[i][j-1] \\ dp[i-1][j-1] \end{pmatrix} & \text{if } X[i-1] \neq Y[j-1]
            \end{cases}
            $$

            - **Time Complexity**: $O(m \cdot n)$
            - **Space Complexity**:
              - Full Table: $O(m \cdot n)$ (Reconstructs sequence of Insert, Delete, Substitute operations).
              - Space-Optimized: $O(\min(m, n))$
            - **Normalized Metric**:
              $$
              \text{Similarity} = \left(1 - \frac{\text{Distance}}{\max(m, n)}\right) \times 100\%
              $$
            """
        )

    st.markdown("---")
    st.markdown("#### 3. Knuth-Morris-Pratt (KMP) Algorithm")
    st.markdown(
        r"""
        - **LPS Definition**: $LPS[i]$ stores the length of the longest proper prefix of $P[0..i]$ that is also a suffix of $P[0..i]$.
        - **Invariant**: On mismatch at character $j$, text pointer $i$ is never decremented; pattern index resets to $LPS[j-1]$.
        - **Complexity**:
          - LPS Preprocessing: $O(m)$ time, $O(m)$ space.
          - Text Search: $O(n)$ time.
          - Overall: $O(n + m)$ time, $O(m)$ auxiliary space.
        """
    )

    st.markdown("---")
    st.markdown("### ⚖️ Theoretical & Empirical Complexity Comparison")

    comp_summary = pd.DataFrame([
        {
            "Algorithm": "LCS (Full DP Table)",
            "Time (Worst)": "O(m · n)",
            "Time (Average)": "O(m · n)",
            "Auxiliary Space": "O(m · n)",
            "Reconstruction Support": "Yes (Backtrack)",
        },
        {
            "Algorithm": "LCS (Space-Optimized)",
            "Time (Worst)": "O(m · n)",
            "Time (Average)": "O(m · n)",
            "Auxiliary Space": "O(min(m, n))",
            "Reconstruction Support": "No (Length only)",
        },
        {
            "Algorithm": "Edit Distance (Full Table)",
            "Time (Worst)": "O(m · n)",
            "Time (Average)": "O(m · n)",
            "Auxiliary Space": "O(m · n)",
            "Reconstruction Support": "Yes (Full Op Sequence)",
        },
        {
            "Algorithm": "Edit Distance (Optimized)",
            "Time (Worst)": "O(m · n)",
            "Time (Average)": "O(m · n)",
            "Auxiliary Space": "O(min(m, n))",
            "Reconstruction Support": "No (Distance only)",
        },
        {
            "Algorithm": "KMP String Matching",
            "Time (Worst)": "O(n + m)",
            "Time (Average)": "O(n + m)",
            "Auxiliary Space": "O(m)",
            "Reconstruction Support": "Exact Match Offsets",
        },
        {
            "Algorithm": "Naive String Matching",
            "Time (Worst)": "O(n · m)",
            "Time (Average)": "O(n)",
            "Auxiliary Space": "O(1)",
            "Reconstruction Support": "Exact Match Offsets",
        },
    ])
    st.dataframe(comp_summary, use_container_width=True)

    st.markdown("---")
    st.markdown("### 👤 Author & Project Profile")
    st.markdown(
        """
        - **Project Title**: Resume–Job Description Matching System (ResumeMatch)
        - **Subject Area**: Design and Analysis of Algorithms (DAA) Laboratory
        - **Developer**: **Simran Semwal**
        - **GitHub Profile**: [SimranSemwal-ux](https://github.com/SimranSemwal-ux)
        - **License**: MIT Open Source License
        """
    )
