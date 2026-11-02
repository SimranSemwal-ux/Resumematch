"""
Graph Generator Module for ResumeMatch
Renders interactive Plotly charts illustrating empirical time and memory complexities
alongside theoretical curves.
"""

import os
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple


def get_benchmark_dataframe(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads benchmark results into a pandas DataFrame.
    """
    if csv_path is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(curr_dir, "benchmark_results.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark CSV not found at {csv_path}. Run benchmark.py first.")

    return pd.read_csv(csv_path)


def plot_string_matching_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Plots execution time vs input size for Naive vs KMP string matching.
    """
    sm_df = df[df["algorithm_category"] == "String Matching"]

    fig = go.Figure()

    colors = {
        "Naive Search (Worst-Case)": "#FF4B4B",
        "KMP Search (Worst-Case)": "#00CEC9",
        "Naive Search (Average)": "#FFA502",
        "KMP Search (Average)": "#6C5CE7"
    }

    dash_styles = {
        "Naive Search (Worst-Case)": "solid",
        "KMP Search (Worst-Case)": "solid",
        "Naive Search (Average)": "dash",
        "KMP Search (Average)": "dash"
    }

    for algo_name in sm_df["algorithm_name"].unique():
        sub = sm_df[sm_df["algorithm_name"] == algo_name].sort_values("input_dimension_n")
        fig.add_trace(go.Scatter(
            x=sub["input_dimension_n"],
            y=sub["median_time_ms"],
            mode="lines+markers",
            name=algo_name,
            line=dict(
                color=colors.get(algo_name, "#FFFFFF"),
                width=3,
                dash=dash_styles.get(algo_name, "solid")
            ),
            marker=dict(size=8),
            hovertemplate="<b>%{fullData.name}</b><br>Text Length: %{x:,}<br>Runtime: %{y:.4f} ms<extra></extra>"
        ))

    fig.update_layout(
        title="<b>String Matching: Naive vs KMP Runtime Analysis</b>",
        xaxis_title="Text Length (n) [Pattern Length m = 50]",
        yaxis_title="Median Execution Time (ms)",
        template="plotly_dark",
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font=dict(family="Inter, sans-serif", size=13, color="#E6EDF3"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(22, 27, 34, 0.8)"
        ),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


def plot_dp_runtime_comparison(df: pd.DataFrame, category: str = "LCS") -> go.Figure:
    """
    Plots runtime comparison between Full DP Table and Space-Optimized variants.
    """
    dp_df = df[df["algorithm_category"] == category].sort_values("input_dimension_n")

    fig = go.Figure()

    colors = {
        "Full DP Table (with Backtrack)": "#FF6B6B",
        "Full DP Table (with Operations)": "#FF6B6B",
        "Space-Optimized (2 Rows)": "#48DBFB",
        "Space-Optimized (1 Row)": "#1DD1A1"
    }

    for algo_name in dp_df["algorithm_name"].unique():
        sub = dp_df[dp_df["algorithm_name"] == algo_name]
        fig.add_trace(go.Scatter(
            x=sub["input_dimension_n"],
            y=sub["median_time_ms"],
            mode="lines+markers",
            name=algo_name,
            line=dict(color=colors.get(algo_name, "#FFFFFF"), width=3),
            marker=dict(size=8),
            hovertemplate="<b>%{fullData.name}</b><br>Dimensions: %{x}x%{x}<br>Runtime: %{y:.3f} ms<extra></extra>"
        ))

    fig.update_layout(
        title=f"<b>{category}: Full DP Table vs Space-Optimized Execution Time</b>",
        xaxis_title="Sequence Length (m = n tokens)",
        yaxis_title="Median Execution Time (ms)",
        template="plotly_dark",
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font=dict(family="Inter, sans-serif", size=13, color="#E6EDF3"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(22, 27, 34, 0.8)"
        ),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


def plot_dp_memory_comparison(df: pd.DataFrame, category: str = "LCS") -> go.Figure:
    """
    Plots peak memory consumption (KB) illustrating O(m*n) vs O(min(m, n)).
    """
    dp_df = df[df["algorithm_category"] == category].sort_values("input_dimension_n")

    fig = go.Figure()

    colors = {
        "Full DP Table (with Backtrack)": "#FF7675",
        "Full DP Table (with Operations)": "#FF7675",
        "Space-Optimized (2 Rows)": "#74B9FF",
        "Space-Optimized (1 Row)": "#55EFC4"
    }

    for algo_name in dp_df["algorithm_name"].unique():
        sub = dp_df[dp_df["algorithm_name"] == algo_name]
        fig.add_trace(go.Bar(
            x=[f"{x}x{x}" for x in sub["input_dimension_n"]],
            y=sub["peak_memory_kb"],
            name=algo_name,
            marker_color=colors.get(algo_name, "#FFFFFF"),
            hovertemplate="<b>%{fullData.name}</b><br>Dimension: %{x}<br>Peak Memory: %{y:.2f} KB<extra></extra>"
        ))

    fig.update_layout(
        title=f"<b>{category}: Peak Memory Consumption [O(m*n) vs O(min(m, n))]</b>",
        xaxis_title="Input Grid Dimension (m x n)",
        yaxis_title="Peak Memory (KB) [Log Scale]",
        yaxis_type="log",
        barmode="group",
        template="plotly_dark",
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font=dict(family="Inter, sans-serif", size=13, color="#E6EDF3"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(22, 27, 34, 0.8)"
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


def export_charts_to_html(output_dir: Optional[str] = None) -> None:
    """
    Generates and saves standalone interactive HTML charts into the assets directory.
    """
    if output_dir is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(curr_dir)
        output_dir = os.path.join(project_root, "assets", "graphs")

    os.makedirs(output_dir, exist_ok=True)
    df = get_benchmark_dataframe()

    fig_sm = plot_string_matching_comparison(df)
    fig_sm.write_html(os.path.join(output_dir, "string_matching_comparison.html"))

    fig_lcs_time = plot_dp_runtime_comparison(df, "LCS")
    fig_lcs_time.write_html(os.path.join(output_dir, "lcs_runtime_comparison.html"))

    fig_lcs_mem = plot_dp_memory_comparison(df, "LCS")
    fig_lcs_mem.write_html(os.path.join(output_dir, "lcs_memory_comparison.html"))

    fig_ed_time = plot_dp_runtime_comparison(df, "Edit Distance")
    fig_ed_time.write_html(os.path.join(output_dir, "edit_distance_runtime_comparison.html"))

    fig_ed_mem = plot_dp_memory_comparison(df, "Edit Distance")
    fig_ed_mem.write_html(os.path.join(output_dir, "edit_distance_memory_comparison.html"))

    print(f"[+] All interactive charts exported to {output_dir}")


if __name__ == "__main__":
    export_charts_to_html()
