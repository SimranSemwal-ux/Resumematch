"""
Benchmarking package for ResumeMatch.
Exports benchmark runners, profilers, and graph generators.
"""

from .performance import (
    measure_execution_and_memory,
    generate_random_token_sequence,
    generate_string_matching_worst_case,
    generate_random_string,
)
from .benchmark import run_all_benchmarks
from .generate_graphs import (
    get_benchmark_dataframe,
    plot_string_matching_comparison,
    plot_dp_runtime_comparison,
    plot_dp_memory_comparison,
    export_charts_to_html,
)

__all__ = [
    "measure_execution_and_memory",
    "generate_random_token_sequence",
    "generate_string_matching_worst_case",
    "generate_random_string",
    "run_all_benchmarks",
    "get_benchmark_dataframe",
    "plot_string_matching_comparison",
    "plot_dp_runtime_comparison",
    "plot_dp_memory_comparison",
    "export_charts_to_html",
]
