"""
Comprehensive Benchmarking Suite for ResumeMatch
Conducts rigorous empirical comparisons across input dimensions, recording execution time
and memory consumption, and exports data to CSV.

Comparisons:
1. Naive Pattern Matching vs Knuth-Morris-Pratt (KMP) (Worst-Case & Random)
2. Full-Table DP LCS vs Space-Optimized LCS
3. Full-Table DP Edit Distance vs Space-Optimized Edit Distance
"""

import os
import csv
import random
from typing import List, Dict, Any

from algorithms.lcs import lcs_full, lcs_length_optimized
from algorithms.edit_distance import edit_distance_full, edit_distance_optimized
from algorithms.kmp import kmp_search
from algorithms.naive_matching import naive_search
from algorithms.optimized_algorithms import lcs_single_row, edit_distance_single_row
from benchmarking.performance import (
    measure_execution_and_memory,
    generate_random_token_sequence,
    generate_string_matching_worst_case,
    generate_random_string,
)


def run_string_matching_benchmarks() -> List[Dict[str, Any]]:
    """
    Benchmarks Naive String Matching against KMP across increasing text lengths
    under both worst-case and random scenarios.
    """
    results: List[Dict[str, Any]] = []
    text_sizes = [500, 1000, 2500, 5000, 10000, 25000]
    pattern_len = 50

    print("--- Benchmarking String Matching: Naive vs KMP ---")

    for n in text_sizes:
        # Worst-case scenario (A...AB against A...B)
        text_wc, pattern_wc = generate_string_matching_worst_case(n, pattern_len)

        res_naive_wc = measure_execution_and_memory(naive_search, text_wc, pattern_wc, repetitions=5)
        res_kmp_wc = measure_execution_and_memory(kmp_search, text_wc, pattern_wc, repetitions=5)

        results.append({
            "algorithm_category": "String Matching",
            "algorithm_name": "Naive Search (Worst-Case)",
            "input_dimension_n": n,
            "input_dimension_m": pattern_len,
            "input_type": "Worst-Case Repetitive",
            "median_time_ms": res_naive_wc["median_time_ms"],
            "peak_memory_kb": res_naive_wc["peak_memory_kb"],
            "theoretical_time": "O(n * m)",
            "theoretical_space": "O(1)"
        })

        results.append({
            "algorithm_category": "String Matching",
            "algorithm_name": "KMP Search (Worst-Case)",
            "input_dimension_n": n,
            "input_dimension_m": pattern_len,
            "input_type": "Worst-Case Repetitive",
            "median_time_ms": res_kmp_wc["median_time_ms"],
            "peak_memory_kb": res_kmp_wc["peak_memory_kb"],
            "theoretical_time": "O(n + m)",
            "theoretical_space": "O(m)"
        })

        # Random text scenario
        random.seed(42)
        text_rand = generate_random_string(n)
        pattern_rand = generate_random_string(pattern_len)

        res_naive_rand = measure_execution_and_memory(naive_search, text_rand, pattern_rand, repetitions=5)
        res_kmp_rand = measure_execution_and_memory(kmp_search, text_rand, pattern_rand, repetitions=5)

        results.append({
            "algorithm_category": "String Matching",
            "algorithm_name": "Naive Search (Average)",
            "input_dimension_n": n,
            "input_dimension_m": pattern_len,
            "input_type": "Random Text",
            "median_time_ms": res_naive_rand["median_time_ms"],
            "peak_memory_kb": res_naive_rand["peak_memory_kb"],
            "theoretical_time": "O(n * m)",
            "theoretical_space": "O(1)"
        })

        results.append({
            "algorithm_category": "String Matching",
            "algorithm_name": "KMP Search (Average)",
            "input_dimension_n": n,
            "input_dimension_m": pattern_len,
            "input_type": "Random Text",
            "median_time_ms": res_kmp_rand["median_time_ms"],
            "peak_memory_kb": res_kmp_rand["peak_memory_kb"],
            "theoretical_time": "O(n + m)",
            "theoretical_space": "O(m)"
        })

        print(f"  Text Size {n:5d}: Naive WC={res_naive_wc['median_time_ms']}ms | KMP WC={res_kmp_wc['median_time_ms']}ms")

    return results


def run_lcs_benchmarks() -> List[Dict[str, Any]]:
    """
    Benchmarks Full DP Table LCS vs Space-Optimized LCS across token sequence sizes.
    """
    results: List[Dict[str, Any]] = []
    # Token sizes realistic for resumes and job descriptions
    sizes = [50, 100, 200, 400, 600, 800]

    print("--- Benchmarking Longest Common Subsequence (LCS) ---")
    random.seed(101)

    for sz in sizes:
        seq1 = generate_random_token_sequence(sz, vocab_size=30)
        seq2 = generate_random_token_sequence(sz, vocab_size=30)

        # Full table DP
        res_full = measure_execution_and_memory(lcs_full, seq1, seq2, repetitions=3)
        # 2-Row Space-Optimized DP
        res_opt = measure_execution_and_memory(lcs_length_optimized, seq1, seq2, repetitions=3)
        # 1-Row Single Array DP
        res_single = measure_execution_and_memory(lcs_single_row, seq1, seq2, repetitions=3)

        results.append({
            "algorithm_category": "LCS",
            "algorithm_name": "Full DP Table (with Backtrack)",
            "input_dimension_n": sz,
            "input_dimension_m": sz,
            "input_type": "Token Sequence",
            "median_time_ms": res_full["median_time_ms"],
            "peak_memory_kb": res_full["peak_memory_kb"],
            "theoretical_time": "O(m * n)",
            "theoretical_space": "O(m * n)"
        })

        results.append({
            "algorithm_category": "LCS",
            "algorithm_name": "Space-Optimized (2 Rows)",
            "input_dimension_n": sz,
            "input_dimension_m": sz,
            "input_type": "Token Sequence",
            "median_time_ms": res_opt["median_time_ms"],
            "peak_memory_kb": res_opt["peak_memory_kb"],
            "theoretical_time": "O(m * n)",
            "theoretical_space": "O(min(m, n))"
        })

        results.append({
            "algorithm_category": "LCS",
            "algorithm_name": "Space-Optimized (1 Row)",
            "input_dimension_n": sz,
            "input_dimension_m": sz,
            "input_type": "Token Sequence",
            "median_time_ms": res_single["median_time_ms"],
            "peak_memory_kb": res_single["peak_memory_kb"],
            "theoretical_time": "O(m * n)",
            "theoretical_space": "O(min(m, n))"
        })

        print(f"  Size {sz:4d}x{sz:4d}: Full DP={res_full['median_time_ms']}ms ({res_full['peak_memory_kb']} KB) | Opt 2-Row={res_opt['median_time_ms']}ms ({res_opt['peak_memory_kb']} KB)")

    return results


def run_edit_distance_benchmarks() -> List[Dict[str, Any]]:
    """
    Benchmarks Full DP Table Edit Distance vs Space-Optimized Edit Distance.
    """
    results: List[Dict[str, Any]] = []
    sizes = [50, 100, 200, 400, 600, 800]

    print("--- Benchmarking Edit Distance (Levenshtein) ---")
    random.seed(202)

    for sz in sizes:
        seq1 = generate_random_token_sequence(sz, vocab_size=30)
        seq2 = generate_random_token_sequence(sz, vocab_size=30)

        # Full table DP
        res_full = measure_execution_and_memory(edit_distance_full, seq1, seq2, repetitions=3)
        # 2-Row Space-Optimized DP
        res_opt = measure_execution_and_memory(edit_distance_optimized, seq1, seq2, repetitions=3)
        # 1-Row Single Array DP
        res_single = measure_execution_and_memory(edit_distance_single_row, seq1, seq2, repetitions=3)

        results.append({
            "algorithm_category": "Edit Distance",
            "algorithm_name": "Full DP Table (with Operations)",
            "input_dimension_n": sz,
            "input_dimension_m": sz,
            "input_type": "Token Sequence",
            "median_time_ms": res_full["median_time_ms"],
            "peak_memory_kb": res_full["peak_memory_kb"],
            "theoretical_time": "O(m * n)",
            "theoretical_space": "O(m * n)"
        })

        results.append({
            "algorithm_category": "Edit Distance",
            "algorithm_name": "Space-Optimized (2 Rows)",
            "input_dimension_n": sz,
            "input_dimension_m": sz,
            "input_type": "Token Sequence",
            "median_time_ms": res_opt["median_time_ms"],
            "peak_memory_kb": res_opt["peak_memory_kb"],
            "theoretical_time": "O(m * n)",
            "theoretical_space": "O(min(m, n))"
        })

        results.append({
            "algorithm_category": "Edit Distance",
            "algorithm_name": "Space-Optimized (1 Row)",
            "input_dimension_n": sz,
            "input_dimension_m": sz,
            "input_type": "Token Sequence",
            "median_time_ms": res_single["median_time_ms"],
            "peak_memory_kb": res_single["peak_memory_kb"],
            "theoretical_time": "O(m * n)",
            "theoretical_space": "O(min(m, n))"
        })

        print(f"  Size {sz:4d}x{sz:4d}: Full DP={res_full['median_time_ms']}ms ({res_full['peak_memory_kb']} KB) | Opt 2-Row={res_opt['median_time_ms']}ms ({res_opt['peak_memory_kb']} KB)")

    return results


def run_all_benchmarks(save_csv: bool = True) -> List[Dict[str, Any]]:
    """
    Executes full benchmarking suite and writes results to CSV.
    """
    all_results = []
    all_results.extend(run_string_matching_benchmarks())
    all_results.extend(run_lcs_benchmarks())
    all_results.extend(run_edit_distance_benchmarks())

    if save_csv:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(curr_dir, "benchmark_results.csv")
        fieldnames = [
            "algorithm_category",
            "algorithm_name",
            "input_dimension_n",
            "input_dimension_m",
            "input_type",
            "median_time_ms",
            "peak_memory_kb",
            "theoretical_time",
            "theoretical_space"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[+] Benchmark results successfully saved to: {csv_path}")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
