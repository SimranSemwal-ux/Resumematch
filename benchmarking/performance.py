"""
Performance Measurement Utilities for ResumeMatch
Provides standardized time and memory profiling using time.perf_counter and tracemalloc.
"""

import time
import tracemalloc
import statistics
import random
import string
from typing import Callable, Any, Tuple, Dict, List


def measure_execution_and_memory(
    func: Callable[..., Any],
    *args,
    repetitions: int = 5,
    **kwargs
) -> Dict[str, float]:
    """
    Measures median execution time and peak memory consumption for a given callable.

    Methodology:
        - Executes function `repetitions` times to compute stable median runtime.
        - Traces memory using standard library `tracemalloc`.
        - Reports time in milliseconds (ms) and peak memory in kilobytes (KB).

    Args:
        func: Algorithm callable.
        *args: Positional arguments to pass to func.
        repetitions: Number of benchmark trials (default: 5).
        **kwargs: Keyword arguments to pass to func.

    Returns:
        Dict[str, float]: {
            'median_time_ms': float,
            'mean_time_ms': float,
            'std_dev_ms': float,
            'peak_memory_kb': float,
            'repetitions': int
        }
    """
    runtimes_ms: List[float] = []

    # Warmup run to prime JIT/Python bytecode caches
    try:
        _ = func(*args, **kwargs)
    except Exception:
        pass

    # Trace peak memory over a dedicated single execution to isolate GC interference
    tracemalloc.start()
    _ = func(*args, **kwargs)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_kb = round(peak_bytes / 1024.0, 2)

    # Time measurement runs
    for _ in range(repetitions):
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        end = time.perf_counter()
        runtimes_ms.append((end - start) * 1000.0)

    median_ms = round(statistics.median(runtimes_ms), 4)
    mean_ms = round(statistics.mean(runtimes_ms), 4)
    std_dev_ms = round(statistics.stdev(runtimes_ms) if len(runtimes_ms) > 1 else 0.0, 4)

    return {
        "median_time_ms": median_ms,
        "mean_time_ms": mean_ms,
        "std_dev_ms": std_dev_ms,
        "peak_memory_kb": peak_kb,
        "repetitions": repetitions
    }


def generate_random_token_sequence(length: int, vocab_size: int = 50) -> List[str]:
    """
    Generates a synthetic list of string tokens drawn from a vocabulary of size `vocab_size`.
    """
    vocab = [f"token_{i}" for i in range(vocab_size)]
    return [random.choice(vocab) for _ in range(length)]


def generate_string_matching_worst_case(text_len: int, pattern_len: int) -> Tuple[str, str]:
    """
    Constructs classical worst-case input for naive string matching:
    Text = 'A' * (text_len - 1) + 'B'
    Pattern = 'A' * (pattern_len - 1) + 'B'

    In this configuration, Naive string matching checks (pattern_len - 1) characters
    before failing at every alignment, triggering O(n * m) character comparisons.
    In contrast, KMP uses its LPS table to advance in linear O(n + m) time.
    """
    if pattern_len > text_len:
        pattern_len = text_len

    text = "A" * (text_len - 1) + "B"
    pattern = "A" * (pattern_len - 1) + "B"
    return text, pattern


def generate_random_string(length: int) -> str:
    """
    Generates a random alphanumeric string of specified length.
    """
    chars = string.ascii_letters + string.digits + " "
    return "".join(random.choice(chars) for _ in range(length))
