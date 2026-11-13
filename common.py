"""Shared measurement helpers, so every experiment is timed in exactly the same way.

Timing method (used by every script in this project):
  1. one warm-up call (not timed);
  2. RUNS separate timed runs; each run makes CALLS calls and records the
     average time per call;
  3. we report mean +/- standard deviation over the runs, in milliseconds.
Memory is measured separately (tracemalloc slows Python down, so it is never
switched on while timing).
"""
import statistics
import time
import tracemalloc

import numpy as np

RUNS = 5  # default number of timed runs for every measurement


def integrate_trapezoid(y, x):
    """Integrate with the NumPy API available in this environment."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(y, x)


def time_ms(function, runs=RUNS, calls=1, warmup=1):
    """Return (mean_ms, sd_ms, per_run_ms) for one call of `function`."""
    for _ in range(warmup):
        function()
    per_run = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        per_run.append((time.perf_counter() - start) / calls * 1000)
    sd = statistics.stdev(per_run) if len(per_run) > 1 else 0.0
    return statistics.mean(per_run), sd, per_run


def peak_memory_mb(function):
    """Peak memory (MB) allocated during one call of `function`."""
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def fmt(mean, sd, decimals=3):
    """Format a timing as 'mean ± sd ms'."""
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f} ms"
