"""Ablation study: which part of fast_mamdani gives the speed, which gives the accuracy?

We start from the full fix and remove ONE component at a time:
  - active : evaluate only the rules that can fire (lookup by non-zero terms)
  - cache  : reuse each input's membership values instead of recomputing per rule
  - merge  : combine rules with the same output term and drop rules that do not fire
  - exact  : compute the centroid exactly instead of on a 1000-point grid
  - vector : NumPy instead of a Python loop (only used when 'exact' is off,
             so it is tested together with removing 'exact')
Systems: the paper's tipping FIS (accuracy over 121 inputs + time at (4, 8)) and
the random rule bases from limitation_B_fixed.py (4, 6 and 8 inputs, 5 seeds).
Timing: common.time_ms (warm-up + 3 runs per seed), reported as mean ± sd over seeds.
A configuration is stopped as soon as the warm-up call for ANY seed takes longer
than TIME_CAP seconds, and reported as "> TIME_CAP s (seed k)", where k is the
first seed that hit the cap (we do not wait minutes for one inference).

Why there is a "- merge - exact" row: the components interact. The exact centroid
looks for crossings between every pair of output sets on each segment, so its cost
grows with the SQUARE of the number of sets. With merge, there are at most 3 sets
(one per output term); without merge, there is one set per rule (up to 6561).
Removing merge alone therefore measures merge AND the quadratic cost of exact
together. Removing both shows how much merge is worth when the grid is used.
"""
import statistics
import time

import numpy as np

from common import time_ms
from fast_mamdani import fast_mamdani
from limitation_A_fixed import FS as TIPPING, reference_tip
from limitation_B_fixed import build_system

TIME_CAP = 20.0   # seconds for one call
SEEDS = [1, 2, 3, 4, 5]
SIZES = [4, 6, 8]

CONFIGS = {
    "Simpful (baseline)":        None,
    "Full fix":                  {},
    "- active":                  {"active": False},
    "- cache":                   {"cache": False},
    "- merge":                   {"merge": False},
    "- exact (NumPy grid)":      {"exact": False},
    "- exact - vector (loop)":   {"exact": False, "vector": False},
    "- merge - exact (grid)":    {"merge": False, "exact": False},
}


def infer(FS, output, config):
    if config is None:
        return FS.Mamdani_inference([output], subdivisions=1000, ignore_warnings=True)[output]
    return fast_mamdani(FS, [output], **config)[output]


def tipping_row(config):
    """Max error over 121 inputs, and time at input (4, 8)."""
    errors = []
    for s in np.linspace(0, 10, 11):
        for f in np.linspace(0, 10, 11):
            TIPPING.set_variable("Service", s)
            TIPPING.set_variable("Food", f)
            errors.append(abs(infer(TIPPING, "Tip", config) - reference_tip(s, f)))
    TIPPING.set_variable("Service", 4)
    TIPPING.set_variable("Food", 8)
    mean, sd, _ = time_ms(lambda: infer(TIPPING, "Tip", config), calls=10)
    return max(errors), mean, sd


def random_row(config, n):
    """Mean ± sd time over seeds, or a '> cap' marker if too slow."""
    times = []
    for seed in SEEDS:
        FS, _ = build_system(n, seed)
        start = time.perf_counter()
        infer(FS, "Output", config)                      # warm-up (also the cap check)
        if time.perf_counter() - start > TIME_CAP:
            return f"> {TIME_CAP:.0f} s (seed {seed})"
        times.append(time_ms(lambda: infer(FS, "Output", config), runs=3, warmup=0)[0])
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    return f"{statistics.mean(times):.1f} ± {sd:.1f} ms"


if __name__ == "__main__":
    print("=== ablation.py\n")
    print("Remove one component at a time from the full fix.")
    print("Tipping: max error over 121 inputs; time at (4, 8), mean ± sd over 5 runs of 10 calls.")
    print("Random rule bases: mean ± sd over 5 seeds (each seed: warm-up + 3 runs).\n")
    header = ["Configuration", "Tipping max error", "Tipping time"] + \
             [f"{n} inputs ({3 ** n} rules)" for n in SIZES]
    print(" | ".join(header))
    print("-" * 120)
    for name, config in CONFIGS.items():
        err, mean, sd = tipping_row(config)
        cells = [name, f"{err:.2e}", f"{mean:.3f} ± {sd:.3f} ms"]
        cells += [random_row(config, n) for n in SIZES]
        print(" | ".join(cells), flush=True)
    print()
    print("Note: '- merge' and '- exact' interact. Without merge, the exact centroid")
    print("compares every pair of output sets (one per rule), so its cost grows with")
    print("rules^2; the '- merge - exact' row shows merge's effect on the grid method.")
