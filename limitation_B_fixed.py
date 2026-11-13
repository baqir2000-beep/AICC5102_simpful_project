import itertools
import statistics
import random

from simpful import *
from fast_mamdani import fast_mamdani
from common import time_ms, peak_memory_mb, RUNS


TERMS = ["low", "medium", "high"]
OUTPUT_TERMS = ["small", "average", "large"]
# Centroids of the three output sets, used as the crisp outputs of the Sugeno twin
# (ablation_mamdani_vs_sugeno.py): small = Tri(0,0,10) -> 10/3,
# average = Tri(0,10,20) -> 10, large = Trap(10,20,25,25) -> 235/12.
OUTPUT_CENTROIDS = {"small": 10 / 3, "average": 10.0, "large": 235 / 12}


def random_spec(num_inputs, seed):
    """The random part of a rule base: one output term per full-grid rule, then the
    input values. Shared by Simpful, Scikit-Fuzzy and the Sugeno twin, so all three
    get exactly the same rules and inputs for a given seed."""
    random.seed(seed)
    rules = [(combination, random.choice(OUTPUT_TERMS))
             for combination in itertools.product(TERMS, repeat=num_inputs)]
    inputs = [random.uniform(0, 10) for _ in range(num_inputs)]
    return rules, inputs


def build_system(num_inputs, seed, sugeno=False):
    """Full-grid random rule base (3^n rules). sugeno=True gives the zero-order
    Sugeno twin: same inputs and rules, each output set replaced by its centroid."""
    rule_spec, inputs = random_spec(num_inputs, seed)

    FS = FuzzySystem(show_banner=False)

    # --------------------------------------------------------
    # INPUT VARIABLES
    # --------------------------------------------------------

    for i in range(num_inputs):
        FS.add_linguistic_variable(
            f"Input{i}",
            LinguisticVariable(
                [FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="low"),
                 FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="medium"),
                 FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="high")],
                universe_of_discourse=[0, 10]
            )
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    if sugeno:
        for term, value in OUTPUT_CENTROIDS.items():
            FS.set_crisp_output_value(term, value)
    else:
        FS.add_linguistic_variable(
            "Output",
            LinguisticVariable(
                [FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small"),
                 FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average"),
                 FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="large")],
                universe_of_discourse=[0, 25]
            )
        )

    # --------------------------------------------------------
    # RULES
    # --------------------------------------------------------

    rules = []
    for combination, output_term in rule_spec:
        antecedent = " AND ".join(f"(Input{i} IS {term})" for i, term in enumerate(combination))
        rules.append(f"IF {antecedent} THEN (Output IS {output_term})")
    FS.add_rules(rules)

    for i, value in enumerate(inputs):
        FS.set_variable(f"Input{i}", value)

    return FS, len(rules)


# ============================================================
# RUN TEST: SAME PROCEDURE FOR SIMPFUL AND FOR OUR FIX
# ============================================================
# Settings (same for both methods):
#   - Simpful at its default 1000 subdivisions (how users normally run it)
#   - 5 seeds = 5 different random rule bases and inputs per size
#   - per seed: warm-up + 3 timed runs (common.time_ms); seed time = mean of runs
#   - reported: mean ± sd over the 5 seeds
#   - memory measured in a separate call (tracemalloc is never on while timing)

SUBDIVISIONS = 1000
TIMED_RUNS = 3

input_counts = [2, 4, 6, 8]
seeds = [1, 2, 3, 4, 5]


def simpful_run(FS):
    return FS.Mamdani_inference(["Output"], subdivisions=SUBDIVISIONS)["Output"]


def ours_run(FS):
    return fast_mamdani(FS, ["Output"])["Output"]


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


if __name__ == "__main__":
    print()
    print(f"Seeds: {len(seeds)} | Simpful subdivisions: {SUBDIVISIONS} | "
          f"time = mean ± sd over seeds (each seed: warm-up + {TIMED_RUNS} runs)")
    print("Inputs | Rules | Simpful time | Ours time | Speed-up | Simpful mem | Ours mem | Max |diff|")
    print("-" * 100)

    for num_inputs in input_counts:
        t_s, t_o, m_s, m_o, diffs = [], [], [], [], []
        for seed in seeds:
            FS, rule_count = build_system(num_inputs, seed)
            diffs.append(abs(simpful_run(FS) - ours_run(FS)))
            t_s.append(time_ms(lambda: simpful_run(FS), runs=TIMED_RUNS)[0])
            t_o.append(time_ms(lambda: ours_run(FS), runs=TIMED_RUNS)[0])
            m_s.append(peak_memory_mb(lambda: simpful_run(FS)))
            m_o.append(peak_memory_mb(lambda: ours_run(FS)))
        print(
            f"{num_inputs} | {rule_count} | "
            f"{statistics.mean(t_s):.2f} ± {sd(t_s):.2f} ms | "
            f"{statistics.mean(t_o):.2f} ± {sd(t_o):.2f} ms | "
            f"{statistics.mean(t_s) / statistics.mean(t_o):.1f}x | "
            f"{statistics.mean(m_s):.2f} MB | {statistics.mean(m_o):.2f} MB | "
            f"{max(diffs):.4f}",
            flush=True
        )
    print("Max |diff| = Simpful's own discretisation error (ours is exact).")

