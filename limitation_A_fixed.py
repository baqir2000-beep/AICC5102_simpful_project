import numpy as np

from common import time_ms, fmt, RUNS, integrate_trapezoid
from fast_mamdani import fast_mamdani

from simpful import *


# ============================================================
# BUILD THE SAME TIPPING FIS AS LISTING 1
# ============================================================

FS = FuzzySystem(show_banner=False)

# Service
S1 = FuzzySet(
    function=Triangular_MF(a=0, b=0, c=5),
    term="poor"
)

S2 = FuzzySet(
    function=Triangular_MF(a=0, b=5, c=10),
    term="good"
)

S3 = FuzzySet(
    function=Triangular_MF(a=5, b=10, c=10),
    term="excellent"
)

FS.add_linguistic_variable(
    "Service",
    LinguisticVariable(
        [S1, S2, S3],
        universe_of_discourse=[0, 10]
    )
)

# Food
F1 = FuzzySet(
    function=Triangular_MF(a=0, b=0, c=10),
    term="rancid"
)

F2 = FuzzySet(
    function=Triangular_MF(a=0, b=10, c=10),
    term="delicious"
)

FS.add_linguistic_variable(
    "Food",
    LinguisticVariable(
        [F1, F2],
        universe_of_discourse=[0, 10]
    )
)

# Tip
T1 = FuzzySet(
    function=Triangular_MF(a=0, b=0, c=10),
    term="small"
)

T2 = FuzzySet(
    function=Triangular_MF(a=0, b=10, c=20),
    term="average"
)

T3 = FuzzySet(
    function=Trapezoidal_MF(a=10, b=20, c=25, d=25),
    term="generous"
)

FS.add_linguistic_variable(
    "Tip",
    LinguisticVariable(
        [T1, T2, T3],
        universe_of_discourse=[0, 25]
    )
)

rules = [
    "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)",
    "IF (Service IS good) THEN (Tip IS average)",
    "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
]

FS.add_rules(rules)

FS.set_variable("Service", 4)
FS.set_variable("Food", 8)


# ============================================================
# HIGH-RESOLUTION REFERENCE
# ============================================================

def reference_tip(service, food, points=1_000_001):

    x = np.linspace(0, 25, points)

    # Input membership functions
    poor = np.interp(service, [0, 5], [1, 0])
    good = np.interp(service, [0, 5, 10], [0, 1, 0])
    excellent = np.interp(service, [5, 10], [0, 1])

    rancid = np.interp(food, [0, 10], [1, 0])
    delicious = np.interp(food, [0, 10], [0, 1])

    # Rule firing strengths
    a1 = max(poor, rancid)
    a2 = good
    a3 = max(excellent, delicious)

    # Output membership functions
    small = np.clip(
        1 - x / 10,
        0,
        1
    )

    average = np.where(
        x <= 10,
        x / 10,
        (20 - x) / 10
    )

    average = np.clip(average, 0, 1)

    generous = np.piecewise(
    x,
    [
        x < 10,
        (x >= 10) & (x < 20),
        x >= 20
    ],
    [
        0,
        lambda z: (z - 10) / 10,
        1
    ]
    )

    # Clip output sets
    r1 = np.minimum(a1, small)
    r2 = np.minimum(a2, average)
    r3 = np.minimum(a3, generous)

    # Aggregate
    aggregated = np.maximum.reduce([r1, r2, r3])

    # Centroid
    numerator = integrate_trapezoid(x * aggregated, x)
    denominator = integrate_trapezoid(aggregated, x)

    return numerator / denominator


if __name__ == "__main__":
    reference = reference_tip(4, 8)

    print("High-resolution reference:", reference)


    # ============================================================
    # PART 1: ONE INPUT (Service = 4, Food = 8), AS IN LISTING 1
    # ============================================================
    # Timing: common.time_ms -> warm-up, then RUNS runs of 10 calls each,
    # reported as mean +/- standard deviation (ms per call).

    settings = [10, 100, 1000, 10000]

    print()
    print(f"Part 1 - input (4, 8); time = mean ± sd over {RUNS} runs of 10 calls")
    print("Method | Result | Error | Time")

    for subdivisions in settings:
        value = FS.Mamdani_inference(["Tip"], subdivisions=subdivisions)["Tip"]
        mean, sd, _ = time_ms(
            lambda: FS.Mamdani_inference(["Tip"], subdivisions=subdivisions),
            calls=10
        )
        print(
            f"Simpful {subdivisions} subdivisions | {value:.6f} | "
            f"{abs(value - reference):.6f} | {fmt(mean, sd)}"
        )

    value = fast_mamdani(FS, ["Tip"])["Tip"]
    mean, sd, _ = time_ms(lambda: fast_mamdani(FS, ["Tip"]), calls=10)
    print(f"Ours (exact) | {value:.6f} | {abs(value - reference):.2e} | {fmt(mean, sd)}")


    # ============================================================
    # PART 2: MANY INPUTS (11 x 11 = 121 combinations)
    # ============================================================
    # One input can hide the worst case, so we repeat the accuracy test on
    # every combination of Service and Food in {0, 1, ..., 10}.

    grid = np.linspace(0, 10, 11)
    references = {(s, f): reference_tip(s, f) for s in grid for f in grid}


    def errors_over_grid(infer):
        errors = []
        for (s, f), ref in references.items():
            FS.set_variable("Service", s)
            FS.set_variable("Food", f)
            errors.append(abs(infer() - ref))
        return max(errors), float(np.mean(errors))


    print()
    print("Part 2 - 121 inputs (Service, Food = 0..10)")
    print("Method | Max error | Mean error")

    for subdivisions in settings:
        worst, average = errors_over_grid(
            lambda: FS.Mamdani_inference(["Tip"], subdivisions=subdivisions)["Tip"]
        )
        print(f"Simpful {subdivisions} subdivisions | {worst:.6f} | {average:.6f}")

    worst, average = errors_over_grid(lambda: fast_mamdani(FS, ["Tip"])["Tip"])
    print(f"Ours (exact) | {worst:.2e} | {average:.2e}")

    FS.set_variable("Service", 4)
    FS.set_variable("Food", 8)
