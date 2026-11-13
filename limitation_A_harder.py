"""Limitation A, harder conditions (proposal section 4A): narrow and Gaussian/sigmoid
output sets.

Same tipping inputs and rules as Listing 1; only the three output sets change:
  narrow   : narrow triangles/trapezoid (widths 1-3 on a 0-25 universe), so a grid
             with spacing 0.025 has few points inside each set
  gaussian : Gaussian output sets
  sigmoid  : decreasing sigmoid / Gaussian / increasing sigmoid
For each variant we report, over the same 121 inputs as limitation_A_fixed.py, the
max and mean error of Simpful (100, 1000, 10000 subdivisions) and of fast_mamdani
against a 1,000,001-point trapezoidal reference, plus the time at input (4, 8).

For narrow sets fast_mamdani is exact. For Gaussian/sigmoid sets it is NOT exact:
it falls back to a 200,001-point grid with the trapezoidal rule.
"""
import numpy as np
from simpful import (FuzzySet, FuzzySystem, LinguisticVariable, Gaussian_MF,
                     Sigmoid_MF, InvSigmoid_MF, Triangular_MF, Trapezoidal_MF)

from common import time_ms, fmt, RUNS, integrate_trapezoid
from fast_mamdani import fast_mamdani

SUBDIVISIONS = [100, 1000, 10000]
GRID = np.linspace(0, 10, 11)


# ------------------------------------------------ output-set variants
# Each entry: name -> list of (term, Simpful function, numpy membership on array x)
def _tri(a, b, c):
    return lambda x: np.interp(x, [a, b, c], [0, 1, 0])


def _trap(a, b, c, d):
    return lambda x: np.interp(x, [a, b, c, d], [0, 1, 1, 0])


def _gauss(mu, sigma):
    return lambda x: np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def _sig(c, a):
    return lambda x: 1.0 / (1.0 + np.exp(-a * (x - c)))


def _invsig(c, a):
    return lambda x: 1.0 - 1.0 / (1.0 + np.exp(-a * (x - c)))


VARIANTS = {
    "narrow (triangles/trapezoid, width 1-3)": [
        ("small", Triangular_MF(a=2, b=3, c=4), _tri(2, 3, 4)),
        ("average", Triangular_MF(a=12, b=12.5, c=13), _tri(12, 12.5, 13)),
        ("generous", Trapezoidal_MF(a=21, b=22, c=23, d=24), _trap(21, 22, 23, 24)),
    ],
    "gaussian": [
        ("small", Gaussian_MF(mu=5, sigma=2), _gauss(5, 2)),
        ("average", Gaussian_MF(mu=12.5, sigma=2.5), _gauss(12.5, 2.5)),
        ("generous", Gaussian_MF(mu=20, sigma=2.5), _gauss(20, 2.5)),
    ],
    "sigmoid (inv. sigmoid / Gaussian / sigmoid)": [
        ("small", InvSigmoid_MF(c=6, a=1), _invsig(6, 1)),
        ("average", Gaussian_MF(mu=12.5, sigma=2.5), _gauss(12.5, 2.5)),
        ("generous", Sigmoid_MF(c=19, a=1), _sig(19, 1)),
    ],
}


def build(variant):
    FS = FuzzySystem(show_banner=False)
    FS.add_linguistic_variable("Service", LinguisticVariable(
        [FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor"),
         FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good"),
         FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")],
        universe_of_discourse=[0, 10]))
    FS.add_linguistic_variable("Food", LinguisticVariable(
        [FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid"),
         FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")],
        universe_of_discourse=[0, 10]))
    FS.add_linguistic_variable("Tip", LinguisticVariable(
        [FuzzySet(function=mf, term=term) for term, mf, _ in VARIANTS[variant]],
        universe_of_discourse=[0, 25]))
    FS.add_rules([
        "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)",
        "IF (Service IS good) THEN (Tip IS average)",
        "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)",
    ])
    return FS


X = np.linspace(0, 25, 1_000_001)


def reference(variant, service, food):
    poor = np.interp(service, [0, 5], [1, 0])
    good = np.interp(service, [0, 5, 10], [0, 1, 0])
    excellent = np.interp(service, [5, 10], [0, 1])
    rancid = np.interp(food, [0, 10], [1, 0])
    delicious = np.interp(food, [0, 10], [0, 1])
    strengths = [max(poor, rancid), good, max(excellent, delicious)]
    sets = [np.clip(f(X), 0, 1) for _, _, f in VARIANTS[variant]]
    agg = np.maximum.reduce([np.minimum(w, s) for w, s in zip(strengths, sets)])
    return integrate_trapezoid(X * agg, X) / integrate_trapezoid(agg, X)


def errors(FS, refs, infer):
    errs = []
    for (s, f), ref in refs.items():
        FS.set_variable("Service", s)
        FS.set_variable("Food", f)
        errs.append(abs(infer() - ref))
    return max(errs), float(np.mean(errs))


if __name__ == "__main__":
    print("=== limitation_A_harder.py\n")
    print("Tipping rules and inputs as Listing 1; output sets replaced. 121 inputs "
          "(Service, Food = 0..10); reference = 1,000,001-point trapezoidal rule.")
    print(f"Time at input (4, 8): mean ± sd over {RUNS} runs of 10 calls.\n")
    for variant in VARIANTS:
        FS = build(variant)
        refs = {(s, f): reference(variant, s, f) for s in GRID for f in GRID}
        print(f"--- {variant}")
        print("Method | Max error | Mean error | Time at (4, 8)")
        for n in SUBDIVISIONS:
            worst, mean = errors(FS, refs, lambda: FS.Mamdani_inference(
                ["Tip"], subdivisions=n, ignore_warnings=True)["Tip"])
            FS.set_variable("Service", 4); FS.set_variable("Food", 8)
            t = time_ms(lambda: FS.Mamdani_inference(["Tip"], subdivisions=n,
                                                     ignore_warnings=True), calls=10)
            print(f"Simpful {n} subdivisions | {worst:.2e} | {mean:.2e} | {fmt(*t[:2])}")
        worst, mean = errors(FS, refs, lambda: fast_mamdani(FS, ["Tip"])["Tip"])
        FS.set_variable("Service", 4); FS.set_variable("Food", 8)
        t = time_ms(lambda: fast_mamdani(FS, ["Tip"]), calls=10)
        label = "Ours (exact)" if variant.startswith("narrow") else "Ours (200,001-point grid)"
        print(f"{label} | {worst:.2e} | {mean:.2e} | {fmt(*t[:2])}\n")
    print("Note: for Gaussian/sigmoid sets 'Ours' is a fine grid, not exact; its error "
          "is bounded by the reference's own accuracy (~1e-9).")
