"""Two correctness checks on Simpful 2.12.0 that the paper does not discuss.

F1  Rule weights in Sugeno inference.
    A weighted Sugeno output should be a weighted average:
        y = sum(w_i * f_i * z_i) / sum(w_i * f_i)
    (f_i = firing strength, w_i = rule weight, z_i = crisp output).
    Reading Simpful 2.12's code suggests the weight multiplies the numerator
    only. Then a single rule with weight 0.5 would return HALF its crisp value,
    which is not a weighted average. We test this on tiny systems where the
    correct answer is obvious.

F2  Labels of the sepsis membership functions (Listing 3).
    Sigmoid_MF increases with x. The paper uses Sigmoid_MF for the terms
    called "low" and InvSigmoid_MF for "high". If that is right, "low" should
    be close to 1 for LOW values. We print both terms at a clearly low and a
    clearly high value of each variable. We also check that the two creatinine
    terms add up to 1 (they use different slopes, 0.2 and 0.1).

Nothing here is a clinical statement; F2 only checks that each label matches
the shape of its function.
"""
import numpy as np
from simpful import (FuzzySet, FuzzySystem, LinguisticVariable,
                     Sigmoid_MF, InvSigmoid_MF, Triangular_MF)

from fast_mamdani import fast_mamdani
from fast_sugeno import fast_sugeno


# ============================================================ F1: rule weights

def weighted_sugeno_system(rules):
    FS = FuzzySystem(show_banner=False)
    FS.add_linguistic_variable("X", LinguisticVariable(
        [FuzzySet(points=[[0., 1.], [10., 0.]], term="low")],
        universe_of_discourse=[0, 10]))
    FS.set_crisp_output_value("a", 10)
    FS.set_crisp_output_value("b", 20)
    FS.add_rules(rules)
    FS.set_variable("X", 0)          # membership of "low" = 1, so every rule fires at 1
    return FS


def check_weights_sugeno():
    cases = [
        ("One rule, weight 0.5",
         ["IF (X IS low) THEN (Y IS a) WEIGHT 0.5"],
         [(0.5, 10)]),
        ("One rule, weight 1.0 (control)",
         ["IF (X IS low) THEN (Y IS a) WEIGHT 1.0"],
         [(1.0, 10)]),
        ("Two rules, weights 0.5 and 1.0",
         ["IF (X IS low) THEN (Y IS a) WEIGHT 0.5",
          "IF (X IS low) THEN (Y IS b) WEIGHT 1.0"],
         [(0.5, 10), (1.0, 20)]),
    ]
    print("F1 - Sugeno rule weights (input X = 0, so every rule fires with strength 1)")
    print("Case | Simpful | Weighted average | Numerator-only weighting | Simpful matches | "
          "Fixed (fast_sugeno weights='average')")
    print("-" * 100)
    for name, rules, wz in cases:
        weighted_avg = sum(w * z for w, z in wz) / sum(w for w, _ in wz)
        numerator_only = sum(w * z for w, z in wz) / len(wz)
        try:
            FS = weighted_sugeno_system(rules)
            value = float(FS.Sugeno_inference(["Y"], ignore_warnings=True)["Y"])
            fixed = float(fast_sugeno(FS, ["Y"], weights="average")["Y"])
        except Exception as e:                     # e.g. if WEIGHT syntax is not accepted
            print(f"{name} | ERROR {type(e).__name__}: {e}")
            continue
        if abs(value - weighted_avg) < 1e-9:
            verdict = "weighted average"
        elif abs(value - numerator_only) < 1e-9:
            verdict = "numerator-only"
        else:
            verdict = "neither"
        ok = "OK" if abs(fixed - weighted_avg) < 1e-9 else "WRONG"
        print(f"{name} | {value:.4f} | {weighted_avg:.4f} | {numerator_only:.4f} | {verdict} | "
              f"{fixed:.4f} ({ok})")
    print("If the weight-0.5 rows say 'numerator-only', Simpful's weighted Sugeno output")
    print("is not a weighted average: a single rule with weight 0.5 halves its output.")
    print("The last column is our fix: the weight scales the firing strength in both the")
    print("numerator and the denominator, as Simpful's own Mamdani inference does (F1b).")


def check_weights_mamdani():
    """Mamdani: the weight should scale the firing strength (clip level)."""
    def build(w):
        FS = FuzzySystem(show_banner=False)
        FS.add_linguistic_variable("X", LinguisticVariable(
            [FuzzySet(points=[[0., 1.], [10., 0.]], term="low")], universe_of_discourse=[0, 10]))
        FS.add_linguistic_variable("Y", LinguisticVariable(
            [FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")],
            universe_of_discourse=[0, 10]))
        FS.add_rules([f"IF (X IS low) THEN (Y IS small) WEIGHT {w}"])
        FS.set_variable("X", 0)
        return FS

    print("\nF1b - Mamdani rule weights (single rule, X = 0)")
    print("Weight | Simpful (1000 subdivisions) | Ours (exact)")
    print("-" * 60)
    for w in (1.0, 0.5):
        try:
            FS = build(w)
            simpful = float(FS.Mamdani_inference(["Y"], ignore_warnings=True)["Y"])
            ours = float(fast_mamdani(FS, ["Y"])["Y"])
            print(f"{w} | {simpful:.6f} | {ours:.6f}")
        except Exception as e:
            print(f"{w} | ERROR {type(e).__name__}: {e}")
    print("A single clipped triangle (0,0,10) has centroid 10/3 = 3.3333 at weight 1.")
    print("Lower clip levels move the centroid right; Simpful and ours should agree.")


# ============================================================ F2: sepsis labels

def check_sepsis_labels():
    variables = [
        # name, low-term MF, high-term MF, clearly low value, clearly high value
        ("PaO2", Sigmoid_MF(c=40, a=0.1), InvSigmoid_MF(c=40, a=0.1), 10, 70),
        ("Trombocytes", Sigmoid_MF(c=50, a=0.75), InvSigmoid_MF(c=50, a=0.75), 20, 80),
        ("Creatinine", Sigmoid_MF(c=300, a=0.2), InvSigmoid_MF(c=300, a=0.1), 100, 500),
    ]
    print("\nF2 - Sepsis membership labels (Listing 3)")
    print("Variable | value | 'low' membership | 'high' membership")
    print("-" * 70)
    for name, low_mf, high_mf, lo, hi in variables:
        for x in (lo, hi):
            print(f"{name} | {x} | {float(low_mf._execute(x)):.3f} | {float(high_mf._execute(x)):.3f}")
    print("If 'low' is near 0 at the low value and near 1 at the high value, the labels")
    print("are inverted relative to the shapes (Sigmoid_MF increases with x).")

    x = np.linspace(0, 600, 601)
    low = np.array([float(Sigmoid_MF(c=300, a=0.2)._execute(v)) for v in x])
    high = np.array([float(InvSigmoid_MF(c=300, a=0.1)._execute(v)) for v in x])
    gap = np.abs(low + high - 1)
    print(f"\nCreatinine: max |low + high - 1| over 0..600 = {gap.max():.3f} "
          f"(at {x[gap.argmax()]:.0f}); the two terms do not form a partition.")


if __name__ == "__main__":
    print("=== findings_checks.py\n")
    check_weights_sugeno()
    check_weights_mamdani()
    check_sepsis_labels()
