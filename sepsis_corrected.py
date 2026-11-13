"""Fix for finding F2: the sepsis model of Listing 3 with labels that match their shapes.

In Listing 3, every term called "low" uses Sigmoid_MF, which INCREASES with x, and
every "high" uses InvSigmoid_MF, which decreases (findings_checks.py, F2). The two
creatinine terms also have different slopes (0.2 and 0.1), so they do not add up to 1.

We compare three versions of the model; the rules, consequents and inputs stay
exactly as printed in the paper:
  original           : Listing 3 as printed (reproduction)
  labels fixed       : "low" = InvSigmoid_MF, "high" = Sigmoid_MF (same centres and
                       slopes), so "low" is close to 1 for low values
  labels + partition : as "labels fixed", and both creatinine terms use slope 0.1,
                       so low + high = 1 everywhere

Reported: the output at the paper's input, how far each version moves the output over
10,000 random inputs (same seed as benchmark_sepsis_sugeno.py), how often it moves the
output to the other side of 50%, and a label check like F2.

Scope: this fixes the labels so they mean what they say. It is NOT a clinical
validation; with the labels fixed, whether the paper's two rules make clinical sense
is a question for a clinician, and we do not claim either version is a better
diagnostic model.
"""
import numpy as np
from simpful import (FuzzySet, FuzzySystem, LinguisticVariable, Gaussian_MF,
                     Sigmoid_MF, InvSigmoid_MF)

from benchmark_sepsis_sugeno import PAPER_INPUT, set_inputs

RULES = [
    "IF (PaO2 IS low) AND (Trombocytes IS high) AND (Creatinine IS high) "
    "AND (BaseExcess IS normal) THEN (Sepsis IS low_probability)",
    "IF (PaO2 IS high) AND (Trombocytes IS low) AND (Creatinine IS low) "
    "AND (NOT(BaseExcess IS normal)) THEN (Sepsis IS high_probability)",
]
VERSIONS = ["original", "labels fixed", "labels + partition"]
# (centre, slope of "low", slope of "high"); universes only matter for plotting
VARIABLES = {"PaO2": (40, 0.1, 0.1, [0, 80]),
             "Trombocytes": (50, 0.75, 0.75, [0, 100]),
             "Creatinine": (300, 0.2, 0.1, [0, 600])}


def build(version):
    FS = FuzzySystem(show_banner=False)
    for name, (c, a_low, a_high, universe) in VARIABLES.items():
        if version == "labels + partition" and name == "Creatinine":
            a_low = a_high = 0.1
        if version == "original":
            low, high = Sigmoid_MF(c=c, a=a_low), InvSigmoid_MF(c=c, a=a_high)
        else:
            low, high = InvSigmoid_MF(c=c, a=a_low), Sigmoid_MF(c=c, a=a_high)
        FS.add_linguistic_variable(name, LinguisticVariable(
            [FuzzySet(function=low, term="low"), FuzzySet(function=high, term="high")],
            universe_of_discourse=universe))
    FS.add_linguistic_variable("BaseExcess", LinguisticVariable(
        [FuzzySet(function=Gaussian_MF(mu=0, sigma=1.25), term="normal")],
        universe_of_discourse=[-10, 10]))
    FS.set_crisp_output_value("low_probability", 1)
    FS.set_crisp_output_value("high_probability", 99)
    FS.add_rules(RULES)
    return FS


def infer(FS, values):
    set_inputs(FS, values)
    return float(FS.Sugeno_inference(["Sepsis"], ignore_warnings=True)["Sepsis"])


def label_check(FS):
    """'low' membership at a clearly low and a clearly high value of each variable."""
    rows = []
    for name, lo, hi in [("PaO2", 10, 70), ("Trombocytes", 20, 80), ("Creatinine", 100, 500)]:
        lv = FS._lvs[name]
        rows.append((name, float(lv.get_values(lo)["low"]), float(lv.get_values(hi)["low"])))
    x = np.linspace(0, 600, 601)
    gap = max(abs(sum(FS._lvs["Creatinine"].get_values(v).values()) - 1) for v in x)
    return rows, gap


if __name__ == "__main__":
    print("=== sepsis_corrected.py\n")
    systems = {v: build(v) for v in VERSIONS}

    print("Label check: 'low' membership at a clearly LOW / clearly HIGH value "
          "(correct labels: near 1 / near 0)")
    print("Version | PaO2 10 / 70 | Trombocytes 20 / 80 | Creatinine 100 / 500 | "
          "Creatinine max |low + high - 1|")
    print("-" * 110)
    for v, FS in systems.items():
        rows, gap = label_check(FS)
        cells = " | ".join(f"{a:.3f} / {b:.3f}" for _, a, b in rows)
        print(f"{v} | {cells} | {gap:.3f}")

    print("\nOutput at the paper's input (PaO2 50, BaseExcess -1.5, Trombocytes 50, "
          "Creatinine 320); paper reports 69.3")
    for v, FS in systems.items():
        print(f"{v} | {infer(FS, PAPER_INPUT):.2f}%")

    rng = np.random.default_rng(20261106)                 # as benchmark_sepsis_sugeno.py
    samples = [dict(zip(PAPER_INPUT, map(float, row)))
               for row in rng.uniform([0, -10, 0, 0], [80, 10, 100, 600], size=(10_000, 4))]
    base = np.array([infer(systems["original"], s) for s in samples])
    print("\nChange vs the original over 10,000 random inputs")
    print("Version | Mean |change| | Max |change| | Crosses 50% (other side)")
    print("-" * 80)
    for v in VERSIONS[1:]:
        out = np.array([infer(systems[v], s) for s in samples])
        change = np.abs(out - base)
        flips = np.mean((out >= 50) != (base >= 50))
        print(f"{v} | {change.mean():.2f} | {change.max():.2f} | {flips:.1%}")
    print("\nScope: labels fixed so they match their shapes; not a clinical validation.")
