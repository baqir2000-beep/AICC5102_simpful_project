"""Fig. 4 of the paper: Simpful vs Scikit-Fuzzy output surface (tipping, Mamdani).

The paper only shows the two surfaces side by side. Here we measure:
  1. the difference between the two libraries on a 21 x 21 grid, and
  2. each library's error against the same high-resolution reference
     (limitation_A_fixed.reference_tip), plus our exact method.
Part 2 shows WHICH library the difference comes from.
"""
import os

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from simpful import *

from fast_mamdani import fast_mamdani
from limitation_A_fixed import reference_tip


# ============================================================
# SIMPFUL MODEL
# ============================================================

FS = FuzzySystem(show_banner=False)

# Service
S1 = FuzzySet(points=[[0., 1.], [5., 0.]], term="poor")
S2 = FuzzySet(points=[[0., 0.], [5., 1.], [10., 0.]], term="good")
S3 = FuzzySet(points=[[5., 0.], [10., 1.]], term="excellent")

FS.add_linguistic_variable(
    "Service",
    LinguisticVariable(
        [S1, S2, S3],
        universe_of_discourse=[0, 10]
    )
)

# Food
F1 = FuzzySet(points=[[0., 1.], [10., 0.]], term="rancid")
F2 = FuzzySet(points=[[0., 0.], [10., 1.]], term="delicious")

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


# ============================================================
# SCIKIT-FUZZY MODEL
# ============================================================

service = ctrl.Antecedent(np.linspace(0, 10, 1001), "Service")
food = ctrl.Antecedent(np.linspace(0, 10, 1001), "Food")
tip = ctrl.Consequent(np.linspace(0, 25, 1001), "Tip")

service["poor"] = fuzz.trimf(service.universe, [0, 0, 5])
service["good"] = fuzz.trimf(service.universe, [0, 5, 10])
service["excellent"] = fuzz.trimf(service.universe, [5, 10, 10])

food["rancid"] = fuzz.trimf(food.universe, [0, 0, 10])
food["delicious"] = fuzz.trimf(food.universe, [0, 10, 10])

tip["small"] = fuzz.trimf(tip.universe, [0, 0, 10])
tip["average"] = fuzz.trimf(tip.universe, [0, 10, 20])
tip["generous"] = fuzz.trapmf(tip.universe, [10, 20, 25, 25])

r1 = ctrl.Rule(
    service["poor"] | food["rancid"],
    tip["small"]
)

r2 = ctrl.Rule(
    service["good"],
    tip["average"]
)

r3 = ctrl.Rule(
    service["excellent"] | food["delicious"],
    tip["generous"]
)

control_system = ctrl.ControlSystem([r1, r2, r3])


# ============================================================
# 21 x 21 COMPARISON
# ============================================================

values = np.linspace(0, 10, 21)

simpful_outputs = []
skfuzzy_outputs = []
ours_outputs = []
reference_outputs = []

for s in values:
    simpful_row = []
    skfuzzy_row = []

    for f in values:

        # Simpful
        FS.set_variable("Service", s)
        FS.set_variable("Food", f)

        result = FS.Mamdani_inference(
            ["Tip"],
            subdivisions=1000
        )

        simpful_value = result["Tip"]
        simpful_row.append(simpful_value)
        ours_outputs.append(fast_mamdani(FS, ["Tip"])["Tip"])
        reference_outputs.append(reference_tip(s, f))

        # Scikit-Fuzzy
        simulation = ctrl.ControlSystemSimulation(control_system)

        simulation.input["Service"] = s
        simulation.input["Food"] = f

        simulation.compute()

        skfuzzy_value = simulation.output["Tip"]
        skfuzzy_row.append(skfuzzy_value)

    simpful_outputs.append(simpful_row)
    skfuzzy_outputs.append(skfuzzy_row)


simpful_outputs = np.array(simpful_outputs)
skfuzzy_outputs = np.array(skfuzzy_outputs)

difference = np.abs(simpful_outputs - skfuzzy_outputs)
reference = np.array(reference_outputs).reshape(simpful_outputs.shape)
ours = np.array(ours_outputs).reshape(simpful_outputs.shape)

print("=== figure4_test.py\n")
print("Part 1 - Simpful vs Scikit-Fuzzy, 21 x 21 grid (441 inputs)")
print(f"Maximum difference: {np.max(difference):.6f}")
print(f"Mean difference: {np.mean(difference):.6f}")
print()
print("Part 2 - each method vs the high-resolution reference (same 441 inputs)")
print("Method | Max error | Mean error")
for name, values in [("Simpful (1000 subdivisions)", simpful_outputs),
                     ("Scikit-Fuzzy (1001-point universe)", skfuzzy_outputs),
                     ("Ours (exact)", ours)]:
    err = np.abs(values - reference)
    print(f"{name} | {np.max(err):.2e} | {np.mean(err):.2e}")
print("If Scikit-Fuzzy's error is much smaller than Simpful's, the Part 1 difference")
print("comes from Simpful's grid, not from Scikit-Fuzzy.")

os.makedirs("results", exist_ok=True)
np.savetxt(
    "results/figure4_difference.csv",
    difference,
    delimiter=","
)

print("Saved results/figure4_difference.csv")