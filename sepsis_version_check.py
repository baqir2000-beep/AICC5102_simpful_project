"""Run Listing 3 (sepsis) exactly as printed in the paper and print the Simpful
version, so the 69.3 (paper) vs 68.90 (Simpful 2.12.0) gap can be traced to a
version change.

Written to run on old Simpful releases too: no show_banner, no ignore_warnings,
no plotting. Use it through check_old_simpful.sh (see README, "Sepsis gap").
"""
try:
    from importlib.metadata import version
except ImportError:
    from pkg_resources import get_distribution

    def version(distribution):
        return get_distribution(distribution).version

from simpful import *

FS = FuzzySystem()
FS.add_linguistic_variable("PaO2", LinguisticVariable(
    [FuzzySet(function=Sigmoid_MF(c=40, a=0.1), term="low"),
     FuzzySet(function=InvSigmoid_MF(c=40, a=0.1), term="high")],
    concept="PaO2 level in blood", universe_of_discourse=[0, 80]))
FS.add_linguistic_variable("BaseExcess", LinguisticVariable(
    [FuzzySet(function=Gaussian_MF(mu=0, sigma=1.25), term="normal")],
    concept="Base excess of the blood", universe_of_discourse=[-10, 10]))
FS.add_linguistic_variable("Trombocytes", LinguisticVariable(
    [FuzzySet(function=Sigmoid_MF(c=50, a=0.75), term="low"),
     FuzzySet(function=InvSigmoid_MF(c=50, a=0.75), term="high")],
    concept="Trombocytes in blood", universe_of_discourse=[0, 100]))
FS.add_linguistic_variable("Creatinine", LinguisticVariable(
    [FuzzySet(function=Sigmoid_MF(c=300, a=0.2), term="low"),
     FuzzySet(function=InvSigmoid_MF(c=300, a=0.1), term="high")],
    concept="Creatinine in blood", universe_of_discourse=[0, 600]))
FS.set_crisp_output_value("low_probability", 1)
FS.set_crisp_output_value("high_probability", 99)
FS.add_rules([
    "IF (PaO2 IS low) AND (Trombocytes IS high) AND (Creatinine IS high) "
    "AND (BaseExcess IS normal) THEN (Sepsis IS low_probability)",
    "IF (PaO2 IS high) AND (Trombocytes IS low) AND (Creatinine IS low) "
    "AND (NOT(BaseExcess IS normal)) THEN (Sepsis IS high_probability)",
])
FS.set_variable("PaO2", 50)
FS.set_variable("BaseExcess", -1.5)
FS.set_variable("Trombocytes", 50)
FS.set_variable("Creatinine", 320)

result = float(FS.Sugeno_inference(["Sepsis"])["Sepsis"])
print(f"RESULT simpful {version('simpful')}: sepsis = {result:.4f} (paper: 69.3)")
