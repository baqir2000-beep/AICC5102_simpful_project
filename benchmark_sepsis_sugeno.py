"""Reproduce and benchmark the two-rule Sugeno sepsis example from Simpful.

This is a computational equivalence benchmark, not a clinical validation.  The
published example is illustrative and the sampled inputs must not be construed
as patient data or a diagnostic-performance study.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from simpful import (
    FuzzySet,
    FuzzySystem,
    Gaussian_MF,
    InvSigmoid_MF,
    LinguisticVariable,
    Sigmoid_MF,
)

from fast_sugeno import fast_sugeno
from common import time_ms, fmt, RUNS


OUTPUT_PATH = Path(__file__).with_name("results") / "sepsis_sugeno_results.txt"
PAPER_INPUT = {"PaO2": 50.0, "BaseExcess": -1.5, "Trombocytes": 50.0, "Creatinine": 320.0}


def build_sepsis_system() -> FuzzySystem:
    """Listing 3 from the Simpful paper, excluding plotting calls."""
    system = FuzzySystem(show_banner=False)
    system.add_linguistic_variable(
        "PaO2",
        LinguisticVariable(
            [
                FuzzySet(function=Sigmoid_MF(c=40, a=0.1), term="low"),
                FuzzySet(function=InvSigmoid_MF(c=40, a=0.1), term="high"),
            ],
            concept="PaO2 level in blood",
            universe_of_discourse=[0, 80],
        ),
    )
    system.add_linguistic_variable(
        "BaseExcess",
        LinguisticVariable(
            [FuzzySet(function=Gaussian_MF(mu=0, sigma=1.25), term="normal")],
            concept="Base excess of the blood",
            universe_of_discourse=[-10, 10],
        ),
    )
    system.add_linguistic_variable(
        "Trombocytes",
        LinguisticVariable(
            [
                FuzzySet(function=Sigmoid_MF(c=50, a=0.75), term="low"),
                FuzzySet(function=InvSigmoid_MF(c=50, a=0.75), term="high"),
            ],
            concept="Trombocytes in blood",
            universe_of_discourse=[0, 100],
        ),
    )
    system.add_linguistic_variable(
        "Creatinine",
        LinguisticVariable(
            [
                FuzzySet(function=Sigmoid_MF(c=300, a=0.2), term="low"),
                FuzzySet(function=InvSigmoid_MF(c=300, a=0.1), term="high"),
            ],
            concept="Creatinine",
            universe_of_discourse=[0, 600],
        ),
    )
    system.set_crisp_output_value("low_probability", 1)
    system.set_crisp_output_value("high_probability", 99)
    system.add_rules(
        [
            "IF (PaO2 IS low) AND (Trombocytes IS high) AND (Creatinine IS high) "
            "AND (BaseExcess IS normal) THEN (Sepsis IS low_probability)",
            "IF (PaO2 IS high) AND (Trombocytes IS low) AND (Creatinine IS low) "
            "AND (NOT(BaseExcess IS normal)) THEN (Sepsis IS high_probability)",
        ]
    )
    return system


def set_inputs(system: FuzzySystem, values: dict[str, float]) -> None:
    for name, value in values.items():
        system.set_variable(name, value)


def independent_reference(values: dict[str, float]) -> float:
    """Direct scalar translation of Listing 3's two zero-order Sugeno rules."""
    sigmoid = lambda x, c, a: 1.0 / (1.0 + np.exp(-a * (x - c)))
    pa_low = sigmoid(values["PaO2"], 40, 0.1)
    pa_high = 1.0 - pa_low
    tr_low = sigmoid(values["Trombocytes"], 50, 0.75)
    tr_high = 1.0 - tr_low
    cr_low = sigmoid(values["Creatinine"], 300, 0.2)
    cr_high = 1.0 - sigmoid(values["Creatinine"], 300, 0.1)
    base_normal = np.exp(-((values["BaseExcess"] - 0.0) ** 2) / (2 * 1.25**2))
    rule_1 = min(pa_low, tr_high, cr_high, base_normal)
    rule_2 = min(pa_high, tr_low, cr_low, 1.0 - base_normal)
    return 0.0 if rule_1 + rule_2 == 0.0 else (rule_1 + 99.0 * rule_2) / (rule_1 + rule_2)


def numerical_check(system: FuzzySystem) -> dict:
    rng = np.random.default_rng(20261106)
    samples = [PAPER_INPUT]
    for row in rng.uniform([0, -10, 0, 0], [80, 10, 100, 600], size=(10_000, 4)):
        samples.append(dict(zip(PAPER_INPUT, map(float, row))))
    baseline_error = []
    fast_error = []
    pair_error = []
    paper_result = None
    for values in samples:
        set_inputs(system, values)
        reference = independent_reference(values)
        baseline = float(system.Sugeno_inference(["Sepsis"], ignore_warnings=True)["Sepsis"])
        optimized = float(fast_sugeno(system, ["Sepsis"])["Sepsis"])
        baseline_error.append(abs(baseline - reference))
        fast_error.append(abs(optimized - reference))
        pair_error.append(abs(optimized - baseline))
        if values is PAPER_INPUT:
            paper_result = {"reference": reference, "simpful": baseline, "fast_sugeno": optimized}
    return {
        "test_inputs": len(samples),
        "random_seed": 20261106,
        "paper_input_result": paper_result,
        "max_abs_error_simpful_vs_reference": max(baseline_error),
        "max_abs_error_fast_vs_reference": max(fast_error),
        "max_abs_difference_fast_vs_simpful": max(pair_error),
    }


def format_results(check: dict, baseline: tuple, optimized: tuple) -> str:
    """Plain-text layout used by results/all_results.txt."""
    paper = check["paper_input_result"]
    speedup = baseline[0] / optimized[0]
    return (
        "=== benchmark_sepsis_sugeno.py\n\n"
        "Model | Test inputs | Simpful result | Ours result | Max |difference|\n"
        "--------------------------------------------------------------------------------\n"
        f"Sepsis Sugeno | {check['test_inputs']} | {paper['simpful']:.6f} | "
        f"{paper['fast_sugeno']:.6f} | {check['max_abs_difference_fast_vs_simpful']:.2e}\n\n"
        f"Time = mean ± sd over {RUNS} runs of 2000 calls (common.time_ms)\n"
        "Method | Time | Speed-up\n"
        "----------------------------------------\n"
        f"Simpful Sugeno | {fmt(*baseline, decimals=4)} | -\n"
        f"Ours (fast_sugeno) | {fmt(*optimized, decimals=4)} | {speedup:.1f}x\n\n"
        "Correctness: independent analytic reference; max error was "
        f"{check['max_abs_error_fast_vs_reference']:.2e}.\n"
        "Paper reports 69.3%; Simpful 2.12.0 and ours both give the value above.\n"
        "Scope: computational equivalence only, not clinical validation.\n"
    )


def main() -> None:
    system = build_sepsis_system()
    check = numerical_check(system)
    set_inputs(system, PAPER_INPUT)
    baseline = time_ms(lambda: system.Sugeno_inference(["Sepsis"], ignore_warnings=True), calls=2000)[:2]
    optimized = time_ms(lambda: fast_sugeno(system, ["Sepsis"]), calls=2000)[:2]
    results_text = format_results(check, baseline, optimized)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(results_text, encoding="utf-8")
    print(results_text, end="")


if __name__ == "__main__":
    main()
