"""Correctness tests for fast_mamdani and fast_sugeno.

Run either way:
    python test_fast_mamdani.py      # prints a table, exit code 1 if any test fails
    pytest test_fast_mamdani.py      # standard test runner

What is tested
  T1  Grid mode equals Simpful: fast_mamdani(exact=False, subdivisions=n) must give
      the same answer as Simpful's Mamdani_inference(subdivisions=n) for n = 10, 100,
      1000. This supports the claim that only the inference step changed: with the
      exact centroid switched off, our code returns Simpful's own grid result.
  T2  Exact mode equals the independent high-resolution reference (tipping FIS).
  T3  Same as T1 and T2 on the second Mamdani system (fan control).
  T4  fast_sugeno equals Simpful's Sugeno_inference on the sepsis FIS.
  T6  Rule-weight fix: fast_sugeno(weights="average") returns true weighted
      averages on systems with an obvious answer, and equals Simpful exactly when
      every weight is 1 (the fix changes nothing for unweighted rules).
  T5  Active-rule lookup changes nothing: fast_mamdani(active=True) equals
      fast_mamdani(active=False) on random rule bases with 2, 4 and 6 inputs.
All random inputs use a fixed seed (20261111), so the test is repeatable.
"""
import sys

import numpy as np

from fast_mamdani import fast_mamdani
from fast_sugeno import fast_sugeno
from limitation_A_fixed import FS as TIPPING, reference_tip
from limitation_B_fixed import build_system
from temperature_control_mamdani import build_temperature_controller, high_resolution_reference
from benchmark_sepsis_sugeno import build_sepsis_system, set_inputs

SEED = 20261111
N_INPUTS = 100
GRID_TOL = 1e-9      # grid mode: same arithmetic as Simpful, only rounding may differ
EXACT_TOL = 1e-8     # exact mode vs a 1,000,001-point reference
SUBDIVISIONS = [10, 100, 1000]


def _max_grid_diff(FS, output, set_random_input, n_inputs=N_INPUTS):
    """Largest |ours(exact=False) - Simpful| over random inputs and all subdivisions."""
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(n_inputs):
        set_random_input(FS, rng)
        for n in SUBDIVISIONS:
            simpful = float(FS.Mamdani_inference([output], subdivisions=n,
                                                 ignore_warnings=True)[output])
            ours = float(fast_mamdani(FS, [output], exact=False, subdivisions=n)[output])
            worst = max(worst, abs(simpful - ours))
    return worst


def _tipping_input(FS, rng):
    FS.set_variable("Service", float(rng.uniform(0, 10)))
    FS.set_variable("Food", float(rng.uniform(0, 10)))


def _fan_input(FS, rng):
    FS.set_variable("Temperature", float(rng.uniform(0, 40)))
    FS.set_variable("Humidity", float(rng.uniform(0, 100)))


def _random_fis_input(n_inputs):
    def setter(FS, rng):
        for i in range(n_inputs):
            FS.set_variable(f"Input{i}", float(rng.uniform(0, 10)))
    return setter


# ---------------------------------------------------------------- tests

def test_grid_mode_equals_simpful_tipping():
    assert _max_grid_diff(TIPPING, "Tip", _tipping_input) < GRID_TOL


def test_grid_mode_equals_simpful_fan():
    assert _max_grid_diff(build_temperature_controller(), "FanSpeed", _fan_input) < GRID_TOL


def test_grid_mode_equals_simpful_random_rule_base():
    FS, _ = build_system(4, seed=1)            # 81 rules
    assert _max_grid_diff(FS, "Output", _random_fis_input(4), n_inputs=20) < GRID_TOL


def test_exact_mode_equals_reference_tipping():
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(N_INPUTS):
        s, f = rng.uniform(0, 10, size=2)
        TIPPING.set_variable("Service", float(s))
        TIPPING.set_variable("Food", float(f))
        worst = max(worst, abs(fast_mamdani(TIPPING, ["Tip"])["Tip"] - reference_tip(s, f)))
    assert worst < EXACT_TOL


def test_exact_mode_equals_reference_fan():
    FS = build_temperature_controller()
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(N_INPUTS):
        t, h = float(rng.uniform(0, 40)), float(rng.uniform(0, 100))
        FS.set_variable("Temperature", t)
        FS.set_variable("Humidity", h)
        worst = max(worst, abs(fast_mamdani(FS, ["FanSpeed"])["FanSpeed"]
                               - high_resolution_reference(t, h)))
    assert worst < EXACT_TOL


def test_fast_sugeno_equals_simpful_sepsis():
    FS = build_sepsis_system()
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for row in rng.uniform([0, -10, 0, 0], [80, 10, 100, 600], size=(1000, 4)):
        set_inputs(FS, dict(zip(["PaO2", "BaseExcess", "Trombocytes", "Creatinine"], map(float, row))))
        simpful = float(FS.Sugeno_inference(["Sepsis"], ignore_warnings=True)["Sepsis"])
        worst = max(worst, abs(simpful - fast_sugeno(FS, ["Sepsis"])["Sepsis"]))
    assert worst < GRID_TOL


def _max_active_diff():
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for n in (2, 4, 6):
        for seed in (1, 2, 3):
            FS, _ = build_system(n, seed)
            for _ in range(10):
                for i in range(n):
                    FS.set_variable(f"Input{i}", float(rng.uniform(0, 10)))
                on = fast_mamdani(FS, ["Output"], active=True)["Output"]
                off = fast_mamdani(FS, ["Output"], active=False)["Output"]
                worst = max(worst, abs(on - off))
    return worst


def _max_weight_fix_error():
    from findings_checks import weighted_sugeno_system
    cases = [  # (rules, expected weighted average); every rule fires at 1
        (["IF (X IS low) THEN (Y IS a) WEIGHT 0.5"], 10.0),
        (["IF (X IS low) THEN (Y IS a) WEIGHT 0.5",
          "IF (X IS low) THEN (Y IS b) WEIGHT 1.0"], (0.5 * 10 + 1.0 * 20) / 1.5),
        (["IF (X IS low) THEN (Y IS a) WEIGHT 0.2",
          "IF (X IS low) THEN (Y IS b) WEIGHT 0.6"], (0.2 * 10 + 0.6 * 20) / 0.8),
    ]
    worst = max(abs(fast_sugeno(weighted_sugeno_system(r), ["Y"], weights="average")["Y"] - e)
                for r, e in cases)
    sep = build_sepsis_system()          # unweighted: fix must equal Simpful
    rng = np.random.default_rng(SEED)
    for row in rng.uniform([0, -10, 0, 0], [80, 10, 100, 600], size=(200, 4)):
        set_inputs(sep, dict(zip(["PaO2", "BaseExcess", "Trombocytes", "Creatinine"], map(float, row))))
        simpful = float(sep.Sugeno_inference(["Sepsis"], ignore_warnings=True)["Sepsis"])
        worst = max(worst, abs(fast_sugeno(sep, ["Sepsis"], weights="average")["Sepsis"] - simpful))
    return worst


def test_weight_fix():
    assert _max_weight_fix_error() < GRID_TOL


def test_active_rules_change_nothing():
    assert _max_active_diff() < GRID_TOL


# ---------------------------------------------------------------- script mode

def _measure():
    """Same checks as the tests, but returning the numbers for the results file."""
    rows = []
    rows.append(("T1 grid mode = Simpful, tipping (100 inputs x n=10,100,1000)",
                 _max_grid_diff(TIPPING, "Tip", _tipping_input), GRID_TOL))
    rows.append(("T1 grid mode = Simpful, fan control (100 inputs x 3 n)",
                 _max_grid_diff(build_temperature_controller(), "FanSpeed", _fan_input), GRID_TOL))
    FS, _ = build_system(4, seed=1)
    rows.append(("T1 grid mode = Simpful, random 81-rule FIS (20 inputs x 3 n)",
                 _max_grid_diff(FS, "Output", _random_fis_input(4), n_inputs=20), GRID_TOL))

    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(N_INPUTS):
        s, f = rng.uniform(0, 10, size=2)
        TIPPING.set_variable("Service", float(s)); TIPPING.set_variable("Food", float(f))
        worst = max(worst, abs(fast_mamdani(TIPPING, ["Tip"])["Tip"] - reference_tip(s, f)))
    rows.append(("T2 exact mode = reference, tipping (100 inputs)", worst, EXACT_TOL))

    fan = build_temperature_controller()
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(N_INPUTS):
        t, h = float(rng.uniform(0, 40)), float(rng.uniform(0, 100))
        fan.set_variable("Temperature", t); fan.set_variable("Humidity", h)
        worst = max(worst, abs(fast_mamdani(fan, ["FanSpeed"])["FanSpeed"]
                               - high_resolution_reference(t, h)))
    rows.append(("T3 exact mode = reference, fan control (100 inputs)", worst, EXACT_TOL))

    sep = build_sepsis_system()
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for row in rng.uniform([0, -10, 0, 0], [80, 10, 100, 600], size=(1000, 4)):
        set_inputs(sep, dict(zip(["PaO2", "BaseExcess", "Trombocytes", "Creatinine"], map(float, row))))
        simpful = float(sep.Sugeno_inference(["Sepsis"], ignore_warnings=True)["Sepsis"])
        worst = max(worst, abs(simpful - fast_sugeno(sep, ["Sepsis"])["Sepsis"]))
    rows.append(("T4 fast_sugeno = Simpful, sepsis (1000 inputs)", worst, GRID_TOL))
    rows.append(("T5 active=True = active=False, random FIS 2/4/6 inputs (90 inputs)",
                 _max_active_diff(), GRID_TOL))
    rows.append(("T6 weights='average' = weighted average (3 cases) and = Simpful when unweighted (200 inputs)",
                 _max_weight_fix_error(), GRID_TOL))
    return rows


if __name__ == "__main__":
    print("=== test_fast_mamdani.py\n")
    print(f"Seed {SEED}. PASS if max |difference| < tolerance.\n")
    print("Test | Max |difference| | Tolerance | Result")
    print("-" * 100)
    failed = 0
    for name, worst, tol in _measure():
        ok = worst < tol
        failed += not ok
        print(f"{name} | {worst:.2e} | {tol:.0e} | {'PASS' if ok else 'FAIL'}")
    print(f"\n{'All tests passed.' if not failed else f'{failed} test(s) FAILED.'}")
    sys.exit(1 if failed else 0)
