"""Independent Mamdani benchmark: temperature and humidity fan control.

This is an interpretable second Mamdani system for testing ``fast_mamdani``.
It is deliberately distinct from the tipping example: it has two environmental
inputs, nine rules, and a fan-speed output.  It is a computational benchmark,
not a physical HVAC controller specification.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from simpful import FuzzySet, FuzzySystem, LinguisticVariable, Trapezoidal_MF, Triangular_MF

from fast_mamdani import fast_mamdani
from common import time_ms, fmt, RUNS, integrate_trapezoid


OUTPUT_PATH = Path(__file__).with_name("results") / "temperature_control_results.txt"


def build_temperature_controller() -> FuzzySystem:
    """Create a 2-input, 9-rule max-min Mamdani fan controller."""
    system = FuzzySystem(show_banner=False)

    system.add_linguistic_variable(
        "Temperature",
        LinguisticVariable(
            [
                FuzzySet(function=Triangular_MF(a=0, b=0, c=22), term="cool"),
                FuzzySet(function=Triangular_MF(a=18, b=25, c=32), term="comfortable"),
                FuzzySet(function=Triangular_MF(a=28, b=40, c=40), term="hot"),
            ],
            universe_of_discourse=[0, 40],
        ),
    )
    system.add_linguistic_variable(
        "Humidity",
        LinguisticVariable(
            [
                FuzzySet(function=Triangular_MF(a=0, b=0, c=55), term="dry"),
                FuzzySet(function=Trapezoidal_MF(a=35, b=45, c=55, d=65), term="normal"),
                FuzzySet(function=Triangular_MF(a=45, b=100, c=100), term="humid"),
            ],
            universe_of_discourse=[0, 100],
        ),
    )
    system.add_linguistic_variable(
        "FanSpeed",
        LinguisticVariable(
            [
                FuzzySet(function=Triangular_MF(a=0, b=0, c=50), term="low"),
                FuzzySet(function=Triangular_MF(a=25, b=50, c=75), term="medium"),
                FuzzySet(function=Trapezoidal_MF(a=50, b=75, c=100, d=100), term="high"),
            ],
            universe_of_discourse=[0, 100],
        ),
    )
    system.add_rules(
        [
            "IF (Temperature IS cool) AND (Humidity IS dry) THEN (FanSpeed IS low)",
            "IF (Temperature IS cool) AND (Humidity IS normal) THEN (FanSpeed IS low)",
            "IF (Temperature IS cool) AND (Humidity IS humid) THEN (FanSpeed IS medium)",
            "IF (Temperature IS comfortable) AND (Humidity IS dry) THEN (FanSpeed IS low)",
            "IF (Temperature IS comfortable) AND (Humidity IS normal) THEN (FanSpeed IS medium)",
            "IF (Temperature IS comfortable) AND (Humidity IS humid) THEN (FanSpeed IS high)",
            "IF (Temperature IS hot) AND (Humidity IS dry) THEN (FanSpeed IS medium)",
            "IF (Temperature IS hot) AND (Humidity IS normal) THEN (FanSpeed IS high)",
            "IF (Temperature IS hot) AND (Humidity IS humid) THEN (FanSpeed IS high)",
        ]
    )
    return system


def triangle(x, a, b, c):
    """Membership function matching Simpful's triangular functions, vectorised."""
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else (x - a) / (b - a)
    right = np.ones_like(x) if b == c else (c - x) / (c - b)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trapezoid(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    rise = np.ones_like(x) if a == b else (x - a) / (b - a)
    fall = np.ones_like(x) if c == d else (d - x) / (d - c)
    return np.clip(np.minimum(rise, fall), 0.0, 1.0)


def high_resolution_reference(temperature: float, humidity: float, points=1_000_001) -> float:
    """Independent, dense-grid max-min centroid reference for this rule base."""
    cool = float(triangle(temperature, 0, 0, 22))
    comfortable = float(triangle(temperature, 18, 25, 32))
    hot = float(triangle(temperature, 28, 40, 40))
    dry = float(triangle(humidity, 0, 0, 55))
    normal = float(trapezoid(humidity, 35, 45, 55, 65))
    humid = float(triangle(humidity, 45, 100, 100))

    # Merge the nine rules by output term under max aggregation.
    low_strength = max(min(cool, dry), min(cool, normal), min(comfortable, dry))
    medium_strength = max(min(cool, humid), min(comfortable, normal), min(hot, dry))
    high_strength = max(min(comfortable, humid), min(hot, normal), min(hot, humid))

    x = np.linspace(0, 100, points)
    low = np.minimum(low_strength, triangle(x, 0, 0, 50))
    medium = np.minimum(medium_strength, triangle(x, 25, 50, 75))
    high = np.minimum(high_strength, trapezoid(x, 50, 75, 100, 100))
    aggregate = np.maximum.reduce([low, medium, high])
    return float(integrate_trapezoid(x * aggregate, x) / integrate_trapezoid(aggregate, x))


def set_inputs(system: FuzzySystem, temperature: float, humidity: float) -> None:
    system.set_variable("Temperature", temperature)
    system.set_variable("Humidity", humidity)


def accuracy_check(system: FuzzySystem) -> dict:
    # A fixed grid is reproducible and covers all input-term overlap regions.
    temperatures = [2.0, 15.0, 20.0, 23.0, 25.0, 29.0, 35.0, 39.0]
    humidities = [2.0, 30.0, 42.0, 50.0, 60.0, 80.0, 98.0]
    baseline_errors, fast_errors, differences = [], [], []
    representative = None
    for temperature in temperatures:
        for humidity in humidities:
            set_inputs(system, temperature, humidity)
            reference = high_resolution_reference(temperature, humidity)
            baseline = float(system.Mamdani_inference(["FanSpeed"], subdivisions=1000)["FanSpeed"])
            optimized = float(fast_mamdani(system, ["FanSpeed"])["FanSpeed"])
            baseline_errors.append(abs(baseline - reference))
            fast_errors.append(abs(optimized - reference))
            differences.append(abs(optimized - baseline))
            if (temperature, humidity) == (29.0, 60.0):   # asymmetric: methods differ here
                representative = {"temperature": temperature, "humidity": humidity,
                                  "reference": reference, "simpful": baseline, "fast_mamdani": optimized}
    return {
        "grid_points": len(temperatures) * len(humidities),
        "reference_method": "independent 1,000,001-point trapezoidal integration",
        "representative_input": representative,
        "max_abs_error_simpful_1000_vs_reference": max(baseline_errors),
        "max_abs_error_fast_vs_reference": max(fast_errors),
        "max_abs_difference_fast_vs_simpful_1000": max(differences),
    }


def format_results(accuracy: dict, baseline: tuple, optimized: tuple) -> str:
    """Plain-text layout used by results/all_results.txt."""
    representative = accuracy["representative_input"]
    speedup = baseline[0] / optimized[0]
    return (
        "=== temperature_control_mamdani.py\n\n"
        f"Time = mean ± sd over {RUNS} runs of 20 calls (common.time_ms)\n"
        "Inputs | Rules | Simpful time | Ours time | Speed-up | Simpful max |error| | Ours max |error|\n"
        "----------------------------------------------------------------------------------------------------\n"
        f"Temperature + Humidity | 9 | {fmt(*baseline)} | {fmt(*optimized)} | {speedup:.1f}x | "
        f"{accuracy['max_abs_error_simpful_1000_vs_reference']:.6f} | "
        f"{accuracy['max_abs_error_fast_vs_reference']:.2e}\n\n"
        "Representative input\n"
        f"Temperature = {representative['temperature']:.0f}, Humidity = {representative['humidity']:.0f}\n"
        f"High-resolution reference: {representative['reference']:.6f}\n"
        f"Simpful (1000 subdivisions): {representative['simpful']:.6f}\n"
        f"Ours (exact): {representative['fast_mamdani']:.6f}\n\n"
        "Reference: independent 1,000,001-point centroid integration over 56 input combinations.\n"
    )


def main() -> None:
    system = build_temperature_controller()
    accuracy = accuracy_check(system)
    set_inputs(system, 29.0, 60.0)
    baseline = time_ms(lambda: system.Mamdani_inference(["FanSpeed"], subdivisions=1000), calls=20)[:2]
    optimized = time_ms(lambda: fast_mamdani(system, ["FanSpeed"]), calls=20)[:2]
    results_text = format_results(accuracy, baseline, optimized)
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    OUTPUT_PATH.write_text(results_text, encoding="utf-8")
    print(results_text, end="")


if __name__ == "__main__":
    main()
