"""Compare Gaussian Mamdani centroid algorithms on identical rules and inputs."""

import csv
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from simpful import FuzzySet, FuzzySystem, GaussianFuzzySet, LinguisticVariable

from common import fmt, time_ms
from fast_mamdani import fast_mamdani

LOWER, UPPER = 0.0, 25.0
RESULTS = Path(__file__).resolve().parent / "results"
METHODS = {
    "Simpful 1,000": lambda system: simpful_output(system, 1000),
    "Simpful 10,000": lambda system: simpful_output(system, 10000),
    "Dense grid": lambda system: output(system, analytic_gaussian=False),
    "Analytic": lambda system: output(system),
    "Without crossings": lambda system: output(system, gaussian_crossings=False),
}


def build_system(bells):
    system = FuzzySystem(show_banner=False)
    system.add_linguistic_variable("Tip", LinguisticVariable(
        [GaussianFuzzySet(mu, sigma, f"bell{i}") for i, (mu, sigma) in enumerate(bells)],
        universe_of_discourse=[LOWER, UPPER]))
    for i, _ in enumerate(bells):
        system.add_linguistic_variable(f"Strength{i}", LinguisticVariable(
            [FuzzySet(points=[[0, 0], [1, 1]], term="active")], universe_of_discourse=[0, 1]))
        system.add_rules([f"IF (Strength{i} IS active) THEN (Tip IS bell{i})"])
    return system


def set_strengths(system, strengths):
    for i, strength in enumerate(strengths):
        system.set_variable(f"Strength{i}", float(strength))


def simpful_output(system, subdivisions):
    return float(system.Mamdani_inference(["Tip"], subdivisions=subdivisions,
                                          ignore_warnings=True)["Tip"])


def output(system, **kwargs):
    return fast_mamdani(system, ["Tip"], **kwargs)["Tip"]


def reference(terms, lower=LOWER, upper=UPPER):
    """Independent adaptive quadrature with peak hints and no crossing solver."""
    active = [(mu, sigma, level) for mu, sigma, level in terms if level > 0]
    hints = {mu + k * sigma for mu, sigma, _ in active for k in (-8, -3, -1, 0, 1, 3, 8)}
    for mu, sigma, level in active:
        radius = sigma * math.sqrt(-2 * math.log(min(level, 1.0)))
        hints.update((mu - radius, mu + radius))
    points = sorted(x for x in hints if lower < x < upper)

    def envelope(x):
        return max((min(level, math.exp(-0.5 * ((x - mu) / sigma) ** 2))
                    for mu, sigma, level in active), default=0.0)

    # Subdivide further with uniform checkpoints. QUADPACK can miss very small
    # changes in which curve is highest when both sides look locally smooth.
    points = sorted(set(points).union(np.linspace(lower, upper, 257)[1:-1]))
    options = dict(points=points, epsabs=2e-11, epsrel=2e-12, limit=1200)
    area = quad(envelope, lower, upper, **options)[0]
    center = (lower + upper) / 2
    moment = quad(lambda x: (x - center) * envelope(x), lower, upper, **options)[0]
    return center + moment / area if area else 0.0


def scenarios():
    for service in range(11):
        for food in range(11):
            strengths = (max(1 - service / 5, 1 - food / 10),
                         1 - abs(service - 5) / 5,
                         max((service - 5) / 5, food / 10))
            yield "Gaussian tipping", f"{service},{food}", ((5, 2), (12.5, 2.5), (20, 2.5)), strengths

    step = (UPPER - LOWER) / 999
    for phase in np.linspace(0, 1, 21):
        mu = (280 + phase) * step
        yield "Narrow peaks", f"{phase:.2f}", ((mu, 0.003), (18, 0.008)), (0.85, 0.55)

    for spacing in (0.8, 2.4, 4.0):
        for level in (0.55, 0.8, 0.98, 1.0):
            yield "Overlapping bells", f"{spacing},{level}", ((9, 1.7), (9 + spacing, 3.0)), (level, 0.95)


def measure():
    groups = {}
    for group, case, bells, strengths in scenarios():
        groups.setdefault(group, []).append((case, bells, strengths))
    results, samples = [], []
    for group, cases in groups.items():
        systems = {}
        prepared = []
        for case, bells, strengths in cases:
            if bells not in systems:
                systems[bells] = build_system(bells)
            expected = reference([(mu, sigma, level) for (mu, sigma), level in zip(bells, strengths)])
            prepared.append((case, systems[bells], strengths, expected))
        for name, infer in METHODS.items():
            errors = []
            for case, system, strengths, expected in prepared:
                set_strengths(system, strengths)
                actual = infer(system)
                error = abs(actual - expected)
                errors.append(error)
                samples.append(dict(group=group, case=case, method=name, reference=expected,
                                    centroid=actual, absolute_error=error))

            def evaluate_batch():
                for _, system, strengths, _ in prepared:
                    set_strengths(system, strengths)
                    infer(system)

            mean, sd, _ = time_ms(evaluate_batch, calls=1)
            summary = dict(group=group, method=name, cases=len(prepared), max_error=max(errors),
                           mean_error=float(np.mean(errors)), ms=mean / len(prepared),
                           sd_ms=sd / len(prepared))
            results.append(summary)
            print(f"{group} | {name} | {len(prepared)} | {max(errors):.3e} | "
                  f"{summary['mean_error']:.3e} | {fmt(summary['ms'], summary['sd_ms'])}", flush=True)
    return results, samples


def save_csv(name, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(results, samples):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fbfaf7", "axes.facecolor": "#fbfaf7"})
    palette = dict(zip(METHODS, ("#a64a26", "#83653c", "#4672a3", "#00766c", "#825392")))
    markers = dict(zip(METHODS, ("o", "s", "^", "D", "X")))
    groups = list(dict.fromkeys(row["group"] for row in results))
    fig, axes = plt.subplots(1, len(groups), figsize=(12, 3.95), sharex=True, sharey=True,
                             layout="constrained")
    for ax, group in zip(axes, groups):
        for row in results:
            if row["group"] == group:
                name = row["method"]
                ax.scatter(row["ms"], max(row["max_error"], 1e-14), color=palette[name],
                           marker=markers[name], s=63, label=name)
        ax.set(xscale="log", yscale="log", xlabel="Mean time per inference (ms)", title=group)
        ax.grid(alpha=.22, which="major")
    axes[0].set_ylabel("Maximum centroid error (output units)")
    axes[-1].legend(loc="lower right", fontsize=8, framealpha=.94)
    fig.suptitle("Runtime and accuracy for Gaussian Mamdani inference", fontweight="bold")
    fig.savefig(RESULTS / "gaussian_runtime_error.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.35), layout="constrained")
    narrow = [row for row in samples if row["group"] == "Narrow peaks"]
    for name in list(METHODS)[:-1]:
        rows = [row for row in narrow if row["method"] == name]
        ax.plot([float(row["case"]) for row in rows], [row["centroid"] for row in rows],
                label=name, color=palette[name], lw=1.9, marker=markers[name], markersize=3)
    rows = [row for row in narrow if row["method"] == "Analytic"]
    ax.plot([float(row["case"]) for row in rows], [row["reference"] for row in rows],
            color="#172c38", linestyle="--", lw=1.0, label="Quadrature reference")
    ax.set(xlabel="Narrow peak position across one default grid interval",
           ylabel="Calculated centroid (output units)", title="A narrow peak passes between Simpful’s grid samples")
    ax.grid(alpha=.22)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(RESULTS / "gaussian_grid_phase.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.35), layout="constrained")
    x = np.linspace(3, 21, 1600)
    mu_a, sigma_a, mu_b, sigma_b = 9, 1.7, 11.4, 3.0
    first = np.exp(-0.5 * ((x - mu_a) / sigma_a) ** 2)
    second = np.minimum(.95, np.exp(-0.5 * ((x - mu_b) / sigma_b) ** 2))
    envelope = np.maximum(first, second)
    ax.fill_between(x, envelope, color="#cbe7df", label="Combined area")
    ax.plot(x, first, color="#4672a3", linestyle="--", label="First rule output")
    ax.plot(x, second, color="#a64a26", linestyle="--", label="Second rule output, clipped")
    ax.plot(x, envelope, color="#00766c", linewidth=1.8, label="Upper envelope")
    crossing = mu_a + (mu_b - mu_a) * sigma_a / (sigma_a + sigma_b)
    height = math.exp(-0.5 * ((crossing - mu_a) / sigma_a) ** 2)
    ax.scatter([crossing], [height], c="#142d39", zorder=4)
    ax.annotate("The winning rule changes", (crossing, height), (12, .58),
                arrowprops={"arrowstyle": "->", "color": "#142d39"}, color="#142d39")
    ax.set(xlabel="Output value", ylabel="Membership", ylim=(0, 1.07),
           title="Direct integration follows the upper envelope")
    ax.legend(loc="upper right", fontsize=8, framealpha=.92)
    fig.savefig(RESULTS / "gaussian_envelope.png", dpi=200)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    print("Case group | Method | Cases | Max error | Mean error | Mean ± SD per inference", flush=True)
    results, samples = measure()
    save_csv("gaussian_centroid_summary.csv", results)
    save_csv("gaussian_centroid_cases.csv", samples)
    plot(results, samples)


if __name__ == "__main__":
    main()
