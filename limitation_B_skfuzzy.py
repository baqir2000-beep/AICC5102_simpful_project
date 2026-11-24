"""Limitation B with Scikit-Fuzzy (proposal section 4B: "Simpful vs. Scikit-Fuzzy").

Same random full-grid rule bases and inputs as limitation_B_fixed.py (shared
generator random_spec), same 5 seeds. Scikit-Fuzzy uses min for AND, max
aggregation and centroid defuzzification, like Simpful; its output universe has
1001 points (Simpful's default grid has 1000).

Reported per size: Scikit-Fuzzy time per inference, our time, Scikit-Fuzzy's max
error against our exact result, and the time to build the Scikit-Fuzzy system.
Simpful's times are in limitation_B_fixed.py (not repeated here, it takes minutes).

Fairness notes
  - Scikit-Fuzzy's result cache is switched off (cache=False); otherwise repeated
    calls with the same input would return a stored answer.
  - Inputs are set inside every timed call, as a user would do.
  - The fixed, single-stage rule base has no intermediate terms. Build its
    control graph once and reuse its rule order. Scikit-Fuzzy still evaluates
    every rule and performs its own accumulation and defuzzification.
"""
import functools
import operator
import statistics
import time

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from common import time_ms
from fast_mamdani import fast_mamdani
from limitation_B_fixed import TIMED_RUNS, build_system, input_counts, random_spec, sd, seeds


class FixedRuleControlSystem(ctrl.ControlSystem):
    """Scikit-Fuzzy controller for a fixed set of independent rules."""

    def __init__(self, rules):
        super().__init__()
        self._ordered_rules = tuple(rules)
        for rule in self._ordered_rules:
            self.graph.update(rule.graph)

    @property
    def rules(self):
        return iter(self._ordered_rules)


def build_skfuzzy(num_inputs, seed):
    rule_spec, inputs = random_spec(num_inputs, seed)
    antecedents = []
    for i in range(num_inputs):
        a = ctrl.Antecedent(np.linspace(0, 10, 101), f"Input{i}")
        a["low"] = fuzz.trimf(a.universe, [0, 0, 5])
        a["medium"] = fuzz.trimf(a.universe, [0, 5, 10])
        a["high"] = fuzz.trimf(a.universe, [5, 10, 10])
        antecedents.append(a)
    out = ctrl.Consequent(np.linspace(0, 25, 1001), "Output")
    out["small"] = fuzz.trimf(out.universe, [0, 0, 10])
    out["average"] = fuzz.trimf(out.universe, [0, 10, 20])
    out["large"] = fuzz.trapmf(out.universe, [10, 20, 25, 25])
    rules = [ctrl.Rule(functools.reduce(operator.and_,
                                        [antecedents[i][t] for i, t in enumerate(combination)]),
                       out[term])
             for combination, term in rule_spec]
    simulation = ctrl.ControlSystemSimulation(FixedRuleControlSystem(rules), cache=False)

    def run():
        for i, value in enumerate(inputs):
            simulation.input[f"Input{i}"] = value
        simulation.compute()
        return float(simulation.output["Output"])
    return run


def measure_seed(n, seed):
    start = time.perf_counter()
    run = build_skfuzzy(n, seed)
    build_s = time.perf_counter() - start
    skf_value = run()                                   # warm-up
    FS, _ = build_system(n, seed)
    exact = fast_mamdani(FS, ["Output"])["Output"]
    t_skf = time_ms(run, runs=TIMED_RUNS, warmup=0)[0]
    t_ours = time_ms(lambda: fast_mamdani(FS, ["Output"]), runs=TIMED_RUNS)[0]
    return build_s, t_skf, t_ours, abs(skf_value - exact)


if __name__ == "__main__":
    print("=== limitation_B_skfuzzy.py\n")
    print(f"Seeds: {len(seeds)} | time = mean ± sd over seeds "
          f"(each seed: warm-up + {TIMED_RUNS} runs)")
    print("Scikit-Fuzzy uses a preordered control graph; result caching is off.")
    print("Inputs | Rules | Scikit-Fuzzy build | Scikit-Fuzzy time | Ours time | "
          "Speed-up (ours vs SKF) | SKF max |error|")
    print("-" * 110)
    for n in input_counts:
        rules = 3 ** n
        rows = [measure_seed(n, seed) for seed in seeds]
        build_s, t_skf, t_ours, errs = map(list, zip(*rows))
        print(f"{n} | {rules} | {statistics.mean(build_s):.2f} s | "
              f"{statistics.mean(t_skf):.2f} ± {sd(t_skf):.2f} ms | "
              f"{statistics.mean(t_ours):.2f} ± {sd(t_ours):.2f} ms | "
              f"{statistics.mean(t_skf) / statistics.mean(t_ours):.1f}x | {max(errs):.2e}",
              flush=True)
    print("\nSKF max |error| = Scikit-Fuzzy vs our exact centroid (its sampled-universe error).")
