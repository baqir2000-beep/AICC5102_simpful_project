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
  - Each seed (build + warm-up + timed runs) runs in its own process and is stopped
    after TIME_CAP seconds. If a size hits the cap, larger sizes are not attempted
    (they have 9x more rules each).
"""
import multiprocessing as mp
import queue
import functools
import operator
import statistics
import time

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from common import time_ms
from fast_mamdani import fast_mamdani
from limitation_B_fixed import build_system, random_spec, input_counts, seeds

TIME_CAP = 120.0  # seconds per seed: build + warm-up + timed runs (both methods)
TIMED_RUNS = 3


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
    simulation = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules), cache=False)

    def run():
        for i, value in enumerate(inputs):
            simulation.input[f"Input{i}"] = value
        simulation.compute()
        return float(simulation.output["Output"])
    return run


def measure_seed(n, seed, results):
    """Runs in a child process so it can be stopped at TIME_CAP."""
    start = time.perf_counter()
    run = build_skfuzzy(n, seed)
    build_s = time.perf_counter() - start
    skf_value = run()                                   # warm-up
    FS, _ = build_system(n, seed)
    exact = fast_mamdani(FS, ["Output"])["Output"]
    t_skf = time_ms(run, runs=TIMED_RUNS, warmup=0)[0]
    t_ours = time_ms(lambda: fast_mamdani(FS, ["Output"]), runs=TIMED_RUNS)[0]
    results.put((build_s, t_skf, t_ours, abs(skf_value - exact)))


def run_with_cap(n, seed):
    """(build_s, t_skf, t_ours, error), or None if the seed exceeded TIME_CAP."""
    results = mp.Queue()
    worker = mp.Process(target=measure_seed, args=(n, seed, results))
    worker.start()
    try:
        return results.get(timeout=TIME_CAP)
    except queue.Empty:
        return None
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join()


if __name__ == "__main__":
    print("=== limitation_B_skfuzzy.py\n")
    print(f"Seeds: {len(seeds)} | time = mean ± sd over seeds "
          f"(each seed: warm-up + {TIMED_RUNS} runs) | cap {TIME_CAP:.0f} s per seed")
    print("Inputs | Rules | Scikit-Fuzzy build | Scikit-Fuzzy time | Ours time | "
          "Speed-up (ours vs SKF) | SKF max |error|")
    print("-" * 110)
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
    stop = False
    for n in input_counts:
        rules = 3 ** n
        if stop:
            print(f"{n} | {rules} | not attempted (smaller size already exceeded the cap)", flush=True)
            continue
        rows = []
        for seed in seeds:
            row = run_with_cap(n, seed)
            if row is None:
                print(f"{n} | {rules} | > {TIME_CAP:.0f} s for seed {seed} "
                      f"(build + {TIMED_RUNS + 1} inferences); size skipped", flush=True)
                stop = True
                break
            rows.append(row)
        if stop:
            continue
        build_s, t_skf, t_ours, errs = map(list, zip(*rows))
        print(f"{n} | {rules} | {statistics.mean(build_s):.2f} s | "
              f"{statistics.mean(t_skf):.2f} ± {sd(t_skf):.2f} ms | "
              f"{statistics.mean(t_ours):.2f} ± {sd(t_ours):.2f} ms | "
              f"{statistics.mean(t_skf) / statistics.mean(t_ours):.1f}x | {max(errs):.2e}",
              flush=True)
    print("\nSKF max |error| = Scikit-Fuzzy vs our exact centroid (its sampled-universe error).")
