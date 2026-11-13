"""Proposal ablation: Mamdani vs Sugeno on the same rule base, to locate where the
cost comes from.

The Sugeno twin has the same inputs, rules and input values as the random rule base
from limitation_B_fixed.py; each output set is replaced by its centroid as a crisp
value (small 10/3, average 10, large 235/12). Sugeno has no defuzzification grid, so
    time(Mamdani) - time(Sugeno)  ~  cost of defuzzification
    time(Sugeno)                  ~  cost of evaluating the rules
The same split is reported for our methods (fast_mamdani vs fast_sugeno).

Note: Mamdani and Sugeno give different outputs (different methods); this test is
about where the time goes, not about which answer is better. The mean |difference|
between the two outputs is printed for information only.
fast_sugeno has no active-rule lookup, so its column shows rule-evaluation cost
with caching only.
"""
import statistics

from common import time_ms
from fast_mamdani import fast_mamdani
from fast_sugeno import fast_sugeno
from limitation_B_fixed import build_system, input_counts, seeds

SUBDIVISIONS = 1000
TIMED_RUNS = 3


def mean_sd(values):
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), sd


if __name__ == "__main__":
    print("=== ablation_mamdani_vs_sugeno.py\n")
    print(f"Seeds: {len(seeds)} | Simpful Mamdani at {SUBDIVISIONS} subdivisions | "
          f"time = mean ± sd over seeds (each seed: warm-up + {TIMED_RUNS} runs)")
    print("Inputs | Rules | Simpful Mamdani | Simpful Sugeno | Simpful defuzz share | "
          "Ours Mamdani | Ours Sugeno | Mean |Mamdani - Sugeno|")
    print("-" * 120)
    for n in input_counts:
        sm, ss, om, os_, gaps = [], [], [], [], []
        for seed in seeds:
            FSm, rules = build_system(n, seed)
            FSs, _ = build_system(n, seed, sugeno=True)
            mamdani = FSm.Mamdani_inference(["Output"], subdivisions=SUBDIVISIONS,
                                            ignore_warnings=True)["Output"]
            sugeno = FSs.Sugeno_inference(["Output"], ignore_warnings=True)["Output"]
            gaps.append(abs(float(mamdani) - float(sugeno)))
            sm.append(time_ms(lambda: FSm.Mamdani_inference(
                ["Output"], subdivisions=SUBDIVISIONS, ignore_warnings=True), runs=TIMED_RUNS)[0])
            ss.append(time_ms(lambda: FSs.Sugeno_inference(
                ["Output"], ignore_warnings=True), runs=TIMED_RUNS)[0])
            om.append(time_ms(lambda: fast_mamdani(FSm, ["Output"]), runs=TIMED_RUNS)[0])
            os_.append(time_ms(lambda: fast_sugeno(FSs, ["Output"]), runs=TIMED_RUNS)[0])
        (smm, sms), (ssm, sss) = mean_sd(sm), mean_sd(ss)
        (omm, oms), (osm, oss) = mean_sd(om), mean_sd(os_)
        share = max(0.0, (smm - ssm) / smm)
        print(f"{n} | {rules} | {smm:.2f} ± {sms:.2f} ms | {ssm:.2f} ± {sss:.2f} ms | "
              f"{share:.0%} | {omm:.2f} ± {oms:.2f} ms | {osm:.2f} ± {oss:.2f} ms | "
              f"{statistics.mean(gaps):.3f}", flush=True)
    print("\nSimpful defuzz share = (Simpful Mamdani - Simpful Sugeno) / Simpful Mamdani: "
          "the part of Simpful's Mamdani time spent on the defuzzification grid.")
