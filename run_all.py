"""Run every experiment in order and write one combined results file.

Usage:
    python run_all.py                       # run everything (about 25-35 minutes)
    python run_all.py ablation sepsis.py    # run only the named scripts

Output: results/all_results.txt (plus figures/CSV written by individual scripts).
Run it on an otherwise idle machine: other programs running at the same time
make the timings slower and noisier.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# (script, what it shows)
SCRIPTS = [
    ("mamdani.py",                     "Reproduction: Listing 1, Mamdani tipping (paper: 14.17)"),
    ("sugeno.py",                      "Reproduction: Listing 2, Sugeno tipping (paper: 14.77)"),
    ("sepsis.py",                      "Reproduction: Listing 3, sepsis (paper: 69.3; figure: results/sepsis_mfs.pdf)"),
    ("repressilator.py",               "Reproduction: Listing 4, repressilator (figure: results/repressilator_dynamics.pdf)"),
    ("figure4_test.py",                "Reproduction: Fig. 4, Simpful vs Scikit-Fuzzy surface"),
    ("limitation_A_fixed.py",          "Limitation A + fix: accuracy (1 input and 121 inputs)"),
    ("limitation_A_harder.py",         "Limitation A, harder conditions: narrow and Gaussian/sigmoid output sets"),
    ("benchmark_gaussian_centroid.py", "Gaussian centroid: runtime, accuracy, narrow peaks and crossing ablation"),
    ("limitation_B_fixed.py",          "Limitation B + fix: speed with 9 to 6561 rules"),
    ("limitation_B_skfuzzy.py",        "Limitation B: Scikit-Fuzzy on the same rule bases"),
    ("temperature_control_mamdani.py", "Second Mamdani system: fan control"),
    ("benchmark_sepsis_sugeno.py",     "Sugeno system: sepsis speed and equivalence"),
    ("ablation.py",                    "Ablation: remove one component of the fix at a time"),
    ("ablation_mamdani_vs_sugeno.py",  "Ablation: Mamdani vs Sugeno on the same rule base (proposal)"),
    ("test_fast_mamdani.py",           "Tests: grid mode = Simpful, exact mode = reference, fast_sugeno = Simpful"),
    ("test_gaussian_centroid.py",       "Tests: analytic Gaussian centroid and integration with Simpful"),
    ("findings_checks.py",             "Findings: Sugeno rule weights (with fix), sepsis membership labels"),
    ("sepsis_corrected.py",            "Fix for F2: sepsis model with labels matching their shapes"),
]

BANNER_MARKERS = ("____", "___)", "___ ", "|_|", "Created by", "github.com/aresio", "Detected Sugeno", " v2.12")


def clean(text):
    """Drop Simpful's start-up banner lines so the results file stays readable."""
    kept = [l for l in text.splitlines() if not any(m in l for m in BANNER_MARKERS)]
    out = []
    for line in kept:                      # collapse runs of blank lines
        if line.strip() or (out and out[-1].strip()):
            out.append(line)
    return "\n".join(out).strip()


def main():
    wanted = sys.argv[1:]
    scripts = [(s, d) for s, d in SCRIPTS
               if not wanted or any(s.startswith(w) for w in wanted)]
    RESULTS.mkdir(exist_ok=True)
    # MPLBACKEND=Agg: no plot windows. PYTHONUTF8/PYTHONIOENCODING: on Windows a
    # piped child process writes stdout in the ANSI code page (cp1252), so printing
    # Simpful's banner or a "±" sign crashes with UnicodeEncodeError. Force UTF-8.
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    sections, summary = [], []
    env_info = environment_info()
    # A partial run does not overwrite a full one: name the file after the scripts run.
    out = RESULTS / ("all_results.txt" if not wanted else f"results_{'_'.join(wanted)}.txt")
    for script, description in scripts:
        print(f"Running {script} ...", flush=True)
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, script], cwd=HERE, env=env,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        seconds = time.perf_counter() - start
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        output = clean(proc.stdout)
        if proc.stderr.strip():
            output += "\n\nSTDERR:\n" + proc.stderr.strip()
        sections.append(f"{'=' * 100}\n{script} - {description}\n"
                        f"Status: {status} | run time {seconds:.1f} s\n{'=' * 100}\n{output}\n")
        summary.append(f"{status:<8} {seconds:7.1f} s  {script}")
        print(f"  {status} in {seconds:.1f} s", flush=True)
        write_results(out, env_info, summary, sections)   # save after every script
    print(f"\nWrote {out}")
    return int(any(entry.startswith("FAILED") for entry in summary))


def write_results(out, env_info, summary, sections):
    header = env_info + "\nSummary\n" + "\n".join(summary) + "\n"
    out.write_text(header + "\n" + "\n".join(sections), encoding="utf-8")


def cpu_name():
    """CPU model, so absolute timings can be compared between machines."""
    import platform
    try:                                   # Linux
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:                                   # macOS
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except OSError:
        pass
    return platform.processor() or "unknown"   # Windows


def environment_info():
    import platform
    from importlib.metadata import version
    lines = ["AICC5102 Simpful project - combined results",
             f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
             f"Python {platform.python_version()} on {platform.system()} {platform.machine()}",
             f"CPU: {cpu_name()}"]
    for package in ("simpful", "scikit-fuzzy", "numpy", "scipy", "matplotlib"):
        try:
            lines.append(f"{package} {version(package)}")
        except Exception:
            lines.append(f"{package} (not found)")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
