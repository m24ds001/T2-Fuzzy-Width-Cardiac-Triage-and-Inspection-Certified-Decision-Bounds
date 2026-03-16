"""
run_all_validations.py — Master runner for the full validation suite.

Executes all five validation scripts and prints a summary table.

Usage:
    cd code
    python run_all_validations.py

Requires: numpy>=1.24, scipy>=1.10

Repository: https://github.com/m24ds001/T2-Fuzzy-Width-Cardiac-Triage-and-Inspection-Certified-Decision-Bounds
Year: 2026
"""

import subprocess, sys, time, os

scripts = [
    ("Lemmas 1-2, Prop 1, Theorems 1-3, Corollary 1",  "validate_L1_L2_P1_T1_T3.py"),
    ("Theorems 4-6 (triangular, Mamdani, pipeline, AR)", "validate_T4_T6.py"),
    ("Theorems 7-11 (margins, structural, budget, etc.)", "validate_T7_T11.py"),
    ("Theorems 12-16 (phase, Lipschitz, sub-Gaussian, etc.)", "validate_T12_T16.py"),
    ("Experimental scenarios 4.1-4.5",                   "validate_experiments.py"),
]

code_dir    = os.path.dirname(os.path.abspath(__file__))
total_start = time.time()
results     = []

print("=" * 65)
print("  IT2 Fuzzy Width: Cardiac Triage and Inspection")
print("  Certified Decision Bounds — Full Validation Suite")
print("  Year: 2026")
print("=" * 65)
print()

for label, script in scripts:
    path  = os.path.join(code_dir, script)
    start = time.time()
    print(f">>> {script}")
    print(f"    ({label})")
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, cwd=code_dir
        )
        elapsed = time.time() - start
        out = proc.stdout + proc.stderr
        n_pass = out.count("PASS")
        n_fail = out.count("FAIL")
        results.append((script, n_pass, n_fail, elapsed))
        print(proc.stdout)
        if proc.returncode != 0 and proc.stderr:
            print(f"    STDERR: {proc.stderr[:400]}")
        status = "OK" if n_fail == 0 else f"{n_fail} FAIL(s)"
        print(f"    [{status}]  {n_pass} passed  {elapsed:.1f}s")
    except Exception as e:
        print(f"    [ERROR] {e}")
        results.append((script, 0, 1, 0))
    print()

elapsed_total = time.time() - total_start
total_pass = sum(r[1] for r in results)
total_fail = sum(r[2] for r in results)

print("=" * 65)
print(f"  {'Script':<44} {'Pass':>5} {'Fail':>5}  {'Time':>6}")
print(f"  {'-'*44} {'-'*5} {'-'*5}  {'-'*6}")
for s, np_, nf, t in results:
    ok = "\033[92mOK\033[0m" if nf == 0 else f"\033[91m{nf} FAIL\033[0m"
    print(f"  {s:<44} {np_:>5} {nf:>5}  {t:>5.1f}s  {ok}")
print(f"  {'TOTAL':<44} {total_pass:>5} {total_fail:>5}  {elapsed_total:>5.1f}s")
print()
if total_fail == 0:
    print("  \033[92mAll checks passed. All manuscript results validated.\033[0m")
else:
    print(f"  \033[91m{total_fail} check(s) failed. Review output above.\033[0m")
print("=" * 65)
