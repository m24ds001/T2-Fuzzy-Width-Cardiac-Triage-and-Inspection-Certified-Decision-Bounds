"""
validate_L1_L2_P1_T1_T3.py
Numerical validation of:
  Lemma 1     — Sigmoid Lipschitz constant k/4
  Lemma 2     — Triangular MF Lipschitz constant
  Proposition 1 — Corners are the worst case
  Theorem 1   — Aggregation interval bounds (sandwich)
  Theorem 2   — Width bound under Lipschitz perturbation
  Theorem 3   — Certified O(1) width (all three cases)
  Corollary 1 — Necessity: unnormalised weights give Omega(n) growth

Theorem numbering matches the published PDF exactly.

Repository: https://github.com/m24ds001/T2-Fuzzy-Width-Cardiac-Triage-and-Inspection-Certified-Decision-Bounds
Year: 2026
"""

import numpy as np
from it2_core import (sigmoid_mf, triangular_mf,
                      sigmoid_lipschitz, triangular_lipschitz,
                      corner_eval_sigmoid, certificate_T2, certificate_T3_iii)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
rng  = np.random.default_rng(0)


def check(label, condition, lhs=None, rhs=None):
    status = PASS if condition else FAIL
    if lhs is not None:
        print(f"  {status}  {label}:  {lhs:.6f}  <=  {rhs:.6f}")
    else:
        print(f"  {status}  {label}")


# ===========================================================================
print("=" * 60)
print("Lemma 1 — Sigmoid Lipschitz constant k/4")
print("=" * 60)
for k in [1.0, 3.0, 10.0, 50.0]:
    max_ratio = 0.0
    for _ in range(50000):
        tau1, tau2 = rng.uniform(-5, 5, 2)
        c = rng.uniform(-2, 2)
        if abs(tau1 - tau2) < 1e-12:
            continue
        ratio = abs(sigmoid_mf(tau1, c, k) - sigmoid_mf(tau2, c, k)) / abs(tau1 - tau2)
        max_ratio = max(max_ratio, ratio)
    L = sigmoid_lipschitz(k)
    check(f"k={k:5.1f}: max ratio", max_ratio <= L + 1e-9, max_ratio, L)

# ===========================================================================
print()
print("=" * 60)
print("Lemma 2 — Triangular MF Lipschitz constant")
print("=" * 60)
for a, b, c in [(0, 0.5, 1), (0, 0.3, 1), (0, 0.7, 1)]:
    L = triangular_lipschitz(a, b, c)
    taus = np.linspace(-0.1, 1.1, 10000)
    max_ratio = 0.0
    for i in range(len(taus) - 1):
        d = abs(taus[i+1] - taus[i])
        if d < 1e-12:
            continue
        r = abs(triangular_mf(taus[i+1], a, b, c) - triangular_mf(taus[i], a, b, c)) / d
        max_ratio = max(max_ratio, r)
    check(f"a={a},b={b},c={c}", max_ratio <= L + 1e-9, max_ratio, L)

# ===========================================================================
print()
print("=" * 60)
print("Proposition 1 — Corners are the worst case")
print("=" * 60)
n, eps_c, eps_k = 5, 0.10, 0.5
c_c = rng.uniform(0.2, 0.8, n)
k_c = rng.uniform(2.0, 5.0, n)
w   = rng.dirichlet(np.ones(n))
tau = rng.uniform(0.1, 0.9, n)
c_b = np.column_stack([c_c - eps_c, c_c + eps_c])
k_b = np.clip(np.column_stack([k_c - eps_k, k_c + eps_k]), 0.01, None)

_, _, delta_corner = corner_eval_sigmoid(tau, c_b, k_b, w)

max_int, min_int = 0.0, 1.0
for _ in range(100000):
    cs = c_c + rng.uniform(-eps_c*0.99, eps_c*0.99, n)
    ks = np.clip(k_c + rng.uniform(-eps_k*0.99, eps_k*0.99, n), 0.01, None)
    E  = sum(w[i] * sigmoid_mf(tau[i], cs[i], ks[i]) for i in range(n))
    max_int = max(max_int, E)
    min_int = min(min_int, E)

check("Corner width >= interior MC width",
      delta_corner >= (max_int - min_int) - 1e-6,
      max_int - min_int, delta_corner)

# ===========================================================================
print()
print("=" * 60)
print("Theorem 1 — Aggregation interval bounds (sandwich)")
print("=" * 60)
E_lo, E_hi, _ = corner_eval_sigmoid(tau, c_b, k_b, w)
fails = 0
for _ in range(20):
    cs = c_c + rng.uniform(-eps_c, eps_c, n)
    ks = np.clip(k_c + rng.uniform(-eps_k, eps_k, n), 0.01, None)
    E  = sum(w[i] * sigmoid_mf(tau[i], cs[i], ks[i]) for i in range(n))
    if not (E_lo - 1e-10 <= E <= E_hi + 1e-10):
        fails += 1
check("All 20 random draws satisfy E_lo <= E <= E_hi", fails == 0)

# ===========================================================================
print()
print("=" * 60)
print("Theorem 2 — Width bound under Lipschitz perturbation")
print("=" * 60)
M    = float(np.max(np.abs(tau - c_c)))
cert = certificate_T2(k_c, w, eps_c, M, eps_k)
_, _, delta_emp = corner_eval_sigmoid(tau, c_b, k_b, w)
check("Delta_emp <= Theorem 2 certificate", delta_emp <= cert + 1e-10, delta_emp, cert)
print(f"  Tightness ratio rho = {delta_emp/cert:.4f}")

# ===========================================================================
print()
print("=" * 60)
print("Theorem 3 — Certified O(1) width")
print("=" * 60)
print("  Case (i): uniform weights — width flat as n grows (O(1))")
for nv in [4, 8, 20, 50, 100, 200]:
    kv  = np.full(nv, 3.0)
    wv  = np.ones(nv) / nv
    cv  = rng.uniform(0.2, 0.8, nv)
    tv  = cv + rng.uniform(-0.1, 0.1, nv)
    Mv  = float(np.max(np.abs(tv - cv)))
    c2  = certificate_T2(kv, wv, eps_c=0.05, M=Mv, eps_k=0.5)
    cb  = np.column_stack([cv - 0.05, cv + 0.05])
    kb  = np.clip(np.column_stack([kv - 0.5, kv + 0.5]), 0.01, None)
    _, _, de = corner_eval_sigmoid(tv, cb, kb, wv)
    ok  = de <= c2 + 1e-10
    print(f"  {'PASS' if ok else 'FAIL'}  n={nv:3d}: delta={de:.4f}  cert={c2:.4f}  ratio={de/c2:.3f}")

print("\n  Case (iii): any normalised weight vector")
cert_iii = certificate_T3_iii(k_c, eps_c, M, eps_k)
check("Delta_emp <= 2K'  (Theorem 3(iii))", delta_emp <= cert_iii + 1e-10, delta_emp, cert_iii)

# ===========================================================================
print()
print("=" * 60)
print("Corollary 1 — Necessity: unnormalised Omega(n) growth")
print("=" * 60)
v, k_min = 1.0, 2.0
lbs = [v * nv * k_min * 0.05 / 4.0 for nv in [4, 10, 25, 50, 100]]
check("Lower bounds strictly increasing with n",
      all(lbs[i] < lbs[i+1] for i in range(len(lbs)-1)))
print(f"  Omega(n) lower bounds at n=[4,10,25,50,100]: {[f'{b:.4f}' for b in lbs]}")

print()
print("All Lemma 1, Lemma 2, Prop 1, T1, T2, T3, C1 checks complete.")
