"""
validate_T4_T6.py
Numerical validation of:
  Theorem 4   — IT2 width certificate for triangular MFs
  Corollary 2 — Mamdani IT2 rule-firing width bound
  Theorem 5   — Cascaded pipeline width certificate (SM-Theorem 1.1)
  Theorem 6   — Width certificate under AR inputs (SM-Propositions 5.1-5.2)

Theorem numbering matches the published PDF exactly.

Repository: https://github.com/m24ds001/T2-Fuzzy-Width-Cardiac-Triage-and-Inspection-Certified-Decision-Bounds
Year: 2026
"""

import numpy as np
from it2_core import (triangular_lipschitz, corner_eval_sigmoid,
                      corner_eval_triangular, certificate_T4)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
rng  = np.random.default_rng(42)


def check(label, condition, lhs=None, rhs=None):
    status = PASS if condition else FAIL
    if lhs is not None:
        print(f"  {status}  {label}:  {lhs:.6f}  <=  {rhs:.6f}")
    else:
        print(f"  {status}  {label}")


# ===========================================================================
print("=" * 60)
print("Theorem 4 — IT2 width certificate for triangular MFs")
print("=" * 60)
n   = 6
eps = 0.04
a_c = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
b_c = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
c_c = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
w   = rng.dirichlet(np.ones(n))
tau = rng.uniform(0.05, 0.95, n)

L_tri = np.array([triangular_lipschitz(a_c[i], b_c[i], c_c[i]) for i in range(n)])
L_bar = float(np.dot(w, L_tri))
cert4 = certificate_T4(L_bar, eps)

a_b = np.column_stack([a_c - eps, a_c + eps])
b_b = np.column_stack([b_c - eps, b_c + eps])
c_b = np.column_stack([c_c - eps, c_c + eps])
_, _, delta_tri = corner_eval_triangular(tau, a_b, b_b, c_b, w)

check("Delta_tri <= 2*L_bar*eps  (Theorem 4)", delta_tri <= cert4 + 1e-10, delta_tri, cert4)
print(f"  Tightness ratio rho = {delta_tri/cert4:.4f}")

# ===========================================================================
print()
print("=" * 60)
print("Corollary 2 — Mamdani IT2 rule-firing width bound")
print("=" * 60)
L_max      = float(np.max(L_tri))
Lambda_out = 0.25   # <= (y_R - y_L)/4 for unit output universe (Remark 1)
cert_C2    = cert4 + 2.0 * Lambda_out * L_max * eps
print(f"  L_bar={L_bar:.4f}, L_max={L_max:.4f}, Lambda_out={Lambda_out:.4f}")
print(f"  Corollary 2 certificate: {cert_C2:.6f}")
check("Bound positive and finite", cert_C2 > 0)

# ===========================================================================
print()
print("=" * 60)
print("Theorem 5 — Cascaded pipeline (SM-Theorem 1.1)")
print("=" * 60)
# Use rho_AR = 0.55 so that kappa_0 < 1 (rho_AR < 4/(4+k_avg) = 4/7 ≈ 0.571)
stages = [
    {"k_avg": 3.0, "eps_c": 0.05, "eps_k": 0.5, "M": 0.10},
    {"k_avg": 2.0, "eps_c": 0.06, "eps_k": 0.4, "M": 0.08},
    {"k_avg": 2.5, "eps_c": 0.04, "eps_k": 0.3, "M": 0.06},
]
kappa   = [s["k_avg"] / 4.0 for s in stages]
K_prime = [s["k_avg"] / 4.0 * s["eps_c"] + s["M"] / 4.0 * s["eps_k"]
           for s in stages]

# Unrolled recurrence (Eq. 4 of the paper)
Delta = [2.0 * K_prime[0]]
for p in range(1, 3):
    Delta.append(kappa[p] * Delta[-1] + 2.0 * K_prime[p])

kappa_max = max(kappa)
K_star    = max(K_prime)
geom_bound = 2.0 * K_star / (1.0 - kappa_max)
print(f"  kappa_max={kappa_max:.4f} < 1: geometric bound valid")
print(f"  Geometric bound 2K*/(1-kappa) = {geom_bound:.6f}")
for p, dv in enumerate(Delta):
    check(f"  Stage {p+1}: recurrence <= geom bound", dv <= geom_bound + 1e-10, dv, geom_bound)

# Empirical check
print("\n  Empirical cascade simulation (5 trials):")
for trial in range(5):
    ws = []
    for s in stages:
        nv = 5
        kv = s["k_avg"] * np.ones(nv)
        cv = 0.5 * np.ones(nv)
        tv = cv + rng.uniform(-s["M"], s["M"], nv)
        wv = np.ones(nv) / nv
        cb = np.column_stack([cv - s["eps_c"], cv + s["eps_c"]])
        kb = np.clip(np.column_stack([kv - s["eps_k"], kv + s["eps_k"]]), 0.01, None)
        _, _, dv = corner_eval_sigmoid(tv, cb, kb, wv)
        ws.append(dv)
    ok = all(e <= c + 1e-6 for e, c in zip(ws, Delta))
    print(f"  {'PASS' if ok else 'FAIL'}  trial {trial+1}: "
          f"widths={[f'{w:.4f}' for w in ws]}  certs={[f'{c:.4f}' for c in Delta]}")

# ===========================================================================
print()
print("=" * 60)
print("Theorem 6 — Width certificate under AR inputs")
print("  (SM-Propositions 5.1 and 5.2)")
print("=" * 60)
rho_AR, sigma_eta, k_avg = 0.55, 0.018, 3.0
eps_c, eps_k = 0.05, 0.5

M_inf    = sigma_eta / (1.0 - rho_AR)
kappa_0  = (k_avg / 4.0) * rho_AR / (1.0 - rho_AR)
print(f"  rho_AR={rho_AR}, sigma_eta={sigma_eta}")
print(f"  SM-Prop 5.1: M_inf = {M_inf:.4f}")
print(f"  SM-Prop 5.2: kappa_0 = {kappa_0:.4f} < 1: finiteness condition satisfied")
check("kappa_0 < 1", kappa_0 < 1)

Delta_inf_cert = (k_avg * eps_c / 2.0 + M_inf * eps_k / 2.0) / (1.0 - kappa_0)
print(f"  Theorem 6 certificate Delta_inf = {Delta_inf_cert:.4f}")

# Simulate AR(1) process
T, n_src = 500, 5
cv = 0.5 * np.ones(n_src)
wv = np.ones(n_src) / n_src
tau_t = cv.copy()
widths = []
for t in range(T):
    eta   = rng.uniform(-sigma_eta, sigma_eta, n_src)
    tau_t = rho_AR * tau_t + (1 - rho_AR) * cv + eta
    tau_t = np.clip(tau_t, 0.0, 1.0)
    kv    = k_avg * np.ones(n_src)
    cb    = np.column_stack([cv - eps_c, cv + eps_c])
    kb    = np.clip(np.column_stack([kv - eps_k, kv + eps_k]), 0.01, None)
    _, _, dv = corner_eval_sigmoid(tau_t, cb, kb, wv)
    if t >= 100:
        widths.append(dv)

max_emp = max(widths)
check("max AR width <= Theorem 6 certificate",
      max_emp <= Delta_inf_cert + 1e-6, max_emp, Delta_inf_cert)

print()
print("All T4, C2, T5, T6 checks complete.")
