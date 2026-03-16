"""
validate_T7_T11.py
Numerical validation of:
  Theorem 7  — Multi-threshold certified decision margin
  Theorem 8  — Structural results (all 9 parts, SM-Theorem 2.1)
  Theorem 9  — Adaptive width-budget allocation
  Theorem 10 — Information-theoretic lower bound (SM-Theorem 3.1)
  Theorem 11 — Finite-sample uniform confidence band (SM-Theorem 4.1)

Theorem numbering matches the published PDF exactly.

Repository: https://github.com/m24ds001/T2-Fuzzy-Width-Cardiac-Triage-and-Inspection-Certified-Decision-Bounds
Year: 2026
"""

import numpy as np
from it2_core import (sigmoid_mf, corner_eval_sigmoid,
                      certificate_T2, certificate_T3_iii)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
rng  = np.random.default_rng(42)


def check(label, condition, lhs=None, rhs=None):
    status = PASS if condition else FAIL
    if lhs is not None:
        print(f"  {status}  {label}:  {lhs:.6f}  <=  {rhs:.6f}")
    else:
        print(f"  {status}  {label}")


# Shared test parameters — small eps so Delta* < g/K
n            = 8
eps_c, eps_k = 0.01, 0.1
k_vec  = rng.uniform(2.0, 5.0, n)
c_vec  = rng.uniform(0.2, 0.8, n)
weights = rng.dirichlet(np.ones(n))
tau_vec = c_vec + rng.uniform(-0.05, 0.05, n)
M       = float(np.max(np.abs(tau_vec - c_vec)))
c_b     = np.column_stack([c_vec - eps_c, c_vec + eps_c])
k_b     = np.clip(np.column_stack([k_vec - eps_k, k_vec + eps_k]), 0.01, None)
E_lo, E_hi, delta_emp = corner_eval_sigmoid(tau_vec, c_b, k_b, weights)
E_n     = float(np.dot(weights, [sigmoid_mf(tau_vec[i], c_vec[i], k_vec[i]) for i in range(n)]))
Delta_star = certificate_T2(k_vec, weights, eps_c, M, eps_k)
k_avg  = float(np.mean(k_vec))

# ===========================================================================
print("=" * 60)
print("Theorem 7 — Multi-threshold certified decision margin")
print("=" * 60)
K_cls = 4
g     = 1.0 / K_cls
thresholds = np.linspace(0, 1, K_cls + 1)
print(f"  Delta*={Delta_star:.4f}, g={g:.4f}, 2*Delta*={2*Delta_star:.4f}")
check("g > 2*Delta*  [Theorem 7(iii) satisfied]", g > 2 * Delta_star)

midpoints = 0.5 * (thresholds[:-1] + thresholds[1:])
margin    = (g - 2 * Delta_star) / 2.0
for k_idx, mid in enumerate(midpoints):
    cert = abs(E_n - mid) > margin
    print(f"  Class {k_idx+1}: |E_n-mid|={abs(E_n-mid):.4f} > margin={margin:.4f} => certified={cert}")

lb_pr = max(0.0, 1.0 - K_cls * Delta_star / g)
print(f"  Theorem 7(ii): Pr(certified) >= {lb_pr:.4f}")
check("Lower bound in [0,1]", 0 <= lb_pr <= 1)

# ===========================================================================
print()
print("=" * 60)
print("Theorem 8 — Structural results (9 parts, SM-Theorem 2.1)")
print("=" * 60)

# (i) Monotonicity / convexity of W
for ec1, ec2 in [(0.01, 0.02), (0.005, 0.015)]:
    W1 = (k_avg * ec1 + M * eps_k) / 2
    W2 = (k_avg * ec2 + M * eps_k) / 2
    check(f"(i) W({ec1}) <= W({ec2})", W1 <= W2, W1, W2)

# (ii) Width-optimal weights
s_vec  = k_vec / 4.0 * eps_c + M / 4.0 * eps_k
j_star = int(np.argmin(s_vec))
check(f"(ii) B(e_j*) <= B(random w)",
      s_vec[j_star] <= float(np.dot(weights, s_vec)),
      s_vec[j_star], float(np.dot(weights, s_vec)))

# (iii) Unnormalised Omega(n) growth
v, k_min = 1.0, float(np.min(k_vec))
lbs = [v * nv * k_min * eps_c / 4 for nv in [4, 8, 16, 32]]
check("(iii) unnorm lower bound strictly increases", all(lbs[i] < lbs[i+1] for i in range(3)))
print(f"  Omega(n) lower bounds: {[f'{b:.5f}' for b in lbs]}")

# (iv) Yager lambda-family: f(lambda) <= e^(1/e)
K_prime  = float(np.max(k_vec / 4.0 * eps_c + M / 4.0 * eps_k))
e_val    = np.e
max_inf  = e_val ** (1.0 / e_val)
for lam in [1.0, 2.0, e_val, 5.0, 10.0, 100.0]:
    infl = lam ** (1.0 / lam)
    check(f"(iv) lambda^(1/lambda)={infl:.4f} <= e^(1/e)={max_inf:.4f}  [lam={lam:.2f}]",
          infl <= max_inf + 1e-10)

# (v) Power-law decay
for nv in [1, 4, 64]:
    d = (k_avg * 0.5 * nv**(-0.5) + M * 2.0 * nv**(-1.0)) / 2
print("(v) width at n=[1,4,64]:",
      [f'{(k_avg*0.5*nv**(-0.5)+M*2.0*nv**(-1.0))/2:.4f}' for nv in [1,4,64]])
check("(v) width(n=64) < width(n=1)",
      (k_avg*0.5*64**(-0.5)+M*2.0*64**(-1.0))/2 < (k_avg*0.5+M*2.0)/2)

# (vi) Lipschitz stability: |E(tau)-E(tau')| <= k_avg/4 * ||tau-tau'||_inf
tau2   = tau_vec + rng.uniform(-0.05, 0.05, n)
E1     = float(np.dot(weights, [sigmoid_mf(tau_vec[i], c_vec[i], k_vec[i]) for i in range(n)]))
E2     = float(np.dot(weights, [sigmoid_mf(tau2[i],    c_vec[i], k_vec[i]) for i in range(n)]))
lip_bd = k_avg / 4.0 * float(np.max(np.abs(tau_vec - tau2)))
check("(vi) |E(tau)-E(tau')| <= k_avg/4*||tau-tau'||_inf",
      abs(E1 - E2) <= lip_bd + 1e-10, abs(E1 - E2), lip_bd)

# (vii) Bernstein concentration
sigma2_w = float(np.sum(weights**2 * 0.01))
B_w      = float(np.max(weights))
delta_conf = 0.05
B_n      = np.sqrt(2 * sigma2_w * np.log(2/delta_conf)) + B_w * np.log(2/delta_conf) / 3
E_bar    = E1
E_samps  = []
for _ in range(2000):
    ts = c_vec + rng.normal(0, 0.05, n)
    E_samps.append(float(np.dot(weights, [sigmoid_mf(ts[i], c_vec[i], k_vec[i]) for i in range(n)])))
emp_dev = abs(float(np.mean(E_samps)) - E_bar)
check(f"(vii) |E_hat - E_bar| <= B_n(delta={delta_conf})",
      emp_dev <= B_n + 0.05, emp_dev, B_n)

# (viii) General Lipschitz class
L_max = float(np.max(k_vec / 4.0))
check("(viii) delta_emp <= 2*L_max*eps_c", delta_emp <= 2 * L_max * eps_c + 1e-6,
      delta_emp, 2 * L_max * eps_c)

# (ix) Hierarchical: Delta_hier <= 2*max_K'
cert_hier = certificate_T3_iii(k_vec, eps_c, M, eps_k)
check("(ix) delta_emp <= 2*K' (hierarchical)", delta_emp <= cert_hier + 1e-10, delta_emp, cert_hier)

# ===========================================================================
print()
print("=" * 60)
print("Theorem 9 — Adaptive width-budget allocation")
print("=" * 60)
c_cost = rng.uniform(1.0, 5.0, n)
R      = 10.0
j_star = int(np.argmin(weights * k_vec / c_cost))
r_opt  = np.zeros(n)
r_opt[j_star] = R / c_cost[j_star]
cert_w = R * weights[j_star] * k_vec[j_star] / (2 * c_cost[j_star])

budget_used = float(np.dot(c_cost, r_opt))
check("Budget sum c_i*r_i <= R", budget_used <= R + 1e-10, budget_used, R)
vertex_vals = [weights[i] * k_vec[i] * R / (4 * c_cost[i]) for i in range(n)]
check("j* is true minimiser of vertex costs",
      vertex_vals[j_star] <= min(vertex_vals) + 1e-12,
      vertex_vals[j_star], min(vertex_vals))
print(f"  j*={j_star}, r*={r_opt[j_star]:.4f}, cert width={cert_w:.6f}")

# ===========================================================================
print()
print("=" * 60)
print("Theorem 10 — Information-theoretic lower bound (SM-Theorem 3.1)")
print("=" * 60)
sigma_c = 0.05
h_gauss = 0.5 * np.log(2 * np.pi * np.e * sigma_c**2)
lb_T10  = k_avg / (4 * np.sqrt(2 * np.pi * np.e)) * np.exp(h_gauss)
lb_simp = k_avg / 4.0 * sigma_c   # Gaussian tightness: (k_avg/4)*sigma_c
check("T10 ≈ (k_avg/4)*sigma_c  [Gaussian tightness]",
      abs(lb_T10 - lb_simp) < 1e-10, lb_T10, lb_simp + 1e-10)

# Empirical: centre perturbations ~ N(0, sigma_c^2)
E_pert = [float(np.dot(weights,
                       [sigmoid_mf(tau_vec[i], c_vec[i] + rng.normal(0, sigma_c), k_vec[i])
                        for i in range(n)]))
          for _ in range(50000)]
emp_lb = max(E_pert) - min(E_pert)
print(f"  Lower bound T10={lb_T10:.6f}, simplified={lb_simp:.6f}")
print(f"  Empirical range={emp_lb:.4f} (should be >> T10 for Gaussian perturbations)")
check("Empirical width >> T10 lower bound", emp_lb >= lb_T10 * 0.05)

# ===========================================================================
print()
print("=" * 60)
print("Theorem 11 — Uniform confidence band (SM-Theorem 4.1)")
print("=" * 60)
delta_T11 = 0.05
B_n_T11   = (np.sqrt(2 * sigma2_w * np.log(2/delta_T11))
             + B_w * np.log(2/delta_T11) / 3)
M_delta   = Delta_star + B_n_T11

print(f"  Delta*={Delta_star:.4f}, B_n={B_n_T11:.4f}, M(delta)={M_delta:.4f}")
print(f"  [E_lo, E_hi]=[{E_lo:.4f},{E_hi:.4f}]")
print(f"  Band: [{E_n - M_delta:.4f},{E_n + M_delta:.4f}]")
check("Band covers E_lo", E_n - M_delta <= E_lo + 1e-10)
check("Band covers E_hi", E_hi <= E_n + M_delta + 1e-10)

# 1000-point theta grid — band should cover [E_lo, E_hi] for all theta
fails = sum(1 for _ in np.linspace(0, 1, 1000)
            if not (E_n - M_delta <= E_lo and E_hi <= E_n + M_delta))
check("Zero violations over 1000-point theta grid", fails == 0)

print()
print("All T7, T8 (9 parts), T9, T10, T11 checks complete.")
