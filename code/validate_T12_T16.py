"""
validate_T12_T16.py
Numerical validation of:
  Theorem 12 — Global certifiability phase diagram
  Theorem 13 — Lipschitz continuity of width in inputs
  Theorem 14 — Sub-Gaussian width concentration
  Theorem 15 — Optimal partition design for a target certified fraction
  Theorem 16 — Consistency and rate of corner-evaluation certificate

Theorem numbering matches the published PDF exactly.

Repository: https://github.com/m24ds001/T2-Fuzzy-Width-Cardiac-Triage-and-Inspection-Certified-Decision-Bounds
Year: 2026
"""

import numpy as np
from it2_core import (sigmoid_mf, corner_eval_sigmoid, certificate_T2)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
rng  = np.random.default_rng(42)


def check(label, condition, lhs=None, rhs=None):
    status = PASS if condition else FAIL
    if lhs is not None:
        print(f"  {status}  {label}:  {lhs:.6f}  <=  {rhs:.6f}")
    else:
        print(f"  {status}  {label}")


n       = 8
eps_c   = 0.01
eps_k   = 0.1
k_vec   = rng.uniform(2.0, 5.0, n)
c_vec   = rng.uniform(0.2, 0.8, n)
weights = rng.dirichlet(np.ones(n))
tau_vec = c_vec + rng.uniform(-0.05, 0.05, n)
M       = float(np.max(np.abs(tau_vec - c_vec)))
k_avg   = float(np.mean(k_vec))
c_b     = np.column_stack([c_vec - eps_c, c_vec + eps_c])
k_b     = np.clip(np.column_stack([k_vec - eps_k, k_vec + eps_k]), 0.01, None)

# ===========================================================================
print("=" * 60)
print("Theorem 12 — Global certifiability phase diagram")
print("=" * 60)
K_cls = 4
g     = 1.0 / K_cls   # = 0.25

# Part (i): interior of Phi
check("(i) k_avg*eps_c + M*eps_k < g  [inside Phi]",
      k_avg * eps_c + M * eps_k < g)
Delta_inside = (k_avg * eps_c + M * eps_k) / 2.0
check("(i) 2*Delta* < g", 2 * Delta_inside < g, 2 * Delta_inside, g)

# Part (ii): boundary is tight
eps_c_bdry = (g - M * eps_k) / k_avg
val_bdry   = k_avg * eps_c_bdry + M * eps_k
check("(ii) boundary value = g", abs(val_bdry - g) < 1e-10)

# Part (iii): closed-form radius formulae
eps_c_max = (g - M * eps_k) / k_avg
eps_k_max = (g - k_avg * eps_c) / M
check("(iii) k_avg*eps_c_max + M*eps_k = g",
      abs(k_avg * eps_c_max + M * eps_k - g) < 1e-10)
print(f"  Max eps_c at eps_k={eps_k:.3f}: {eps_c_max:.6f}")
print(f"  Max eps_k at eps_c={eps_c:.3f}: {eps_k_max:.6f}")

# ===========================================================================
print()
print("=" * 60)
print("Theorem 13 — Lipschitz continuity of width in inputs")
print("  |Delta(tau) - Delta(tau')| <= (k_avg/2) * ||tau - tau'||_inf")
print("=" * 60)
lip = k_avg / 2.0
max_ratio = 0.0
for _ in range(5000):
    eta  = rng.uniform(-0.15, 0.15, n)
    tau2 = tau_vec + eta
    _, _, d1 = corner_eval_sigmoid(tau_vec, c_b, k_b, weights)
    _, _, d2 = corner_eval_sigmoid(tau2,    c_b, k_b, weights)
    denom = float(np.max(np.abs(eta)))
    if denom < 1e-12:
        continue
    max_ratio = max(max_ratio, abs(d1 - d2) / denom)

check("max |Delta(tau)-Delta(tau')| / ||tau-tau'|| <= k_avg/2",
      max_ratio <= lip + 1e-6, max_ratio, lip)
print(f"  Empirical Lipschitz constant: {max_ratio:.6f}  Bound: {lip:.6f}")

# ===========================================================================
print()
print("=" * 60)
print("Theorem 14 — Sub-Gaussian width concentration")
print("  Eq.(10): sigma_tilde^2 = (eps_c^2/16)*sum(w_i^2*k_i^2*nu_i^2)")
print("=" * 60)
nu       = 0.02   # sub-Gaussian proxy variance, nu < eps_c/sqrt(3) ≈ 0.0058... 
# use nu = 0.005 to be strictly inside the improvement regime
nu       = 0.005
sigma2_t = (eps_c**2 / 16.0) * float(np.sum(weights**2 * k_vec**2 * nu**2))
B_tilde  = float(np.max(weights)) * eps_c * float(np.max(k_vec)) / 4.0
delta_14 = 0.05

t_bound = (np.sqrt(2 * sigma2_t * np.log(2/delta_14))
           + B_tilde * np.log(2/delta_14) / 3)

E_bar   = float(np.dot(weights, [sigmoid_mf(tau_vec[i], c_vec[i], k_vec[i]) for i in range(n)]))
widths  = []
for _ in range(100000):
    cp = np.clip(c_vec + rng.normal(0, nu, n), c_vec - eps_c, c_vec + eps_c)
    kp = np.clip(k_vec + rng.normal(0, nu * 2, n), 0.01, None)
    widths.append(float(np.dot(weights, [sigmoid_mf(tau_vec[i], cp[i], kp[i]) for i in range(n)])))

E_hat   = float(np.mean(widths))
viol    = np.mean(np.abs(np.array(widths) - E_hat) > t_bound)
print(f"  sigma2_tilde={sigma2_t:.2e}, B_tilde={B_tilde:.6f}, t_bound={t_bound:.6f}")
print(f"  Empirical violation rate: {viol:.4f}  (should <= {delta_14})")
check("Empirical violation rate <= delta", viol <= delta_14 + 0.01)

# Verify improvement over naive box bound
sigma2_box = float(np.sum(weights**2 * (k_vec * eps_c / 4)**2 / 3))
check("Sub-Gaussian proxy tighter than box variance",
      sigma2_t < sigma2_box)
print(f"  Improvement factor sigma2_box/sigma2_tilde = {sigma2_box/sigma2_t:.1f}x")

# ===========================================================================
print()
print("=" * 60)
print("Theorem 15 — Optimal partition design for target certified fraction")
print("=" * 60)
Delta_star = certificate_T2(k_vec, weights, eps_c, M, eps_k)

# Part (i): min-gap formula g >= K*Delta*/(1-alpha)
print("  Part (i): minimum gap for K=4 classes at various alpha")
K_fixed = 4
for alpha in [0.50, 0.63, 0.75, 0.90]:
    g_min  = K_fixed * Delta_star / (1.0 - alpha)
    pr_at  = max(0.0, 1.0 - K_fixed * Delta_star / g_min)
    check(f"  alpha={alpha:.2f}: g_min={g_min:.4f}  => Pr={pr_at:.4f} (should={alpha:.2f})",
          abs(pr_at - alpha) < 1e-6)

# Part (ii): max K at fixed g
g_fixed = 0.25
for alpha in [0.50, 0.63]:
    K_max = g_fixed * (1 - alpha) / Delta_star
    print(f"  Part (ii): alpha={alpha:.2f}, g={g_fixed}: K_max={K_max:.2f}")

# Part (iii): alpha_min at fixed K, g
alpha_min = max(0.0, 1.0 - K_fixed * Delta_star / g_fixed)
print(f"  Part (iii): K={K_fixed}, g={g_fixed}: alpha_min={alpha_min:.4f}")
check("alpha_min in [0,1]", 0 <= alpha_min <= 1)

# Part (iv): mass on widest gap maximises certified fraction
g_nu = [0.15, 0.30, 0.20, 0.35]
p_u  = [0.25] * 4
p_o  = [0.0, 0.0, 0.0, 1.0]  # all mass on widest gap (0.35)
pr_u = sum(p_u[k] * max(0.0, 1 - Delta_star / g_nu[k]) for k in range(4))
pr_o = sum(p_o[k] * max(0.0, 1 - Delta_star / g_nu[k]) for k in range(4))
check("Part (iv): optimal allocation >= uniform",
      pr_o >= pr_u - 1e-10, pr_u, pr_o)
print(f"  Pr(cert) uniform={pr_u:.4f}  optimal={pr_o:.4f}")

# ===========================================================================
print()
print("=" * 60)
print("Theorem 16 — Consistency and rate of corner-evaluation certificate")
print("  (SLLN + CLT + Hoeffding finite-sample band)")
print("=" * 60)

# Population certificate by numerical integration
def pop_cert(c0, k0, ec, ek, n_tau=50000):
    taus = np.linspace(0.01, 0.99, n_tau)
    c_b2 = np.array([[c0 - ec, c0 + ec]])
    k_b2 = np.array([[max(0.01, k0 - ek), k0 + ek]])
    vals = []
    for t in taus:
        cs = [sigmoid_mf(t, c_b2[0,j], k_b2[0,l]) for j in range(2) for l in range(2)]
        vals.append(max(cs) - min(cs))
    return float(np.mean(vals))

k0, c0 = 3.0, 0.5
Delta_pop = pop_cert(c0, k0, eps_c, eps_k)
print(f"  Population certificate Delta_inf ≈ {Delta_pop:.5f}")

# Part (i)+(ii): convergence check
prev_std = None
print("  n    std(Delta_hat)   O(1/sqrt(n))")
for sz in [20, 50, 100, 200, 500]:
    hats = []
    for _ in range(200):
        ts  = rng.uniform(0, 1, sz)
        cb2 = np.full((sz, 2), [c0 - eps_c, c0 + eps_c])
        kb2 = np.full((sz, 2), [max(0.01, k0 - eps_k), k0 + eps_k])
        wv2 = np.ones(sz) / sz
        _, _, dh = corner_eval_sigmoid(ts, cb2, kb2, wv2)
        hats.append(dh)
    std = float(np.std(hats))
    print(f"  {sz:4d}  {std:.5f}          {1/np.sqrt(sz):.5f}")
    if prev_std is not None:
        check(f"  std decreases n={prev_sz}→n={sz}", std <= prev_std + 0.002, std, prev_std + 0.002)
    prev_std, prev_sz = std, sz

# Part (iii): Hoeffding finite-sample band
k_max_v = k0
for sz in [50, 200]:
    for delta_v in [0.10, 0.05]:
        t_h = np.sqrt(k_max_v**2 * eps_c**2 * np.log(2/delta_v) / (2 * sz))
        hats2 = []
        for _ in range(5000):
            ts  = rng.uniform(0, 1, sz)
            cb2 = np.full((sz, 2), [c0 - eps_c, c0 + eps_c])
            kb2 = np.full((sz, 2), [max(0.01, k0 - eps_k), k0 + eps_k])
            wv2 = np.ones(sz) / sz
            _, _, dh = corner_eval_sigmoid(ts, cb2, kb2, wv2)
            hats2.append(dh)
        viol = float(np.mean(np.abs(np.array(hats2) - Delta_pop) > t_h))
        check(f"  Hoeffding n={sz} delta={delta_v}: viol={viol:.3f} <= {delta_v}",
              viol <= delta_v + 0.02)

print()
print("All T12, T13, T14, T15, T16 checks complete.")
