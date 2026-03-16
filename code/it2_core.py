"""
it2_core.py — Core IT2 fuzzy aggregation primitives.

Implements the membership functions, corner-evaluation algorithm
(Algorithm 1), and closed-form width certificates (Theorems 2 and 3)
from the manuscript:

  "IT2 Fuzzy Width: Cardiac Triage and Inspection
   Certified Decision Bounds"

Theorem numbering follows the published PDF exactly:
  Lemma 1  — Sigmoid Lipschitz constant k/4
  Lemma 2  — Triangular MF Lipschitz constant
  Prop  1  — Corners are the worst case
  Theorem 1 — Aggregation interval bounds (sandwich)
  Theorem 2 — Width bound under Lipschitz perturbation
  Theorem 3 — Certified O(1) width (three cases)

Authors: Renikunta Ramesh, Ramsha Mehreen
Repository: https://github.com/m24ds001/T2-Fuzzy-Width-Cardiac-Triage-and-Inspection-Certified-Decision-Bounds
Year: 2026
"""

import numpy as np


# ---------------------------------------------------------------------------
# Membership functions (Definition 2)
# ---------------------------------------------------------------------------

def sigmoid_mf(tau, c, k):
    """Sigmoid MF: mu(tau; c, k) = 1 / (1 + exp(-k*(tau - c)))."""
    return 1.0 / (1.0 + np.exp(-k * (tau - c)))


def triangular_mf(tau, a, b, c):
    """Triangular MF with left-foot a, apex b, right-foot c."""
    left  = (tau - a) / (b - a) if b > a else 0.0
    right = (c - tau) / (c - b) if c > b else 0.0
    return float(np.clip(min(left, right), 0.0, 1.0))


def sigmoid_lipschitz(k):
    """Lipschitz constant for sigmoid MF (Lemma 1): k/4."""
    return k / 4.0


def triangular_lipschitz(a, b, c):
    """Lipschitz constant for triangular MF (Lemma 2): max(1/(b-a), 1/(c-b))."""
    return max(1.0 / (b - a), 1.0 / (c - b))


# ---------------------------------------------------------------------------
# Corner-evaluation algorithm (Algorithm 1 / Proposition 1)
# ---------------------------------------------------------------------------

def corner_eval_sigmoid(tau_vec, c_bounds, k_bounds, weights):
    """
    Corner-Evaluation Width Certificate (Algorithm 1) for sigmoid MFs.
    Implements Proposition 1: extrema of mu_i - mu_i occur at box corners.

    Parameters
    ----------
    tau_vec   : array (n,)   — input values
    c_bounds  : array (n,2)  — [[c_lo, c_hi], ...] centre intervals
    k_bounds  : array (n,2)  — [[k_lo, k_hi], ...] steepness intervals
    weights   : array (n,)   — normalised weights (must sum to 1)

    Returns
    -------
    E_min, E_max, delta : floats   (delta = E_max - E_min)
    """
    n = len(tau_vec)
    E_min = 0.0
    E_max = 0.0
    for i in range(n):
        tau = tau_vec[i]
        corners = [
            sigmoid_mf(tau, c_bounds[i, 0], k_bounds[i, 0]),
            sigmoid_mf(tau, c_bounds[i, 0], k_bounds[i, 1]),
            sigmoid_mf(tau, c_bounds[i, 1], k_bounds[i, 0]),
            sigmoid_mf(tau, c_bounds[i, 1], k_bounds[i, 1]),
        ]
        E_min += weights[i] * min(corners)
        E_max += weights[i] * max(corners)
    return E_min, E_max, E_max - E_min


def corner_eval_triangular(tau_vec, a_bounds, b_bounds, c_bounds, weights):
    """Corner-Evaluation for triangular MFs (d=3, 2^3=8 corners per source)."""
    n = len(tau_vec)
    E_min = 0.0
    E_max = 0.0
    for i in range(n):
        tau = tau_vec[i]
        corners = [
            triangular_mf(tau, av, bv, cv)
            for av in a_bounds[i]
            for bv in b_bounds[i]
            for cv in c_bounds[i]
        ]
        E_min += weights[i] * min(corners)
        E_max += weights[i] * max(corners)
    return E_min, E_max, E_max - E_min


# ---------------------------------------------------------------------------
# Closed-form width certificates
# ---------------------------------------------------------------------------

def certificate_T2(k_vec, weights, eps_c, M, eps_k):
    """
    Theorem 2 upper certificate:
        Delta <= 2 * sum_i w_i * (k_i/4 * eps_c + M/4 * eps_k)
    """
    per_source = k_vec / 4.0 * eps_c + M / 4.0 * eps_k
    return 2.0 * float(np.dot(weights, per_source))


def certificate_T3_iii(k_vec, eps_c, M, eps_k):
    """
    Theorem 3(iii) certificate: 2*K'  where K' = max_i(k_i*eps_c/4 + M*eps_k/4).
    Valid for any normalised weight vector.
    """
    K_prime = np.max(k_vec / 4.0 * eps_c + M / 4.0 * eps_k)
    return 2.0 * K_prime


def certificate_T4(L_bar_tri, eps):
    """Theorem 4 certificate for triangular MFs: 2 * L_bar * eps."""
    return 2.0 * L_bar_tri * eps


# ---------------------------------------------------------------------------
# Monte Carlo empirical width (used as ground truth in validation)
# ---------------------------------------------------------------------------

def mc_width_sigmoid(tau_vec, c_centres, k_centres, eps_c, eps_k,
                     weights, N=10000, seed=42):
    """
    Monte Carlo estimate of IT2 interval width.
    Samples (c_i, k_i) uniformly from parameter rectangles.
    """
    rng = np.random.default_rng(seed)
    n = len(tau_vec)
    all_E = []
    for _ in range(N):
        c_s = c_centres + rng.uniform(-eps_c, eps_c, n)
        k_s = np.clip(k_centres + rng.uniform(-eps_k, eps_k, n), 0.01, None)
        E = sum(weights[i] * sigmoid_mf(tau_vec[i], c_s[i], k_s[i])
                for i in range(n))
        all_E.append(E)
    return max(all_E) - min(all_E)
