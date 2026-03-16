# IT2 Fuzzy Width: Cardiac Triage and Inspection Certified Decision Bounds

**Authors:** Renikunta Ramesh, Ramsha Mehreen  
**Affiliation:** Kakatiya Institute of Technology and Science, Warangal, India  
**Repository:** https://github.com/m24ds001/T2-Fuzzy-Width-Cardiac-Triage-and-Inspection-Certified-Decision-Bounds  
**Year:** 2026

---

## Theorem inventory (main paper — exact PDF numbering)

| # | Result | Core statement |
|---|--------|---------------|
| L1 | Sigmoid Lipschitz constant | |μᵢ(τ₁)−μᵢ(τ₂)| ≤ (kᵢ/4)|τ₁−τ₂| |
| L2 | Triangular MF Lipschitz constant | |μ△ᵢ(τ₁)−μ△ᵢ(τ₂)| ≤ L△ᵢ|τ₁−τ₂| |
| P1 | Corners are the worst case | Extrema of μ̄ᵢ−μᵢ at parameter box vertices |
| T1 | Aggregation interval bounds | E ≤ ∑wᵢμᵢ ≤ Ē |
| T2 | Width bound (Lipschitz perturbation) | Ē−E ≤ 2∑wᵢ(kᵢεc/4 + Mεk/4) |
| T3 | Certified O(1) width | Three cases; any normalised w gives Δ ≤ 2K' |
| C1 | Necessary and sufficient condition | Normalisation ⟺ O(1) width |
| T4 | IT2 width certificate (triangular MFs) | Ē−E ≤ 2L̄△ε |
| C2 | Mamdani rule-firing width bound | ΔMamdani ≤ 2L̄△ε + 2Λout·L△max·ε |
| T5 | Cascaded pipeline certificate | Depth-independent geometric bound |
| T6 | Width certificate under AR inputs | Δ∞ ≤ (kavgεc/2+M∞εk/2)/(1−κ₀) |
| T7 | Multi-threshold decision margin | Pr(certified) ≥ 1−KΔ*/g |
| T8 | Structural results (9 parts) | Monotonicity, Ω(n) growth, Yager λ, Bernstein, ... |
| T9 | Adaptive budget allocation | Unique minimum at single calibration vertex |
| T10 | Information-theoretic lower bound | Δ ≥ (kavg/4√2πe)·exp(mean entropy) |
| T11 | Uniform confidence band | Simultaneous Bernstein; saves log K Bonferroni nats |
| T12 | Global certifiability phase diagram | Φ = {kavgεc + Mεk < g} |
| T13 | Lipschitz width in inputs | |Δ(τ)−Δ(τ′)| ≤ (kavg/2)‖τ−τ′‖∞ |
| T14 | Sub-Gaussian width concentration | Exponential tail; tighter than Bernstein when νᵢ<εc/√3 |
| T15 | Optimal partition design | Min-gap formula; mass-allocation optimality |
| T16 | Consistency of corner-evaluation | SLLN + CLT + Hoeffding finite-sample band |

## SM cross-reference

| SM result | Main theorem |
|-----------|-------------|
| SM-Theorem 1.1 | Theorem 5 |
| SM-Theorem 2.1 | Theorem 8 (i)-(v),(vii)-(ix) |
| SM-Theorem 3.1 | Theorem 10 |
| SM-Theorem 4.1 | Theorem 11 |
| SM-Proposition 5.1 | Used in Theorem 6 |
| SM-Proposition 5.2 | Used in Theorem 6 |

## Usage

```bash
pip install numpy scipy
cd code
python run_all_validations.py
```

## Citation

```bibtex
@article{Ramesh2026IT2,
  title  = {IT2 Fuzzy Width: Cardiac Triage and Inspection Certified Decision Bounds},
  author = {Ramesh, Renikunta and Mehreen, Ramsha},
  journal= {International Journal of Fuzzy Systems},
  year   = {2026},
  note   = {Under review}
}
```
