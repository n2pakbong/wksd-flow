# Parameter reference

pi ∝ exp(-V/eps^2).  (L f)(x) = -<grad V, grad f> + eps^2 Laplacian f.
J = squared weighted KSD (metrics.energy_U).  All flags live in wksdflow/config.py.
Print of `config:` at the top of each run is the reproducibility record.

| flag | paper symbol | role | how to choose | failure mode |
|---|---|---|---|---|
| `--eps` | eps | temperature; coefficient of the Laplacian in `generator`; k_pi scales like eps^4 | fixed by the problem | small eps concentrates pi, drives rho very negative, forces large alpha |
| `--ell` | ell | base-kernel length scale | comparable to the spread of pi; in d dims |x-y|^2 ~ 2 d sigma^2, hence ell ∝ sqrt(d) (the only reason for ell=sqrt(dim) in exp3) | too small: K_pi numerically diagonal, all methods look identical, J meaningless. Too large: k_pi underweights fine structure, flow stalls |
| `--s` | s in w(x)=(1+|x|^2)^{-s/2} | the weighting of Prop. admissible_class (`weighted()`) | s>0 only when grad V grows, to keep k_pi bounded. s=0 for Gaussian/logistic, s=2 for the mixture | too large flattens k_pi in the tails; particles stop being pushed back |
| `--delta` | delta | shift inside sqrt for Matern: k_delta(x,y)=f(|x-y|^2+delta) | 1e-6; validate with `python scripts/check_delta.py` | 0 gives NaN (see below); too large biases k_pi by O(delta^{3/2}) |
| `--ell-q` | - | length scale of the collocation kernel q (separate from the flow kernel) | 1.0; only affects conditioning of M | too small: M near-diagonal, ridge dominates |
| `--alpha` | alpha | shift in (-L + alpha) Psi = h_mu (`make_collocation_solver`) | CONSTRAINED: Assumption semiconvexity needs lambda_alpha = alpha + min(rho,2rho) > 0, i.e. alpha > -2 rho when rho<0. Check `tgt.rho`. Then Thm kl_dissipation_regularized: floor is O(alpha^2), time to floor O(alpha^-2) => pick alpha just above threshold | below threshold: theory says nothing, flow may diverge or converge to the wrong thing. Very large: Psi ≈ h_mu/alpha, i.e. the Wasserstein flow / alpha, floor immediately visible |
| `--gamma` | gamma | Tikhonov ridge on M in Algorithm 2 | as small as conditioning allows; M is NOT symmetric and cond(M) grows as alpha decreases. Use `collocation_diagnostics` (want gamma/s_min << 1). float64: 1e-8 usually safe; 1e-6 already biases exp1 | too small: garbage coefficients, velocity blows up in one step. Too large: different equation; exp1's KL identity fails visibly |
| `--n` | N | particles | Thm end_to_end has an O(N^{-1/2}) Monte-Carlo term => N sets the floor of J. Cost O(N^2) kernel + O(N^3) solve per step | too small: exp1's slope -1 saturates at the statistical floor before the 1/t regime appears |
| `--eta` | eta | explicit Euler step | there is a bound eta <= lambda_alpha/(4 kappa_1^2) but kappa_1 is not computable; use `report_step_size` and keep eta*L <= 0.5 | too large: NaN / oscillation in a few steps. Too small: exp2 never reaches the plateau, exp1 is all transient |
| `--n-steps` | T = eta*n_steps | horizon | exp1: T must exceed the initial relaxation time for J ≈ 2 eps^2 D / t to be asymptotic. exp2: T > alpha_min^-2, else `mean(hist[-window:])` averages a decaying curve and the exponent is spurious (exp2 warns) | silent wrong exponent |
| `--plateau-window` | - | exp2: number of final steps averaged | keep small relative to n_steps; verify flatness in the right-hand panel | includes transient |
| `--n-list` | - | exp3 particle counts | halve for a fast pass | - |
| `--fmt`, `--tag`, `--outdir`, `--diag` | - | output control | `--fmt png` previews in Codespaces; `--tag _x` avoids overwriting runs | - |

## Why Matern needs delta (and why it is not a hack)

With u=|x-y|^2, Matern-7/2 expands as f(u)=c0+c1 u+c2 u^2+c3 u^3+c4 u^{7/2}+...
The kernel is C^6 in x-y and k_pi = L_x L_y k needs only 4 derivatives, so k_pi
is bounded on the diagonal. But f''''(u) ~ u^{-1/2} diverges, and in the chain
rule it appears multiplied by u^2; the product is u^{3/2} -> 0. The infinities
cancel mathematically; autodiff evaluates the factors separately, gets inf*0,
returns NaN. Fix: k_delta(x,y) = f(|x-y|^2 + delta).

By Schoenberg's theorem, a radial function PD on every R^d has f completely
monotone; Matern is PD on every R^d; complete monotonicity is preserved by
shifting the argument. So k_delta is itself a positive definite kernel with the
same tail decay, satisfying Prop. admissible_class verbatim, and C^infty since
its argument stays >= delta. This is not an approximation of Matern -- it is a
different, equally legitimate member of the admissible class, converging
pointwise to Matern as delta -> 0. One sentence in the numerics section suffices.

## Precision

Two mechanisms cost digits: (a) cancellation in forming L_x L_y k, a difference
of O(ell^-4) terms that nearly annihilate at moderate |x-y|; (b) cond(M).
Both are measurable: run tests/test_stein_identity.py (which checks
E_pi[k_pi(X,y)] = 0) with and without float64. Expect ~1e-10 vs ~1e-3. That
comparison, not an assertion, is the reason for float64.

## Run order
python -m pytest -q tests/test_stein_identity.py # correctness gate
python scripts/check_delta.py --base matern72 # only if using Matern
python exp1_rates.py --n 32 --n-steps 100 --tag _smoke
python exp2_floor.py --n 32 --n-steps 200 --n-alpha 3 --tag _smoke
python exp3_compare.py --n-list 32 --n-steps 100 --tag _smoke
python exp1_rates.py ; python exp2_floor.py ; python exp3_compare.py

Cost: per step, one NxN matrix of k_pi (fourth-order autodiff per entry) plus one
dense non-symmetric NxN solve. Doubling N: x4 on the first, x8 on the second.
n_steps is linear. To shrink, halve n_steps and raise eta by the same factor to
keep T fixed, then confirm eta*L <= 0.5.
