"""Experiment 2: the O(alpha^2) accuracy floor on a non-log-concave target.

Prediction [Thm kl_dissipation_regularized]: the plateau of J scales like
alpha^2 (slope 2 on log-log axes) and is reached at time O(alpha^{-2}).

Why the defaults changed: with sep=2.0, eps=0.7 the mixture has
rho = 1 - sep^2/eps^2 ~= -7.2, so admissibility needs alpha > 14.4 -- the old
sweep (1e-2.5 .. 1e0.5) lay entirely OUTSIDE the hypotheses of the theorem it
was meant to verify. sep=0.9 gives rho ~= -0.65, threshold 1.3, so the sweep
[1.5, 30] is admissible throughout. Alternative: --sep 2.0 --alpha-min 15
--alpha-max 300. The assertion below refuses to run outside the hypotheses.
"""
import jax, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wksdflow.config import Config
from wksdflow.diagnostics import (report_step_size, collocation_diagnostics,
                                  check_admissible)
from wksdflow.targets import mixture_target
from wksdflow.kernels import make_kernel_bundle, gaussian
from wksdflow.resolvent import make_collocation_solver
from wksdflow.flows import generator_velocity, integrate
from wksdflow.metrics import energy_U

cfg = Config.from_cli(dim=2, n=300, eps=0.7, sep=0.9, base="imq", s=2.0,
                      ell=1.0, eta=2e-2, n_steps=6000, gamma=1e-8,
                      alpha_min=1.5, alpha_max=30.0, n_alpha=7, seed=1)

tgt = mixture_target(cfg.dim, cfg.eps, sep=cfg.sep)
check_admissible(cfg.alpha_min, tgt.rho, "alpha_min")

bundle = make_kernel_bundle(cfg.base, s=cfg.s, base_kwargs=cfg.kernel_kwargs(),
                            grad_V=tgt.grad_V, eps=cfg.eps)
q = lambda x, y: gaussian(x, y, cfg.ell_q)

alphas = np.logspace(np.log10(cfg.alpha_min), np.log10(cfg.alpha_max),
                     cfg.n_alpha)
X0 = 0.5 * jax.random.normal(jax.random.PRNGKey(cfg.seed), (cfg.n, cfg.dim))

plateaus, curves = [], []
for a in alphas:
    print(f"\n=== alpha = {a:.4g} ===", flush=True)
    # plateau time is O(alpha^{-2}); warn if the horizon cannot reach it, else
    # mean(hist[-window:]) averages a still-decaying curve and the exponent is bogus.
    T = cfg.eta * cfg.n_steps
    if T < 1.0 / a ** 2:
        print(f"[warn] T={T:.3g} < alpha^-2={1/a**2:.3g}: plateau may not be "
              f"reached; raise --n-steps", flush=True)

    psi_grad = make_collocation_solver(q, tgt.grad_V, cfg.eps, a, cfg.gamma)
    v = generator_velocity(bundle, psi_grad)
    if cfg.diag:
        collocation_diagnostics(q, tgt.grad_V, cfg.eps, a, cfg.gamma, X0)
        report_step_size(f"alpha={a:.3g}", v, X0, cfg.eta)

    _, hist = integrate(v, X0, cfg.eta, cfg.n_steps,
                        callback=lambda n, X: float(energy_U(bundle["K_pi"], X)))
    hist = np.asarray(hist)
    w = min(cfg.plateau_window, len(hist))
    plateaus.append(float(hist[-w:].mean()))
    curves.append(hist)
    print(f"plateau(mean of last {w}) = {plateaus[-1]:.6e}", flush=True)

plateaus = np.asarray(plateaus)
slope = np.polyfit(np.log(alphas), np.log(plateaus), 1)[0]
print(f"\nfitted log-log slope = {slope:.3f}  (theory: 2)", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].loglog(alphas, plateaus, "o-", label="observed plateau")
ax[0].loglog(alphas, plateaus[0] * (alphas / alphas[0]) ** 2, "--",
             label=r"$\alpha^2$ reference")
ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel(r"plateau of $\mathcal{J}$")
ax[0].set_title(f"slope {slope:.2f} (theory 2)"); ax[0].legend()
for a, h in zip(alphas, curves):
    ax[1].loglog(cfg.eta * np.arange(1, len(h) + 1), h, label=rf"$\alpha$={a:.2g}")
ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\mathcal{J}(\mu_t)$")
ax[1].set_title("check each curve is flat at the right edge")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(cfg.path("exp2_floor"))
print("wrote", cfg.path("exp2_floor"))
np.savez(cfg.path("exp2_floor").rsplit(".", 1)[0] + ".npz",
         alphas=alphas, plateaus=plateaus, slope=slope)
