"""Experiment 3: baselines at matched cost, and the N-dependence of eqn end_to_end."""
import jax, jax.numpy as jnp, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wksdflow.config import Config
from wksdflow.diagnostics import report_step_size, collocation_diagnostics
from wksdflow.targets import logistic_posterior
from wksdflow.kernels import make_kernel_bundle, gaussian, pairwise
from wksdflow.resolvent import make_collocation_solver
from wksdflow.flows import (generator_velocity, wasserstein_velocity,
                            two_kernel_velocity, integrate)
from wksdflow.metrics import energy_U

cfg = Config.from_cli(dim=10, eps=1.0, n_data=200, base="imq", s=0.0,
                      alpha=1e-2, gamma=1e-8, eta=1e-2, n_steps=2000,
                      n_list="64,128,256,512", seed=2)

# ell ~ sqrt(dim): two independent draws satisfy |x-y|^2 ~ 2 d sigma^2, so the
# kernel must widen with dimension or K_pi becomes numerically diagonal.
ELL = cfg.ell if cfg.ell != 1.0 else float(np.sqrt(cfg.dim))
print(f"[diag] using ell = {ELL:.4g} (override with --ell)", flush=True)

tgt = logistic_posterior(jax.random.PRNGKey(cfg.seed), cfg.dim,
                         n_data=cfg.n_data, eps=cfg.eps)
bundle = make_kernel_bundle(cfg.base, s=cfg.s,
                            base_kwargs={"ell": ELL, **{k: v for k, v in
                                cfg.kernel_kwargs().items() if k != "ell"}},
                            grad_V=tgt.grad_V, eps=cfg.eps)
q = lambda x, y: gaussian(x, y, cfg.ell_q)


def svgd_velocity(grad_V, eps, ell):
    """phi(x_i) = (1/N) sum_j [ k(x_j,x_i) s(x_j) + grad_{x_j} k(x_j,x_i) ].
    k symmetric => grad_{x_j}k(x_j,x_i) = (grad_2 k)(x_i,x_j) = dK[i,j],
    so the repulsion sums over the SECOND index."""
    k = lambda x, y: gaussian(x, y, ell)
    K, dK = jax.jit(pairwise(k)), jax.jit(pairwise(jax.grad(k, argnums=1)))

    @jax.jit
    def v(Xp):
        s = -jax.vmap(grad_V)(Xp) / eps ** 2
        return (K(Xp, Xp) @ s + jnp.sum(dK(Xp, Xp), axis=1)) / Xp.shape[0]
    return v


def langevin_velocity(grad_V, eps, dt):
    """Euler-Maruyama as a velocity so the same integrator applies.
    dt MUST equal the step passed to integrate -- it is cfg.eta in both places."""
    def v(Xp, key):
        z = jax.random.normal(key, Xp.shape)
        return -jax.vmap(grad_V)(Xp) + jnp.sqrt(2.0 / dt) * eps * z
    return jax.jit(v)


results = {}
for N in cfg.n_values:
    print(f"\n=== N = {N} ===", flush=True)
    X0 = jax.random.normal(jax.random.PRNGKey(N), (N, cfg.dim))
    psi_grad = make_collocation_solver(q, tgt.grad_V, cfg.eps, cfg.alpha, cfg.gamma)
    if cfg.diag:
        collocation_diagnostics(q, tgt.grad_V, cfg.eps, cfg.alpha, cfg.gamma, X0)

    methods = {
        "gpsd":      (generator_velocity(bundle, psi_grad), False),
        "ksdd":      (wasserstein_velocity(bundle), False),
        "svgd":      (svgd_velocity(tgt.grad_V, cfg.eps, ELL), False),
        "langevin":  (langevin_velocity(tgt.grad_V, cfg.eps, cfg.eta), True),
        "twokernel": (two_kernel_velocity(
                          bundle, lambda x, y: gaussian(x, y, ELL)), False),
    }
    for name, (v, stoch) in methods.items():
        if cfg.diag and not stoch:
            report_step_size(f"N={N} {name}", v, X0, cfg.eta)
        _, hist = integrate(v, X0, cfg.eta, cfg.n_steps,
                            key=jax.random.PRNGKey(7), stochastic=stoch,
                            callback=lambda n, X: float(energy_U(bundle["K_pi"], X)))
        results[(N, name)] = np.asarray(hist)
        print(f"{N:5d} {name:10s} last={hist[-1]:.6e}  best={min(hist):.6e}",
              flush=True)

names = ["gpsd", "ksdd", "svgd", "langevin", "twokernel"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for nm in names:
    ax[0].loglog(cfg.n_values, [results[(N, nm)][-1] for N in cfg.n_values],
                 "o-", label=nm)
ref = results[(cfg.n_values[0], "gpsd")][-1]
ax[0].loglog(cfg.n_values,
             ref * (np.array(cfg.n_values, float) / cfg.n_values[0]) ** -0.5,
             "k--", label=r"$N^{-1/2}$")
ax[0].set_xlabel("N"); ax[0].set_ylabel(r"final $\mathcal{J}$"); ax[0].legend()
Nmax = cfg.n_values[-1]
for nm in names:
    h = results[(Nmax, nm)]
    ax[1].loglog(cfg.eta * np.arange(1, len(h) + 1), h, label=nm)
ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$\mathcal{J}$")
ax[1].set_title(f"N = {Nmax}"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(cfg.path("exp3_compare"))
print("wrote", cfg.path("exp3_compare"))
np.savez(cfg.path("exp3_compare").rsplit(".", 1)[0] + ".npz",
         **{f"{N}_{nm}": results[(N, nm)] for N in cfg.n_values for nm in names})
