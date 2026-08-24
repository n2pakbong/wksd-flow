"""Experiment 1: verify the exact KL identity and the two rates.

Predictions tested:
  (i)  d/dt KL(mu_t | pi) = -(2/eps^2) J(mu_t)        [Theorem kl_dissipation]
  (ii) (1/t) int_0^t J ds <= eps^2 KL_0 / (2t)        [eqn global_rate]
  (iii) J(mu_t) <= 2 eps^2 D / t                      [Theorem nonlinear_local_rate]
"""
import jax, jax.numpy as jnp, numpy as np
import matplotlib.pyplot as plt

from wksdflow.targets import gaussian_target
from wksdflow.kernels import make_kernel_bundle, gaussian
from wksdflow.resolvent import make_collocation_solver
from wksdflow.flows import generator_velocity, wasserstein_velocity, integrate
from wksdflow.metrics import energy_U, kl_gaussian

DIM, N, EPS, SCALE = 1, 200, 1.0, 1.0
ALPHA, GAMMA, ETA, NSTEPS = 1e-3, 1e-6, 5e-2, 4000

tgt = gaussian_target(DIM, EPS, SCALE)
bundle = make_kernel_bundle("matern72", s=0.0, base_kwargs={"ell": 1.0},
                            grad_V=tgt.grad_V, eps=EPS)
psi_grad = make_collocation_solver(lambda x, y: gaussian(x, y, 1.0),
                                   tgt.grad_V, EPS, ALPHA, GAMMA)

key = jax.random.PRNGKey(0)
# start off-target: shifted and over-dispersed
X0 = 1.5 * jax.random.normal(key, (N, DIM)) + 1.0

for name, v in [("generator", generator_velocity(bundle, psi_grad)),
                ("wasserstein", wasserstein_velocity(bundle))]:
    rec = []
    def cb(n, X):
        return (n * ETA, float(energy_U(bundle["K_pi"], X)),
                float(kl_gaussian(X, EPS, SCALE)))
    _, hist = integrate(v, X0, ETA, NSTEPS, callback=cb)
    t, J, KL = map(np.array, zip(*hist))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    # (i) the identity
    dKL = np.gradient(KL, t)
    ax[0].plot(t[10:], -dKL[10:], label=r"$-\frac{d}{dt}\mathrm{KL}$")
    ax[0].plot(t[10:], 2 / EPS**2 * J[10:], "--", label=r"$2\varepsilon^{-2}\mathcal{J}$")
    ax[0].set_xlabel("t"); ax[0].legend(); ax[0].set_title(f"{name}: KL identity")
    # (ii) time average
    cum = np.cumsum(J) * ETA / np.maximum(t, ETA)
    ax[1].loglog(t[1:], cum[1:], label="time average of J")
    ax[1].loglog(t[1:], EPS**2 * KL[0] / (2 * t[1:]), "--", label="bound")
    ax[1].legend(); ax[1].set_xlabel("t")
    # (iii) pointwise
    ax[2].loglog(t[1:], J[1:], label="J(mu_t)")
    ax[2].loglog(t[1:], J[1] * t[1] / t[1:], "--", label="slope -1")
    ax[2].legend(); ax[2].set_xlabel("t")
    fig.tight_layout(); fig.savefig(f"figures/exp1_{name}.pdf")
