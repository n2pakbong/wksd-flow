"""Experiment 2: the O(alpha^2) accuracy floor on a non-log-concave target.

Prediction: the plateau value of J scales like alpha^2, i.e. slope 2 on
log-log axes, and the plateau is reached at time O(alpha^{-2})
[Theorem kl_dissipation_regularized].
"""
import jax, numpy as np, matplotlib.pyplot as plt
from wksdflow.targets import mixture_target
from wksdflow.kernels import make_kernel_bundle, gaussian
from wksdflow.resolvent import make_collocation_solver
from wksdflow.flows import generator_velocity, integrate
from wksdflow.metrics import energy_U

DIM, N, EPS = 2, 300, 0.7
tgt = mixture_target(DIM, EPS, sep=2.0)
bundle = make_kernel_bundle("matern72", s=2.0, base_kwargs={"ell": 1.0},
                            grad_V=tgt.grad_V, eps=EPS)

alphas = np.logspace(-2.5, 0.5, 7)
plateaus = []
key = jax.random.PRNGKey(1)
X0 = 0.5 * jax.random.normal(key, (N, DIM))

for a in alphas:
    psi_grad = make_collocation_solver(lambda x, y: gaussian(x, y, 1.0),
                                       tgt.grad_V, EPS, a, 1e-6)
    v = generator_velocity(bundle, psi_grad)
    _, hist = integrate(v, X0, 2e-2, 6000,
                        callback=lambda n, X: float(energy_U(bundle["K_pi"], X)))
    plateaus.append(np.mean(hist[-500:]))

plt.loglog(alphas, plateaus, "o-", label="observed plateau")
plt.loglog(alphas, plateaus[0] * (alphas / alphas[0]) ** 2, "--", label=r"$\alpha^2$")
plt.xlabel(r"$\alpha$"); plt.ylabel(r"plateau of $\mathcal{J}$"); plt.legend()
plt.savefig("figures/exp2_floor.pdf")
