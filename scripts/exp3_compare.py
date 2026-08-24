"""Experiment 3: baselines at matched cost, and the N-dependence of eqn:end_to_end."""
import jax, jax.numpy as jnp, numpy as np, matplotlib.pyplot as plt
from wksdflow.targets import logistic_posterior
from wksdflow.kernels import make_kernel_bundle, gaussian, pairwise
from wksdflow.resolvent import make_collocation_solver
from wksdflow.flows import (generator_velocity, wasserstein_velocity,
                            two_kernel_velocity, integrate)
from wksdflow.metrics import energy_U, energy_distance

DIM, EPS = 10, 1.0
key = jax.random.PRNGKey(2)
tgt = logistic_posterior(key, DIM, n_data=200, eps=EPS)
bundle = make_kernel_bundle("matern72", s=0.0, grad_V=tgt.grad_V, eps=EPS)

def svgd_velocity(grad_V, eps, ell=1.0):
    """phi(x_i) = (1/N) sum_j [ k(x_j,x_i) s(x_j) + grad_{x_j} k(x_j,x_i) ].
    With k symmetric, grad_{x_j} k(x_j,x_i) = (grad_2 k)(x_i,x_j) = dK[i,j],
    so the repulsion is a sum over the SECOND index.
    """
    k = lambda x, y: gaussian(x, y, ell)
    K = jax.jit(pairwise(k))
    dK = jax.jit(pairwise(jax.grad(k, argnums=1)))

    @jax.jit
    def v(Xp):
        s = -jax.vmap(grad_V)(Xp) / eps ** 2
        return (K(Xp, Xp) @ s + jnp.sum(dK(Xp, Xp), axis=1)) / Xp.shape[0]
    return v

def langevin_velocity(grad_V, eps):
    def v(Xp, key):
        z = jax.random.normal(key, Xp.shape)
        return -jax.vmap(grad_V)(Xp) + jnp.sqrt(2.0) * eps * z / jnp.sqrt(1e-2)
    return v

for N in [64, 128, 256, 512]:
    X0 = jax.random.normal(jax.random.PRNGKey(N), (N, DIM))
    psi_grad = make_collocation_solver(lambda x, y: gaussian(x, y, 1.0),
                                       tgt.grad_V, EPS, 1e-2, 1e-6)
    methods = {
        "gpsd": (generator_velocity(bundle, psi_grad), False),
        "ksdd": (wasserstein_velocity(bundle), False),
        "svgd": (svgd_velocity(tgt.grad_V, EPS), False),
        "langevin": (langevin_velocity(tgt.grad_V, EPS), True),
    }
    for name, (v, stoch) in methods.items():
        _, hist = integrate(v, X0, 1e-2, 2000, key=jax.random.PRNGKey(7),
                            stochastic=stoch,
                            callback=lambda n, X: float(energy_U(bundle["K_pi"], X)))
        print(N, name, hist[-1])
