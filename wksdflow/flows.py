"""Velocity fields and the explicit Euler integrator of Theorem (end to end)."""
import jax, jax.numpy as jnp
from .kernels import pairwise
jax.config.update("jax_enable_x64", True)


def wksd_residual(K_pi, Xp):
    """h_i = (1/N) sum_j k_pi(X_i, X_j), the stationarity residual at particles."""
    return jnp.mean(K_pi(Xp, Xp), axis=1)


def generator_velocity(bundle, psi_grad):
    """v = -2 grad Psi with (-L + alpha) Psi = h_mu, i.e. Algorithm 2."""
    K_pi = bundle["K_pi"]

    @jax.jit
    def v(Xp):
        return -2.0 * psi_grad(Xp, wksd_residual(K_pi, Xp))
    return v


def generator_velocity_stochastic(bundle, psi_grad_sg):
    """Same field with the semigroup estimator, which needs a PRNG key."""
    @jax.jit
    def v(Xp, key):
        return -2.0 * psi_grad_sg(Xp, key)
    return v


def wasserstein_velocity(bundle):
    """v = -g_mu = -2 grad h_mu, kernel Stein discrepancy descent."""
    dK_pi = bundle["dK_pi"]

    @jax.jit
    def v(Xp):
        return -2.0 * jnp.mean(dK_pi(Xp, Xp), axis=1)
    return v


def two_kernel_velocity(bundle, r):
    """v(x) = -(1/N) sum_j r(x, X_j) g_mu(X_j), the RKHS geometry of
    Definition (two kernel geometry) with geometry kernel r."""
    dK_pi = bundle["dK_pi"]
    R = jax.jit(pairwise(r))

    @jax.jit
    def v(Xp):
        G = 2.0 * jnp.mean(dK_pi(Xp, Xp), axis=1)
        return -R(Xp, Xp) @ G / Xp.shape[0]
    return v


def integrate(v, X0, eta, n_steps, key=None, callback=None, stochastic=False):
    """Explicit Euler, the update analyzed in Theorem (end to end).

    Returns (X_final, hist), where hist collects callback(n, X) after each step.
    The best-iterate quantity of the theorem is min over hist of the recorded
    objective. If `stochastic` is True then v is called as v(X, subkey).
    """
    X, hist = X0, []
    for n in range(n_steps):
        if stochastic:
            key, sub = jax.random.split(key)
            X = X + eta * v(X, sub)
        else:
            X = X + eta * v(X)
        if callback is not None:
            hist.append(callback(n, X))
    return X, hist
