"""Velocity fields. Each returns (N, d) given the particle cloud."""
import jax
import jax.numpy as jnp


def wasserstein_velocity(bundle):
    """v = -g_mu = -2 grad h_mu. This is KSD descent, our baseline."""
    @jax.jit
    def v(Xp):
        return -2.0 * jnp.mean(bundle["dK_pi"](Xp, Xp), axis=1)
    return v


def two_kernel_velocity(bundle, r):
    """v = -iota* g_mu, Definition def:two_kernel_geometry."""
    from .kernels import pairwise
    R = jax.jit(pairwise(r))

    @jax.jit
    def v(Xp):
        g = 2.0 * jnp.mean(bundle["dK_pi"](Xp, Xp), axis=1)      # (N, d)
        return -jnp.mean(R(Xp, Xp)[:, :, None] * g[None, :, :], axis=1)
    return v


def generator_velocity(bundle, psi_grad, stochastic=False):
    """v_alpha = -2 grad Psi_alpha, eqn:regularized_poisson_velocity."""
    if stochastic:
        def v(Xp, key):
            return -2.0 * psi_grad(Xp, key)
        return v

    @jax.jit
    def v(Xp):
        h = jnp.mean(bundle["K_pi"](Xp, Xp), axis=1)             # (N,)
        return -2.0 * psi_grad(Xp, h)
    return v


def integrate(v, X0, eta, n_steps, key=None, monitor=None, callback=None,
              stochastic=False):
    """Explicit Euler, the update analyzed in the end-to-end theorem.

    If `monitor` is given it is called as monitor(X) and must return the value of
    the objective. The configuration achieving the smallest monitored value is
    returned as well, which is the quantity the end-to-end bound controls.
    """
    X, hist = X0, []
    best_val, best_X, best_n = jnp.inf, X0, 0
    for n in range(n_steps):
        if stochastic:
            key, sub = jax.random.split(key)
            X = X + eta * v(X, sub)
        else:
            X = X + eta * v(X)
        if monitor is not None:
            val = float(monitor(X))
            if val < best_val:
                best_val, best_X, best_n = val, X, n + 1
        if callback is not None:
            hist.append(callback(n, X))
    return X, hist, (best_X, best_val, best_n)
