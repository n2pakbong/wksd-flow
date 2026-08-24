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


def integrate(v, X0, eta, n_steps, key=None, callback=None, stochastic=False):
    """Explicit Euler, exactly the update analyzed in Theorem thm:end_to_end."""
    X = X0
    hist = []
    for n in range(n_steps):
        if stochastic:
            key, sub = jax.random.split(key)
            X = X + eta * v(X, sub)
        else:
            X = X + eta * v(X)
        if callback is not None:
            hist.append(callback(n, X))
    return X, hist
