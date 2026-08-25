import jax, jax.numpy as jnp
from .kernels import pairwise
jax.config.update("jax_enable_x64", True)

def wksd_residual(K_pi, Xp):
    """h_i = (1/N) sum_j k_pi(X_i, X_j)."""
    return jnp.mean(K_pi(Xp, Xp), axis=1)

def wasserstein_velocity(dK_pi):
    """v = -g_mu = -2 grad h_mu, the Wasserstein flow of the same objective."""
    @jax.jit
    def v(Xp):
        return -2.0 * jnp.mean(dK_pi(Xp, Xp), axis=1)
    return v

def svgd_velocity(k, grad_V, eps):
    """phi(x_i) = (1/N) sum_j [ k(x_j,x_i) s(x_j) + grad_{x_j} k(x_j,x_i) ].
    With k symmetric, grad_{x_j} k(x_j,x_i) = (grad_2 k)(x_i,x_j), so the
    repulsion is a sum over the SECOND index of dK."""
    K = jax.jit(pairwise(k))
    dK = jax.jit(pairwise(jax.grad(k, argnums=1)))

    @jax.jit
    def v(Xp):
        s = -jax.vmap(grad_V)(Xp) / eps ** 2
        return (K(Xp, Xp) @ s + jnp.sum(dK(Xp, Xp), axis=1)) / Xp.shape[0]
    return v

def langevin_velocity(grad_V, eps, dt):
    """Euler-Maruyama written as a stochastic velocity, so that the caller can
    use the same integrator. `dt` MUST equal the step size passed to integrate."""
    def v(Xp, key):
        z = jax.random.normal(key, Xp.shape)
        return -jax.vmap(grad_V)(Xp) + jnp.sqrt(2.0 / dt) * eps * z
    return jax.jit(v)

def integrate(v, X0, eta, n_steps, key=None, monitor=None, callback=None,
              stochastic=False):
    """Explicit Euler, the update analyzed in the end-to-end theorem. If
    `monitor` is given it is called as monitor(X) and must return the objective,
    and the configuration achieving the smallest monitored value is returned as
    well, this being the quantity the end-to-end bound controls."""
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
