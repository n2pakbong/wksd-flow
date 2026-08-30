"""The three targets of Section (numerics). `rho` is the constant of
Assumption (semiconvexity): the admissibility condition is alpha + min(rho,2rho) > 0."""
import jax, jax.numpy as jnp
from collections import namedtuple
jax.config.update("jax_enable_x64", True)

Target = namedtuple("Target", "dim eps V grad_V rho sample log_pi")


def gaussian_target(dim=1, eps=1.0, scale=1.0):
    """V(x) = |x|^2 / (2 scale^2), so pi = N(0, (eps*scale)^2 I) and rho = 1/scale^2."""
    V = lambda x: 0.5 * jnp.sum(x ** 2) / scale ** 2
    tau2 = (eps * scale) ** 2
    logZ = 0.5 * dim * jnp.log(2.0 * jnp.pi * tau2)
    return Target(dim, eps, V, jax.grad(V), 1.0 / scale ** 2,
                  lambda key, n: jnp.sqrt(tau2) * jax.random.normal(key, (n, dim)),
                  lambda X: -jax.vmap(V)(X) / eps ** 2 - logZ)


def mixture_target(dim=2, eps=1.0, sep=2.0):
    """pi = 1/2 N(-m, eps^2 I) + 1/2 N(m, eps^2 I) with m = sep * e_1.
    Along e_1 the Hessian of V is 1 - (sep^2/eps^2) sech^2(<x,m>/eps^2), so
    rho = 1 - sep^2/eps^2 < 0 as soon as sep > eps, which is the regime that
    forces alpha > -2 rho in Theorem (kl_dissipation_regularized)."""
    m = jnp.zeros(dim).at[0].set(sep)

    def V(x):
        a = -0.5 * jnp.sum((x - m) ** 2) / eps ** 2
        b = -0.5 * jnp.sum((x + m) ** 2) / eps ** 2
        return -eps ** 2 * jax.nn.logsumexp(jnp.array([a, b]))

    def sample(key, n):
        k1, k2 = jax.random.split(key)
        sgn = 2.0 * jax.random.bernoulli(k1, 0.5, (n, 1)) - 1.0
        return sgn * m + eps * jax.random.normal(k2, (n, dim))

    return Target(dim, eps, V, jax.grad(V), 1.0 - sep ** 2 / eps ** 2,
                  sample, lambda X: -jax.vmap(V)(X) / eps ** 2)


def logistic_posterior(key, dim=10, n_data=200, eps=1.0, scale=1.0):
    """Bayesian logistic regression with a N(0, scale^2 I) prior. Log-concave,
    with rho >= eps^2 / scale^2 from the prior alone."""
    kx, ky = jax.random.split(key)
    X = jax.random.normal(kx, (n_data, dim))
    theta = jnp.ones(dim) / jnp.sqrt(dim)
    y = jax.random.bernoulli(ky, jax.nn.sigmoid(X @ theta)).astype(X.dtype)

    def V(t):
        z = X @ t
        return eps ** 2 * (jnp.sum(jax.nn.softplus(z) - y * z)
                           + 0.5 * jnp.sum(t ** 2) / scale ** 2)

    return Target(dim, eps, V, jax.grad(V), eps ** 2 / scale ** 2, None,
                  lambda T: -jax.vmap(V)(T) / eps ** 2)
