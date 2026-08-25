import jax, jax.numpy as jnp
from collections import namedtuple
jax.config.update("jax_enable_x64", True)

Target = namedtuple("Target", "dim eps V grad_V rho sample log_pi")

def gaussian_target(dim=2, eps=1.0):
    V = lambda x: 0.5 * jnp.sum(x ** 2)
    logZ = 0.5 * dim * jnp.log(2.0 * jnp.pi * eps ** 2)
    return Target(dim, eps, V, jax.grad(V), 1.0,
                  lambda key, n: eps * jax.random.normal(key, (n, dim)),
                  lambda X: -jax.vmap(V)(X) / eps ** 2 - logZ)

def mixture_target(dim=2, eps=1.0, sep=1.5):
    """pi = 1/2 N(-m, eps^2 I) + 1/2 N(+m, eps^2 I), m = sep * e_1.
    Along the separation axis nabla^2 V = I - (sep^2/eps^2) sech^2(<x,m>/eps^2)
    times the projector onto m, so rho = 1 - sep^2/eps^2, which is negative
    as soon as sep > eps."""
    m = jnp.zeros(dim).at[0].set(sep)
    def V(x):
        q = 0.5 * jnp.sum((x - m) ** 2)
        r = 0.5 * jnp.sum((x + m) ** 2)
        return -eps ** 2 * jax.nn.logsumexp(
            jnp.array([-q / eps ** 2, -r / eps ** 2]))
    def sample(key, n):
        k1, k2 = jax.random.split(key)
        sgn = 2.0 * jax.random.bernoulli(k1, 0.5, (n, 1)) - 1.0
        return sgn * m + eps * jax.random.normal(k2, (n, dim))
    return Target(dim, eps, V, jax.grad(V), 1.0 - sep ** 2 / eps ** 2,
                  sample, lambda X: -jax.vmap(V)(X) / eps ** 2)

def logistic_posterior(key, dim=5, n_data=100, eps=1.0, scale=1.0):
    kx, ky = jax.random.split(key)
    X = jax.random.normal(kx, (n_data, dim))
    theta = jnp.ones(dim) / jnp.sqrt(dim)
    p = jax.nn.sigmoid(X @ theta)
    y = jax.random.bernoulli(ky, p).astype(X.dtype)
    def V(t):
        z = X @ t
        return eps ** 2 * (jnp.sum(jax.nn.softplus(z) - y * z)
                           + 0.5 * jnp.sum(t ** 2) / scale ** 2)
    return Target(dim, eps, V, jax.grad(V), eps ** 2 / scale ** 2, None,
                  lambda T: -jax.vmap(V)(T) / eps ** 2)
