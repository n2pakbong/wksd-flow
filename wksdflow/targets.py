"""Targets: V, grad V, and (where available) exact samples and KL."""
import jax
import jax.numpy as jnp


class Target:
    def __init__(self, V, dim, eps=1.0, name="target", rho=None, exact_sample=None):
        self.V = V
        self.dim = dim
        self.eps = eps
        self.name = name
        self.rho = rho                       # semiconvexity constant, if known
        self.grad_V = jax.jit(jax.grad(V))
        self._exact_sample = exact_sample

    def sample(self, key, n):
        if self._exact_sample is None:
            raise NotImplementedError
        return self._exact_sample(key, n)


def gaussian_target(dim=1, eps=1.0, scale=1.0):
    """pi = N(0, eps^2 scale^2 I) since V = |x|^2/(2 scale^2)."""
    def V(x):
        return 0.5 * jnp.sum(x ** 2) / scale ** 2

    def exact(key, n):
        return eps * scale * jax.random.normal(key, (n, dim))

    return Target(V, dim, eps, f"gaussian{dim}d", rho=1.0 / scale ** 2, exact_sample=exact)


def mixture_target(dim=2, eps=1.0, sep=2.0, w=0.5):
    """Bimodal, non-log-concave: rho < 0, so alpha > -2 rho is needed."""
    mu1 = jnp.concatenate([jnp.array([sep]), jnp.zeros(dim - 1)])
    mu2 = -mu1

    def V(x):
        a = -0.5 * jnp.sum((x - mu1) ** 2) / eps ** 2 + jnp.log(w)
        b = -0.5 * jnp.sum((x - mu2) ** 2) / eps ** 2 + jnp.log(1 - w)
        return -eps ** 2 * jax.scipy.special.logsumexp(jnp.array([a, b]))

    def exact(key, n):
        k1, k2 = jax.random.split(key)
        z = jax.random.bernoulli(k1, w, (n, 1))
        x = eps * jax.random.normal(k2, (n, dim))
        return x + jnp.where(z, mu1, mu2)

    # crude lower bound on the Hessian for a well-separated mixture
    return Target(V, dim, eps, f"mixture{dim}d", rho=-(sep ** 2) / eps ** 2,
                  exact_sample=exact)


def logistic_posterior(key, dim=10, n_data=200, eps=1.0, prior_var=1.0):
    """Log-concave posterior: V = -log lik - log prior. rho >= 1/prior_var."""
    kx, kt, ky = jax.random.split(key, 3)
    X = jax.random.normal(kx, (n_data, dim))
    theta_true = jax.random.normal(kt, (dim,))
    p = jax.nn.sigmoid(X @ theta_true)
    y = jax.random.bernoulli(ky, p).astype(jnp.float32)

    def V(t):
        z = X @ t
        nll = jnp.sum(jnp.logaddexp(0.0, z) - y * z)
        return eps ** 2 * (nll + 0.5 * jnp.sum(t ** 2) / prior_var)

    return Target(V, dim, eps, f"logistic{dim}d", rho=1.0 / prior_var)
