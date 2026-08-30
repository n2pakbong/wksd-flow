"""Base kernels, the weighting of Proposition (admissible class), and the
construction of k_pi = L_x L_y k together with its x-gradient.

k_pi is a FOURTH derivative of the base kernel and the V-statistic evaluates it
on the diagonal x = y. Any kernel written through r = sqrt(|x-y|^2) (Matern,
exponential, Wendland) yields NaN under autodiff there, because of the chain
rule through sqrt at 0. All base kernels here are smooth functions of
u = |x-y|^2, which is what makes the diagonal safe.
"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def gaussian(x, y, ell=1.0):
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2.0 * ell ** 2))


def imq(x, y, ell=1.0, beta=-0.5):
    """Inverse multiquadric, the kernel favoured for Stein discrepancies."""
    return (1.0 + jnp.sum((x - y) ** 2) / ell ** 2) ** beta


def rq(x, y, ell=1.0, a=2.0):
    """Rational quadratic, a scale mixture of Gaussians."""
    return (1.0 + jnp.sum((x - y) ** 2) / (2.0 * a * ell ** 2)) ** (-a)


BASE_KERNELS = {"gaussian": gaussian, "imq": imq, "rq": rq}

# --- radial kernels, shifted so that autodiff is safe on the diagonal --------
# k_delta(x,y) = f(|x-y|^2 + delta). By Schoenberg's theorem f is completely
# monotone (Matern is PD on every R^d), and complete monotonicity is preserved
# by shifting the argument, so k_delta is itself a positive definite kernel and
# a member of the admissible class of Prop. (admissible_class). It is C^infty
# because its argument is bounded below by delta, which is what makes the fourth
# derivative k_pi = L_x L_y k evaluable at x = y. The bias relative to the
# unshifted Matern is O(delta^{3/2}) in the fourth derivatives.

_MATERN_POLY = {
    2.5: (5.0 ** 0.5, lambda t: 1.0 + t + t ** 2 / 3.0),
    3.5: (7.0 ** 0.5, lambda t: 1.0 + t + 2.0 * t ** 2 / 5.0
                                    + t ** 3 / 15.0),
    4.5: (3.0,        lambda t: 1.0 + t + 3.0 * t ** 2 / 7.0
                                    + 2.0 * t ** 3 / 21.0 + t ** 4 / 105.0),
}


def matern(x, y, ell=1.0, nu=3.5, delta=1e-6):
    """Matern with half-integer nu in {2.5, 3.5, 4.5}, shifted by delta."""
    c, poly = _MATERN_POLY[float(nu)]
    t = c * jnp.sqrt(jnp.sum((x - y) ** 2) + delta) / ell
    return poly(t) * jnp.exp(-t)


BASE_KERNELS = {
    "gaussian": gaussian, "imq": imq, "rq": rq,
    "matern52": lambda x, y, **kw: matern(x, y, nu=2.5, **kw),
    "matern72": lambda x, y, **kw: matern(x, y, nu=3.5, **kw),
    "matern92": lambda x, y, **kw: matern(x, y, nu=4.5, **kw),
}


def weighted(kbar, s):
    """k(x,y) = w(x) kbar(x,y) w(y) with w(x) = (1+|x|^2)^{-s/2}, s > 0.
    This is the family of Proposition (admissible class), which is what makes
    k_pi and its first two derivatives bounded for confining potentials."""
    w = lambda x: (1.0 + jnp.sum(x ** 2)) ** (-0.5 * s)
    return lambda x, y: w(x) * kbar(x, y) * w(y)


def generator(f, grad_V, eps, argnum):
    """Apply L = -<grad V, grad .> + eps^2 Laplacian in argument `argnum`."""
    def out(x, y):
        g = jax.grad(f, argnums=argnum)(x, y)
        H = jax.hessian(f, argnums=argnum)(x, y)
        z = x if argnum == 0 else y
        return -jnp.dot(grad_V(z), g) + eps ** 2 * jnp.trace(H)
    return out


def pairwise(f):
    """f(x,y) -> F(X,Y) with F[i,j] = f(X[i],Y[j]), shape (N,M) or (N,M,d)."""
    return lambda X, Y: jax.vmap(jax.vmap(f, (None, 0)), (0, None))(X, Y)


def make_kernel_bundle(base="imq", s=0.0, base_kwargs=None, grad_V=None,
                       eps=1.0):
    """Returns {'k', 'k_pi', 'grad_k_pi', 'K_pi', 'dK_pi'}.

    k_pi(x,y)      scalar Stein kernel
    grad_k_pi(x,y) its gradient in the first argument
    K_pi(X,Y)      (N,M) matrix of k_pi
    dK_pi(X,Y)     (N,M,d) array of grad_k_pi
    """
    if grad_V is None:
        raise ValueError("grad_V is required to build k_pi.")
    kbar = lambda x, y: BASE_KERNELS[base](x, y, **(base_kwargs or {}))
    k = weighted(kbar, s) if s > 0 else kbar
    k_pi = jax.jit(generator(generator(k, grad_V, eps, 1), grad_V, eps, 0))
    grad_k_pi = jax.jit(jax.grad(k_pi, argnums=0))
    return {"k": k, "k_pi": k_pi, "grad_k_pi": grad_k_pi,
            "K_pi": jax.jit(pairwise(k_pi)),
            "dK_pi": jax.jit(pairwise(grad_k_pi))}
