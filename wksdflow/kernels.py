"""Base kernels, weighting, and the diffusion Stein kernel k_pi = L_x L_y k.

The only thing a user must supply is a scalar base kernel kbar(x, y).
Everything else, including k_pi and all its derivatives, is obtained by
automatic differentiation, so no fourth-order derivative is ever hand-coded.
"""
from functools import partial
import jax
import jax.numpy as jnp

# ---------- base kernels: (d,), (d,) -> scalar ----------

def gaussian(x, y, ell=1.0):
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2.0 * ell ** 2))

def imq(x, y, ell=1.0, beta=-0.5):
    return (1.0 + jnp.sum((x - y) ** 2) / ell ** 2) ** beta

def matern72(x, y, ell=1.0):
    # nu = 7/2, C^6, so admissible for kappa_0, kappa_1 (see Prop. admissible_class)
    r = jnp.sqrt(jnp.sum((x - y) ** 2) + 1e-12)
    s = jnp.sqrt(7.0) * r / ell
    poly = 1.0 + s + 2.0 * s ** 2 / 5.0 + s ** 3 / 15.0
    return poly * jnp.exp(-s)

BASE_KERNELS = {"gaussian": gaussian, "imq": imq, "matern72": matern72}


def make_weighted(kbar, s=0.0):
    """k(x,y) = w(x) kbar(x,y) w(y) with w(x) = (1+|x|^2)^{-s/2}.

    s = 0 disables the weight. Take s > m with |grad V| = O(|x|^m) to make
    k_pi and its derivatives bounded, as in Proposition admissible_class.
    """
    if s == 0.0:
        return kbar

    def w(x):
        return (1.0 + jnp.sum(x ** 2)) ** (-0.5 * s)

    def k(x, y):
        return w(x) * kbar(x, y) * w(y)

    return k


def make_stein_kernel(k, grad_V, eps):
    """Return k_pi(x, y) = (L_x L_y k)(x, y) for L f = -<grad V, grad f> + eps^2 Lap f.

    Implementation: apply L in the second argument by autodiff, then apply L in
    the first argument to the resulting function. This is exact and works for
    any k that JAX can differentiate twice in each slot.
    """
    def Ly(x, y):
        g = jax.grad(k, argnums=1)(x, y)
        H = jax.hessian(k, argnums=1)(x, y)
        return -jnp.dot(grad_V(y), g) + eps ** 2 * jnp.trace(H)

    def k_pi(x, y):
        g = jax.grad(Ly, argnums=0)(x, y)
        H = jax.hessian(Ly, argnums=0)(x, y)
        return -jnp.dot(grad_V(x), g) + eps ** 2 * jnp.trace(H)

    return k_pi


def pairwise(f):
    """Vectorize a scalar (x, y) function to an (n, m) matrix."""
    return jax.vmap(jax.vmap(f, in_axes=(None, 0)), in_axes=(0, None))


def make_kernel_bundle(base="matern72", s=0.0, base_kwargs=None,
                       grad_V=None, eps=1.0):
    """Everything the flows need: k_pi, grad_x k_pi, and their vectorizations."""
    base_kwargs = base_kwargs or {}
    kbar = partial(BASE_KERNELS[base], **base_kwargs)
    k = make_weighted(kbar, s=s)
    k_pi = make_stein_kernel(k, grad_V, eps)
    dk_pi = jax.grad(k_pi, argnums=0)          # nabla_x k_pi(x, y)
    return dict(k=jax.jit(k),
                k_pi=jax.jit(k_pi),
                dk_pi=jax.jit(dk_pi),
                K_pi=jax.jit(pairwise(k_pi)),           # (n, m)
                dK_pi=jax.jit(pairwise(dk_pi)))         # (n, m, d)
