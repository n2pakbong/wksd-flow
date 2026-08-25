import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# k_pi is a FOURTH derivative of the base kernel, so any kernel built from
# r = sqrt(|x-y|^2) is singular under autodiff on the diagonal x = y, which is
# exactly where the V-statistic evaluates it. All base kernels below are smooth
# functions of u = |x-y|^2. Matern kernels are therefore excluded from the
# numerics even though Proposition (admissible class) covers them.

def gaussian(x, y, ell=1.0):
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2.0 * ell ** 2))

def imq(x, y, ell=1.0, beta=-0.5):
    return (1.0 + jnp.sum((x - y) ** 2) / ell ** 2) ** beta

def rational_quadratic(x, y, ell=1.0, a=2.0):
    return (1.0 + jnp.sum((x - y) ** 2) / (2.0 * a * ell ** 2)) ** (-a)

BASE_KERNELS = {"gaussian": gaussian, "imq": imq, "rq": rational_quadratic}

def weighted(kbar, s):
    """k(x,y) = w(x) kbar(x,y) w(y) with w(x) = (1+|x|^2)^{-s/2}."""
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
    return lambda X, Y: jax.vmap(jax.vmap(f, (None, 0)), (0, None))(X, Y)

def make_kernel_bundle(base="imq", s=0.0, grad_V=None, eps=1.0, **kw):
    """Returns k_pi, its gradient in x, and batched versions."""
    kbar = lambda x, y: BASE_KERNELS[base](x, y, **kw)
    k = weighted(kbar, s) if s > 0 else kbar
    k_pi = jax.jit(generator(generator(k, grad_V, eps, 1), grad_V, eps, 0))
    grad_k_pi = jax.jit(jax.grad(k_pi, argnums=0))
    return {"k": k, "k_pi": k_pi, "grad_k_pi": grad_k_pi,
            "K_pi": jax.jit(pairwise(k_pi)), "dK_pi": jax.jit(pairwise(grad_k_pi))}
