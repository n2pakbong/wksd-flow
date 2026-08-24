import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from wksdflow.targets import gaussian_target
from wksdflow.kernels import make_kernel_bundle
from wksdflow.metrics import stein_identity_residual

def test_stein_identity():
    tgt = gaussian_target(dim=2, eps=1.0)
    b = make_kernel_bundle("imq", s=0.0, grad_V=tgt.grad_V, eps=1.0)
    Z = tgt.sample(jax.random.PRNGKey(0), 200_000)
    for y in [jnp.zeros(2), jnp.array([1.0, -0.5]), jnp.array([3.0, 2.0])]:
        m, se = stein_identity_residual(b["k_pi"], Z, y)
        assert abs(m) < 4 * se + 1e-8, (y, m, se)

def test_diagonal_positive():
    tgt = gaussian_target(dim=2, eps=1.0)
    b = make_kernel_bundle("imq", s=0.0, grad_V=tgt.grad_V, eps=1.0)
    X = jax.random.normal(jax.random.PRNGKey(1), (50, 2))
    diag = jax.vmap(lambda x: b["k_pi"](x, x))(X)
    assert jnp.all(diag > 0)          # k_pi(x,x) = ||Phi(x)||^2 > 0
