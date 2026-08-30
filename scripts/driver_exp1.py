import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from wksdflow.targets import gaussian_target
from wksdflow.kernels import make_kernel_bundle, imq
from wksdflow.solvers import make_collocation_solver
from wksdflow.flows import wksd_residual, integrate
from wksdflow.metrics import energy_V, energy_U

tgt = gaussian_target(dim=2, eps=1.0)
b = make_kernel_bundle("imq", s=0.0, grad_V=tgt.grad_V, eps=tgt.eps)
q = lambda x, y: imq(x, y, ell=1.0)
solve = make_collocation_solver(q, tgt.grad_V, tgt.eps, alpha=0.1, gamma=1e-8)

N, eta, n_steps = 200, 5e-3, 2000
X0 = 2.0 + jax.random.normal(jax.random.PRNGKey(0), (N, tgt.dim))
v = lambda Xp: solve(Xp, wksd_residual(b["K_pi"], Xp))
cb = lambda n, X: (n * eta, energy_U(b["K_pi"], X))
X, hist, (Xb, Jb, nb) = integrate(
    v, X0, eta, n_steps, monitor=lambda X: energy_V(b["K_pi"], X), callback=cb)
print("best iterate", nb, "objective", Jb)
