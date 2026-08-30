"""Is the sqrt-shift delta of the Matern kernel harmless?

Run:  python scripts/check_delta.py --base matern72 --dim 2
Read: the three J values should agree to ~5 digits. If they do, delta is
irrelevant and you quote the middle one. If they do not, delta is too large
(or float64 is off). Nothing else in the codebase depends on this script.
"""
import jax
jax.config.update("jax_enable_x64", True)
from wksdflow.config import Config
from wksdflow.targets import gaussian_target
from wksdflow.kernels import make_kernel_bundle
from wksdflow.metrics import energy_U

cfg = Config.from_cli(base="matern72", dim=2, n=200, eps=1.0)
tgt = gaussian_target(cfg.dim, cfg.eps)
X0 = jax.random.normal(jax.random.PRNGKey(cfg.seed), (cfg.n, cfg.dim))

for d in (1e-4, 1e-6, 1e-8):
    b = make_kernel_bundle(cfg.base, s=cfg.s,
                           base_kwargs={"ell": cfg.ell, "delta": d},
                           grad_V=tgt.grad_V, eps=cfg.eps)
    print(f"delta={d:.0e}   J = {float(energy_U(b['K_pi'], X0)):.10e}")
