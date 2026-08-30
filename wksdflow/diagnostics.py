"""Diagnostics. Nothing here changes the maths; it only prints numbers that
replace claims I cannot justify ("stable enough", "well conditioned").

Convention used below, must match wksdflow.kernels.generator:
    (L f)(x) = -<grad V(x), grad f(x)> + eps^2 * Laplacian f(x)
If kernels.generator uses a different convention, cond(M) here is still
indicative but not identical to the matrix the solver builds.
"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def estimate_lipschitz(v, Xp, seed=99, n_iter=12):
    """Spectral norm L of the Jacobian of the velocity field at the cloud Xp,
    by power iteration. Explicit Euler on x' = v(x) is stable for eta < 2/L;
    aim for eta * L <= 0.5. This is the computable local substitute for the
    theoretical bound eta <= lambda_alpha/(4 kappa_1^2) of Thm end_to_end,
    whose kappa_1 is a global sup and is not computable.
    """
    w = jax.random.normal(jax.random.PRNGKey(seed), Xp.shape)
    w = w / jnp.linalg.norm(w)
    nrm = 0.0
    for _ in range(n_iter):
        Jw = jax.jvp(lambda X: v(X), (Xp,), (w,))[1]
        nrm = jnp.linalg.norm(Jw)
        w = Jw / (nrm + 1e-30)
    return float(nrm)


def report_step_size(name, v, X0, eta):
    """Print, before integrating, whether eta is safe. Returns suggested eta."""
    try:
        L = estimate_lipschitz(v, X0)
    except Exception as e:            # stochastic velocities take a key
        print(f"[diag] {name}: skipped (velocity not autodiff-able: {e})")
        return eta
    print(f"[diag] {name}: L={L:.4g}  eta*L={eta*L:.4g} (want <=0.5)  "
          f"suggested --eta {0.5/max(L,1e-30):.3g}", flush=True)
    return 0.5 / max(L, 1e-30)


def collocation_diagnostics(q, grad_V, eps, alpha, gamma, Xp):
    """Conditioning of the collocation matrix M of Algorithm 2.
      cond(M)*1e-16 ~= relative error of the solve in float64; if that is
        above ~1e-6 the velocity is not trustworthy.
      gamma/s_min must be << 1, else you are solving a regularized problem and
        the KL identity of exp1 will not close.
    """
    def Lq(x, y):
        gx = jax.grad(q, argnums=0)(x, y)
        H = jax.hessian(q, argnums=0)(x, y)
        return -jnp.dot(grad_V(x), gx) + eps ** 2 * jnp.trace(H)

    M = jax.vmap(jax.vmap(lambda a, b: -Lq(a, b) + alpha * q(a, b),
                          in_axes=(None, 0)), in_axes=(0, None))(Xp, Xp)
    s = jnp.linalg.svd(M, compute_uv=False)
    c, smin = float(s[0] / s[-1]), float(s[-1])
    print(f"[diag] cond(M)={c:.3e}  s_min={smin:.3e}  gamma={gamma:.1e}  "
          f"gamma/s_min={gamma/smin:.3e} (want <<1)  "
          f"float64 loss ~{c*1e-16:.1e}", flush=True)
    return M


def check_admissible(alpha_min, rho, label="alpha"):
    """Assumption semiconvexity needs lambda_alpha = alpha + min(rho, 2rho) > 0,
    i.e. alpha > -2 rho when rho < 0. Raises instead of producing a figure that
    lies about a theorem's hypotheses."""
    thresh = -2.0 * min(float(rho), 0.0)
    if alpha_min <= thresh:
        raise ValueError(
            f"{label}={alpha_min} violates Assumption semiconvexity: "
            f"rho={rho:.4f} requires {label} > {thresh:.4f}")
    print(f"[diag] admissible: rho={rho:.4f}, need {label} > {thresh:.4f}, "
          f"have {alpha_min}", flush=True)
