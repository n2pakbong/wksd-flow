"""Two ways to apply R_alpha = (-L + alpha)^{-1} to h_mu at the particles.

Method A: symmetric collocation in a smooth kernel q (Algorithm alg:gpsd).
Method B: the semigroup estimator based on eqn:resolvent_laplace, which needs
          no linear algebra and scales far better in N.
"""
import jax
import jax.numpy as jnp
from .kernels import pairwise


def make_collocation_solver(q, grad_V, eps, alpha, gamma):
    """Returns psi_grad(Xp, h) -> (N, d) approximating grad Psi at the particles."""
    def Lq_x(x, y):
        g = jax.grad(q, argnums=0)(x, y)
        H = jax.hessian(q, argnums=0)(x, y)
        return -jnp.dot(grad_V(x), g) + eps ** 2 * jnp.trace(H)

    def op(x, y):                      # (-L_x + alpha) q(x, y)
        return -Lq_x(x, y) + alpha * q(x, y)

    OP = jax.jit(pairwise(op))
    dQ = jax.jit(pairwise(jax.grad(q, argnums=0)))

    @jax.jit
    def psi_grad(Xp, h):
        M = OP(Xp, Xp)
        n = M.shape[0]
        a = jnp.linalg.solve(M + gamma * jnp.eye(n), h)
        return jnp.einsum("ijd,j->id", dQ(Xp, Xp), a)

    return psi_grad


def make_semigroup_solver(k_pi_fn, grad_V, eps, alpha, n_traj=8, n_steps=200, dt=None):
    """Estimate grad R_alpha h_mu at the particles without any linear algebra.

    Uses grad_x E[h(X_t^x)] = E[J_t^T grad h(X_t^x)], the same identity as in the
    proof of the resolvent estimates, with J_t the Jacobian of the Euler-Maruyama
    flow, and a left-point Riemann sum for int_0^T exp(-alpha t) (...) dt.
    Truncation at T = n_steps * dt biases the estimate by O(exp(-(alpha+rho) T)),
    so take alpha * T >> 1 or rho * T >> 1.
    """
    if dt is None:
        dt = 0.5 / max(alpha, 1.0)
    sq = jnp.sqrt(2.0) * eps * jnp.sqrt(dt)

    def grad_h(x, Xp):
        g = jax.vmap(jax.grad(k_pi_fn, argnums=0), in_axes=(None, 0))(x, Xp)
        return jnp.mean(g, axis=0)

    def one_path(x0, Xp, noise):
        def step(carry, z):
            x, J, acc, t = carry
            acc = acc + jnp.exp(-alpha * t) * (J.T @ grad_h(x, Xp)) * dt
            HV = jax.jacfwd(grad_V)(x)
            x_new = x - grad_V(x) * dt + sq * z
            J_new = J - (HV @ J) * dt
            return (x_new, J_new, acc, t + dt), None
        d = x0.shape[0]
        init = (x0, jnp.eye(d), jnp.zeros(d), 0.0)
        (_, _, acc, _), _ = jax.lax.scan(step, init, noise)
        return acc

    @jax.jit
    def psi_grad(Xp, key):
        N, d = Xp.shape
        noise = jax.random.normal(key, (N, n_traj, n_steps, d))
        per_traj = jax.vmap(                        # over particles
            jax.vmap(one_path, in_axes=(None, None, 0)),  # over trajectories
            in_axes=(0, None, 0),
        )(Xp, Xp, noise)
        return jnp.mean(per_traj, axis=1)

    return psi_grad
