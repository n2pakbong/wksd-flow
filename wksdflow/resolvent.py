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
    """Estimate grad R_alpha h_mu by pathwise differentiation of the semigroup.

    We use grad_x E[h(X_t^x)] = E[J_t^T grad h(X_t^x)] with J_t the Jacobian of
    the Euler-Maruyama flow, which is exactly the identity used in the proof of
    Lemma lem_resolvent_estimates. Trajectories are truncated at time
    T = n_steps * dt and weighted by exp(-alpha t) dt.
    """
    if dt is None:
        dt = 0.5 / max(alpha, 1.0)

    def grad_h(x, Xp):
        # grad_x (1/N) sum_j k_pi(x, X_j)
        g = jax.vmap(jax.grad(k_pi_fn, argnums=0), in_axes=(None, 0))(x, Xp)
        return jnp.mean(g, axis=0)

    @jax.jit
    def psi_grad(Xp, key):
        N, d = Xp.shape

        def one_path(x0, subkey):
            noise = jax.random.normal(subkey, (n_steps, d))

            def step(carry, inp):
                x, J, acc, t = carry
                z = inp
                acc = acc + jnp.exp(-alpha * t) * (J.T @ grad_h(x, Xp)) * dt
                Hv = jax.hessian(lambda z_: jnp.sum(grad_V(z_)))  # unused placeholder
                x_new = x - grad_V(x) * dt + jnp.sqrt(2.0) * eps * jnp.sqrt(dt) * z
                HV = jax.jacfwd(grad_V)(x)
                J_new = J - HV @ J * dt
                return (x_new, J_new, acc, t + dt), None

            init = (x0, jnp.eye(d), jnp.zeros(d), 0.0)
            (_, _, acc, _), _ = jax.lax.scan(step, init, noise)
            return acc

        keys = jax.random.split(key, N * n_traj).reshape(N, n_traj, 2)
        per_particle = jax.vmap(
            lambda x, ks: jnp.mean(jax.vmap(one_path, in_axes=(None, 0))(x, ks), axis=0)
        )(Xp, keys)
        return per_particle

    return psi_grad
