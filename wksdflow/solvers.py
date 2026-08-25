import jax, jax.numpy as jnp
from .kernels import generator, pairwise
jax.config.update("jax_enable_x64", True)

def make_collocation_solver(q, grad_V, eps, alpha, gamma=1e-6):
    """Algorithm 2: Psi = sum_j a_j q(., X_j) with
    M_ij = ((-L_x + alpha) q)(X_i, X_j), solve (M + gamma I) a = h,
    velocity = -2 sum_j a_j grad_x q(X_i, X_j)."""
    Lq = generator(q, grad_V, eps, 0)
    M_entry = lambda x, y: -Lq(x, y) + alpha * q(x, y)
    M_fn = jax.jit(pairwise(M_entry))
    dq_fn = jax.jit(pairwise(jax.grad(q, argnums=0)))

    @jax.jit
    def velocity(Xp, h):
        M = M_fn(Xp, Xp)
        a = jnp.linalg.solve(M + gamma * jnp.eye(Xp.shape[0]), h)
        return -2.0 * jnp.einsum("j,ijd->id", a, dq_fn(Xp, Xp))
    return velocity

def make_semigroup_solver(grad_k_pi, grad_V, eps, alpha,
                          n_traj=8, n_steps=200, dt=None):
    """Estimator based on the resolvent representation. Uses
    grad_x E[h(X_t^x)] = E[J_t^T grad h(X_t^x)] with J_t the Jacobian of the
    Euler-Maruyama flow, and a left-point Riemann sum for the Laplace integral.
    Truncation at T = n_steps * dt biases the estimate by O(exp(-(alpha+rho)T)),
    so take alpha * T >> 1 or rho * T >> 1."""
    if dt is None:
        dt = 0.5 / max(alpha, 1.0)
    sq = jnp.sqrt(2.0) * eps * jnp.sqrt(dt)

    def grad_h(x, Xp):
        return jnp.mean(jax.vmap(grad_k_pi, (None, 0))(x, Xp), axis=0)

    def one_path(x0, Xp, noise):
        def step(carry, z):
            x, J, acc, t = carry
            acc = acc + jnp.exp(-alpha * t) * (J.T @ grad_h(x, Xp)) * dt
            HV = jax.jacfwd(grad_V)(x)
            return (x - grad_V(x) * dt + sq * z, J - (HV @ J) * dt,
                    acc, t + dt), None
        d = x0.shape[0]
        (_, _, acc, _), _ = jax.lax.scan(
            step, (x0, jnp.eye(d), jnp.zeros(d), 0.0), noise)
        return acc

    @jax.jit
    def velocity(Xp, key):
        N, d = Xp.shape
        noise = jax.random.normal(key, (N, n_traj, n_steps, d))
        per_traj = jax.vmap(jax.vmap(one_path, (None, None, 0)),
                            (0, None, 0))(Xp, Xp, noise)
        return -2.0 * jnp.mean(per_traj, axis=1)
    return velocity
