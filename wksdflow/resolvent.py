"""Two implementations of the resolvent preconditioner, matching Definition
(resolvent preconditioner). Both return grad Psi at the particle locations, and
the factor -2 of v = -2 grad Psi is applied in flows.generator_velocity."""
import jax, jax.numpy as jnp
from .kernels import generator, pairwise
jax.config.update("jax_enable_x64", True)


def make_collocation_solver(q, grad_V, eps, alpha, gamma=1e-8):
    """Algorithm 2. Psi = sum_j a_j q(., X_j) with
    M[i,j] = ((-L_x + alpha) q)(X_i, X_j) and (M + gamma I) a = h.
    Cost per call is O(N^2) kernel evaluations plus one O(N^3) dense solve.
    `gamma` is the Tikhonov parameter of the algorithm and also what keeps M
    invertible, since M is not symmetric and can be ill conditioned."""
    Lq = generator(q, grad_V, eps, 0)
    M_fn = jax.jit(pairwise(lambda x, y: -Lq(x, y) + alpha * q(x, y)))
    dq_fn = jax.jit(pairwise(jax.grad(q, argnums=0)))

    @jax.jit
    def psi_grad(Xp, h):
        M = M_fn(Xp, Xp)
        a = jnp.linalg.solve(M + gamma * jnp.eye(Xp.shape[0]), h)
        return jnp.einsum("j,ijd->id", a, dq_fn(Xp, Xp))
    return psi_grad


def make_semigroup_solver(bundle, grad_V, eps, alpha, n_traj=8, n_steps=200,
                          dt=None):
    """The estimator based on the Laplace representation of the resolvent. Uses
    grad_x E[h(X_t^x)] = E[J_t^T grad h(X_t^x)], with J_t the Jacobian of the
    Euler-Maruyama flow, and a left-point Riemann sum for the t-integral. No
    linear algebra, cost O(N^2 * n_traj * n_steps).

    Returns psi_grad_sg(X, key), so it must be used with
    flows.generator_velocity_stochastic and integrate(..., stochastic=True).

    Truncating at T = n_steps*dt biases the estimate by O(exp(-(alpha+rho)T)),
    so check that (alpha + max(rho,0)) * n_steps * dt is at least about 5.
    """
    grad_k_pi = bundle["grad_k_pi"]
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
    def psi_grad_sg(Xp, key):
        N, d = Xp.shape
        noise = jax.random.normal(key, (N, n_traj, n_steps, d))
        paths = jax.vmap(jax.vmap(one_path, (None, None, 0)),
                         (0, None, 0))(Xp, Xp, noise)
        return jnp.mean(paths, axis=1)
    return psi_grad_sg
