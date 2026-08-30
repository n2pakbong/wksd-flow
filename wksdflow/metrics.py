"""Diagnostics. `energy_V` is the objective the flow actually minimizes,
`energy_U` is the unbiased version used for monitoring only."""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def energy_V(K_pi, Xp):
    return jnp.mean(K_pi(Xp, Xp))


def energy_U(K_pi, Xp):
    n = Xp.shape[0]
    G = K_pi(Xp, Xp)
    return (jnp.sum(G) - jnp.trace(G)) / (n * (n - 1))


def kl_gaussian(Xp, eps=1.0, scale=1.0, jitter=1e-10):
    """KL of the moment-matched Gaussian of the cloud to pi = N(0,(eps*scale)^2 I).

    This is NOT an estimate of KL(mu_t | pi) in general, only of the KL between
    the Gaussian projections. It is what we use to test the identity
    d/dt KL = -(2/eps^2) J on the Gaussian target, where the cloud stays close
    to Gaussian, and it should not be trusted on the mixture target.
    """
    n, d = Xp.shape
    m = jnp.mean(Xp, axis=0)
    Z = Xp - m
    S = Z.T @ Z / (n - 1) + jitter * jnp.eye(d)
    tau2 = (eps * scale) ** 2
    sign, logdet = jnp.linalg.slogdet(S / tau2)
    return 0.5 * (jnp.trace(S) / tau2 + jnp.sum(m ** 2) / tau2 - d - logdet)


def energy_distance(Xp, Yp):
    """Two-sample energy distance, a reference metric independent of k_pi."""
    d = lambda A, B: jnp.mean(jnp.linalg.norm(A[:, None] - B[None], axis=-1))
    return 2.0 * d(Xp, Yp) - d(Xp, Xp) - d(Yp, Yp)


def stein_identity_residual(k_pi, pi_samples, y):
    """Must vanish, by the Stein identity. This is the one test worth running
    before anything else: a nonzero value means k_pi and grad_V disagree.
    Returns (mean, standard error)."""
    vals = jax.vmap(lambda x: k_pi(x, y))(pi_samples)
    return float(jnp.mean(vals)), float(jnp.std(vals) / jnp.sqrt(len(vals)))
