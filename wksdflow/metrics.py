import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def energy_V(K_pi, Xp):
    """V-statistic, the exact gradient of the empirical energy."""
    return jnp.mean(K_pi(Xp, Xp))

def energy_U(K_pi, Xp):
    """U-statistic, unbiased, used only for monitoring."""
    n = Xp.shape[0]
    G = K_pi(Xp, Xp)
    return (jnp.sum(G) - jnp.trace(G)) / (n * (n - 1))

def chi2_estimate(Xp, log_pi, log_mu_hat):
    """Plug-in estimate of chi^2(mu | pi) = E_mu[dmu/dpi] - 1, used to check the
    horizon T_sigma. `log_mu_hat` should be a kernel density estimate."""
    return float(jnp.mean(jnp.exp(log_mu_hat(Xp) - log_pi(Xp))) - 1.0)

def stein_identity_residual(k_pi, pi_samples, y):
    """Must be zero by the Stein identity. This is the single best correctness
    test: if it is not zero, the kernel and grad_V are inconsistent."""
    vals = jax.vmap(lambda x: k_pi(x, y))(pi_samples)
    n = pi_samples.shape[0]
    return float(jnp.mean(vals)), float(jnp.std(vals) / jnp.sqrt(n))
