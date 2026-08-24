import jax
import jax.numpy as jnp


def energy_V(K_pi, Xp):
    """V-statistic: the exact gradient of the empirical energy (biased)."""
    return jnp.mean(K_pi(Xp, Xp))


def energy_U(K_pi, Xp):
    """U-statistic: unbiased, used for monitoring only."""
    G = K_pi(Xp, Xp)
    n = G.shape[0]
    return (jnp.sum(G) - jnp.trace(G)) / (n * (n - 1))


def kl_gaussian(Xp, eps, scale):
    """KL(N(mhat, Shat) || N(0, eps^2 scale^2 I)), a plug-in proxy for
    KL(mu_t || pi) used to test the identity eqn:kl_identity numerically."""
    n, d = Xp.shape
    m = jnp.mean(Xp, axis=0)
    S = jnp.cov(Xp.T, bias=True).reshape(d, d) + 1e-8 * jnp.eye(d)
    s2 = (eps * scale) ** 2
    return 0.5 * (jnp.trace(S) / s2 + jnp.sum(m ** 2) / s2 - d
                  - jnp.linalg.slogdet(S / s2)[1])


def energy_distance(X, Y):
    def d(A, B):
        return jnp.mean(jnp.linalg.norm(A[:, None] - B[None, :], axis=-1))
    return 2 * d(X, Y) - d(X, X) - d(Y, Y)

def chi2_ratio_estimate(Xp, log_pi, log_mu_hat):
    """Crude plug-in estimate of chi^2(mu | pi) = E_mu[dmu/dpi] - 1, needed to
    check the horizon T_sigma of the local-rate theorem. `log_mu_hat` should be a
    kernel density estimate evaluated at the particles."""
    logr = log_mu_hat(Xp) - log_pi(Xp)
    return float(jnp.mean(jnp.exp(logr)) - 1.0)


def stein_identity_residual(k_pi, pi_samples, y):
    """Must be ~0 by the Stein identity. THE single best correctness test:
    if this is not zero, the kernel and grad_V are inconsistent."""
    vals = jax.vmap(lambda x: k_pi(x, y))(pi_samples)
    n = pi_samples.shape[0]
    return float(jnp.mean(vals)), float(jnp.std(vals) / jnp.sqrt(n))
