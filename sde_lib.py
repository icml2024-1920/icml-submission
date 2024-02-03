"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import jax.numpy as jnp
import jax
import numpy as np
from utils import batch_mul, from_flattened_numpy, to_flattened_numpy

from scipy import integrate


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, rng, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a JAX tensor.
      t: a JAX float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * jnp.sqrt(dt)
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - batch_mul(diffusion ** 2, score * (0.5 if self.probability_flow else 1.))
        # Set the diffusion function to zero for ODEs.
        diffusion = jnp.zeros_like(diffusion) if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - batch_mul(G ** 2, score_fn(x, t) * (0.5 if self.probability_flow else 1.))
        rev_G = jnp.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class RFSDE(SDE):
  """
    Method description
      X0 ~ p_0(*) (distribution p_0)
      X1 ~ p_1(*) (distribution p_1)

      q(Xt|X0,X1) ~ t * X1 + (1 - t) * X0
  """

  def __init__(self, N=1000):
    super().__init__(N)
    self.N = N
  
  @property
  def T(self):
    return 1

  def sde(self, x, t):
    # TODO
    # Currently used in `losses.py` for g2 (likelihood-based loss)
    # And sampling (E-M sampler)
    pass
    # beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    # drift = -0.5 * batch_mul(beta_t, x)
    # diffusion = jnp.sqrt(beta_t)
    # return drift, diffusion

  def marginal_prob(self, x, t):
    """
      Input:
        x = (x0, x1)
        x0: clean image
        x1: degraded image
        t: time in [0, self.T]
      Return:
        mean: mean of marginal distribution of bridge distribution q(Xt|X0, X1)
        std: standard deviation of marginal distribution of bridge distribution.
        (xt = mean + std * I)
    """
    assert len(x) == 2
    x0, x1 = x

    mean = batch_mul(x0, 1 - self.T) + batch_mul(x1, self.T)
    std = jnp.zeros_like(t) 
    return mean, std


  def prior_sampling(self, rng, shape):
    # prior_sampling for I2SB starts with the degraded image.
    return jax.random.normal(rng, shape)

  def prior_logp(self, z):
    # TODO
    shape = z.shape
    N = np.prod(shape[1:])
    logp_fn = lambda z: -N / 2. * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.
    return jax.vmap(logp_fn)(z)

  def discretize(self, x, t):
    """DDPM discretization."""
    pass

  def reverse(self, score_fn, probability_flow=True):
    T = self.T
    sde_fn = self.sde

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        pass

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        return score_fn(x, t), jnp.zeros_like(t)

    return RSDE()



class I2SBSDE(SDE):
  """
    Method description
      X0 ~ p_data(*) (clean image)
      X1 ~ p(*|X0) (noisy image)

      q(Xt|X0,X1) ~ N(*; mu_t(X0,X1), Sigma_t)
        mu_t(X0,X1) = [sigmabar_t^2 / (sigmabar_t^2 + sigma_t^2)] * X0 
                    + [sigma_t^2 / (sigmabar_t^2 + sigma_t^2)]    * X1
        Sigma_t = [(sigma_t^2 * sigmabar_t^2) / (sigmabar_t^2 + sigma_t^2)] I

  """

  def __init__(self, beta_min=0.1, beta_max=0.3, N=1000):
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
  
  @property
  def T(self):
    return 1

  def sde(self, x, t):
    # TODO
    # Currently used in `losses.py` for g2 (likelihood-based loss)
    # And sampling (E-M sampler)
    pass
    # beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    # drift = -0.5 * batch_mul(beta_t, x)
    # diffusion = jnp.sqrt(beta_t)
    # return drift, diffusion

  def marginal_prob(self, x, t):
    """
      Input:
        x = (x0, x1)
        x0: clean image
        x1: degraded image
        t: time in [0, self.T]
      Return:
        mean: mean of marginal distribution of bridge distribution q(Xt|X0, X1)
        std: standard deviation of marginal distribution of bridge distribution.
        (xt = mean + std * z
            = (x0_coeff * x0 + x1_coeff * x1) + std * z, z ~ N(0, I))

      Description
        Proposition 3.3 (I2SB)
          q(xt | x0, x1) = N(xt; mu_t(x0, x1), std_t ** 2)
          mu_t(x0, x1) = sigmabar_t ** 2 / (sigmabar_t ** 2 + sigma_t ** 2) * x0
                       + sigma_t ** 2 / (sigmabar_t ** 2 + sigma_t ** 2) * x1
          std_t        = sigma_t * sigmabar_t / sqrt(sigmabar_t ** 2 + sigma_t ** 2)

          sigma_t = int_0^t beta_tau d tau
          sigmabar_t = int_t^1 beta_tau d tau

          beta_tau = beta_min -> beta_max linearly in [0, 0.5], beta_max -> beta_min linearly in [0.5, 1]
    """
    assert isinstance(x, tuple)
    assert len(x) == 2
    x0, x1 = x

    # beta_0 (at 0) -> (beta_0 + beta_1) / 2 (at 0.5) -> beta_0 (at 1)
    beta_before_half = jnp.minimum(self.beta_0 * t + ((self.beta_1 - self.beta_0) * t ** 2) / 2, (self.beta_0 * 3 + self.beta_1) / 8)
    t_flip = self.T - t
    beta_after_half = jnp.minimum(self.beta_0 * t_flip + ((self.beta_1 - self.beta_0) * t_flip ** 2) / 2, (self.beta_0 * 3 + self.beta_1) / 8)
    beta_after_half = (self.beta_0 * 3 + self.beta_1) / 8 - beta_after_half
    beta_t = beta_before_half + beta_after_half

    sigma_t = jnp.sqrt(beta_t)
    sigmabar_t = jnp.sqrt((self.beta_0 * 3 + self.beta_1) / 4 - beta_t)

    x0_coeff = sigmabar_t ** 2 / (sigmabar_t ** 2 + sigma_t ** 2)
    x1_coeff = sigma_t ** 2 / (sigmabar_t ** 2 + sigma_t ** 2)

    # sample: x0_coeff * x0 + x1_coeff * x1 + std * n
    mean = batch_mul(x0_coeff, x0) + batch_mul(x1_coeff, x1)
    std = jnp.sqrt((sigma_t ** 2 * sigmabar_t ** 2) / (sigmabar_t ** 2 + sigma_t ** 2))
    return mean, std

  def get_sigma(self, t):
    """
      Args:
        t: time
      Return:
        sigma_t: sigma_t
    """
    beta_before_half = jnp.minimum(self.beta_0 * t + ((self.beta_1 - self.beta_0) * t ** 2) / 2, (self.beta_0 * 3 + self.beta_1) / 8)
    t_flip = self.T - t
    beta_after_half = jnp.minimum(self.beta_0 * t_flip + ((self.beta_1 - self.beta_0) * t_flip ** 2) / 2, (self.beta_0 * 3 + self.beta_1) / 8)
    beta_after_half = (self.beta_0 * 3 + self.beta_1) / 8 - beta_after_half
    beta_t = beta_before_half + beta_after_half
    sigma_t = jnp.sqrt(beta_t)
    return sigma_t


  def prior_sampling(self, x1):
    # prior_sampling for I2SB starts with the degraded image. (beginning of the sampling process)
    return x1

  def prior_logp(self, z):
    # TODO
    shape = z.shape
    N = np.prod(shape[1:])
    logp_fn = lambda z: -N / 2. * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.
    return jax.vmap(logp_fn)(z)

  def discretize(self, x, t):
    """DDPM discretization."""
    # TODO
    pass

  def get_coeff(self, t):
    """Get signal, score, noise coefficients for sampling."""
    assert isinstance(t, tuple), f"{t} should be a tuple."
    current_t, next_t = t
    sigma_current_t, sigma_next_t = self.get_sigma(current_t), self.get_sigma(next_t)

    # Compute coefficients for prediction of x0.
    # pred_x0 = xt - sigma_current_t * score (Reversing Eq. 12: ||score  - (xt - x0) / sigma_current_t|| --> 0)

    # xt_prev = mu_x0 * pred_x0                        + mu_xt * xt + stdev * z (noise term, optional)
    #         = mu_x0 * (xt - sigma_current_t * score) + mu_xt * xt + stdev * z (noise term, optional)
    #         = (mu_x0 + mu_xt) * xt - (mu_x0 * sigma_current_t) * score + stdev * z
    sigma_diff_t = jnp.sqrt(sigma_current_t ** 2 - sigma_next_t ** 2)
    denom = sigma_next_t ** 2 + sigma_diff_t ** 2
    mu_x0 = sigma_diff_t ** 2 / denom
    mu_xt = sigma_next_t ** 2 / denom
    stdev = jnp.sqrt((sigma_next_t ** 2 * sigma_diff_t ** 2) / denom)

    # c_signal = mu_x0 + mu_xt
    c_signal = jnp.ones_like(current_t)
    c_score = - mu_x0 * sigma_current_t
    c_noise = stdev
    return c_signal, c_score, c_noise


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = jnp.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * batch_mul(beta_t, x)
    diffusion = jnp.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = batch_mul(jnp.exp(log_mean_coeff), x)
    std = jnp.sqrt(1 - jnp.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, rng, shape):
    return jax.random.normal(rng, shape)

  def snr(self, t):
    # Forward diffusion process: q(x_t|x_0) ~ N(alpha_t x_0; sigma_t^2 I)
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    sigma_t = jnp.sqrt(1 - jnp.exp(2. * log_mean_coeff))
    alpha_t = jnp.exp(log_mean_coeff)
    signal_to_noise_ratio = alpha_t ** 2 / sigma_t ** 2

    return alpha_t, sigma_t, signal_to_noise_ratio
    
  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logp_fn = lambda z: -N / 2. * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.
    return jax.vmap(logp_fn)(z)

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
    beta = self.discrete_betas[timestep]
    alpha = self.alphas[timestep]
    sqrt_beta = jnp.sqrt(beta)
    f = batch_mul(jnp.sqrt(alpha), x) - x
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * batch_mul(beta_t, x)
    discount = 1. - jnp.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = jnp.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = batch_mul(jnp.exp(log_mean_coeff), x)
    std = 1 - jnp.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, rng, shape):
    return jax.random.normal(rng, shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logp_fn = lambda z: -N / 2. * jnp.log(2 * np.pi) - jnp.sum(z ** 2) / 2.
    return jax.vmap(logp_fn)(z)


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = jnp.exp(np.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = jnp.zeros_like(x)
    diffusion = sigma * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, rng, shape):
    return jax.random.normal(rng, shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logp_fn = lambda z: -N / 2. * jnp.log(2 * np.pi * self.sigma_max ** 2) - jnp.sum(z ** 2) / (2 * self.sigma_max ** 2)
    return jax.vmap(logp_fn)(z)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).astype(jnp.int32)
    sigma = self.discrete_sigmas[timestep]
    adjacent_sigma = jnp.where(timestep == 0, jnp.zeros_like(timestep), self.discrete_sigmas[timestep - 1])
    f = jnp.zeros_like(x)
    G = jnp.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G


class EDMSDE(SDE):
  def __init__(self, sigma_min=0.002, sigma_max=80, sigma_data=0.5, p_std=1.2, p_mean=-1.2, N=1000):
    """Construct EDM-SDE.

    Args:
      sigma_min: Minimum supported noise level.
      sigma_max: Maximum supported noise level.
      sigma_data: Expected deviation of the training data.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.sigma_data = sigma_data
    self.p_std = p_std
    self.p_mean = p_mean
    self.N = N

    # fixed hyperparameters for sampling
    self.rho = 7.0
    self.S_churn = 1.0
    self.S_min = 0.0
    self.S_max = float('inf')

  @property
  def T(self):
    return self.sigma_max

  def sde(self, x, t):
    # TODO: Do not use here.
    raise NotImplementedError()
    # return drift, diffusion

  def marginal_prob(self, x, t):
    mean = x
    std = t
    return mean, std

  def prior_sampling(self, rng, shape):
    return jax.random.normal(rng, shape) * self.sigma_max
  
  def prior_logp(self, z):
    raise NotImplementedError()

  def heun_params(self, t):
    """Parameters for Heun solver.
      Return: {dsigma, sigma, s} at time t.
        sigma(t) = t
        dsigma(t) = 1
        s(t) = 1
    """
    dsigma = jnp.ones_like(t, dtype=jnp.float32)
    sigma = t
    s = jnp.ones_like(t, dtype=jnp.float32)
    ds = jnp.zeros_like(t, dtype=jnp.float32)
    return dsigma, sigma, s, ds

  # TODO: required?
  def discretize(self, x, t):
    raise NotImplementedError()
