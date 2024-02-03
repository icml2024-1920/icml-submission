# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import jax
import jax.numpy as jnp
import jax.random as random
import abc
import flax
import numpy as np

from models.utils import get_score_fn
from scipy import integrate
import sde_lib
from utils import batch_mul, from_flattened_numpy, to_flattened_numpy, get_timestep, get_sampling_interval, rescale_time

from models import utils as mutils


_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, **kwargs):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """
  gen_reflow = kwargs['gen_reflow'] if 'gen_reflow' in kwargs else False

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  rtol=config.sampling.tol,
                                  atol=config.sampling.tol,
                                  eps=eps,
                                  gen_reflow=gen_reflow,
                                  reflow_t=config.training.reflow_t if 'reflow_t' in config.training else 1,
                                  adaptive_interval=config.training.adaptive_interval)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    if config.sampling.predictor == 'rf_solver':
      nfe_multiplier = 1
    elif config.sampling.predictor == 'rf_solver_heun':
      nfe_multiplier = 2
    else:
      raise ValueError()
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 inverse_scaler=inverse_scaler,
                                 probability_flow=config.sampling.probability_flow,
                                 eps=eps,
                                 gen_reflow=gen_reflow,
                                 reflow_t=config.training.reflow_t if 'reflow_t' in config.training else 1,
                                 nfe_multiplier=nfe_multiplier,
                                 save_trajectory=config.eval.save_trajectory,
                                 adaptive_interval=config.training.adaptive_interval)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    if isinstance(sde, sde_lib.VPSDE):
      self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn
    self.probability_flow = probability_flow

  @abc.abstractmethod
  def update_fn(self, rng, x, t):
    """One update of the predictor.

    Args:
      rng: A JAX random state.
      x: A JAX array representing the current state
      t: A JAX array representing the current time step.

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='rf_solver')
class RFPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=True, **kwargs):
    super().__init__(sde, score_fn, probability_flow)
    assert 'total_interval' in kwargs
    self.total_interval = kwargs['total_interval']
  
  def update_fn(self, rng, x, t, index):
    sde = self.sde
    current_t, next_t = t

    score = self.score_fn(x, current_t, index)
    x = x + batch_mul(score, self.total_interval) * 0.999 # TODO: adaptive case + some calculation error by bug in RF code.
    # x = x + batch_mul(score, next_t - current_t)
    return x, x


# TODO: Second-order RF solver
@register_predictor(name='rf_solver_heun')
class RFHeunPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=True, **kwargs):
    super().__init__(sde, score_fn, probability_flow)
    assert 'total_interval' in kwargs
    self.total_interval = kwargs['total_interval']

  def update_fn(self, rng, x, t, index):
    sde = self.sde
    current_t, next_t = t

    # algorithm
    score = self.score_fn(x, current_t, index)
    x_mid = x + batch_mul(score, self.total_interval / 2) * 0.999
    score_mid = self.score_fn(x_mid, next_t, index)
    x = x + (batch_mul(score_mid, self.total_interval / 2) + batch_mul(score, self.total_interval / 2)) * 0.999
    return x, x
#############################################################################################################
def shared_predictor_update_fn(rng, state, x, t, index, sde, predictor, probability_flow, eps=None, **kwargs):
  """A wrapper that configures and returns the update function of predictors."""
  assert 'total_interval' in kwargs
  score_fn = mutils.get_score_fn(sde, state, state.opt_state_ema.ema, train=False, eps=eps)
  predictor_obj = predictor(sde, score_fn, probability_flow, total_interval=kwargs['total_interval'])
  return predictor_obj.update_fn(rng, x, t, index)


def get_pc_sampler(sde, shape, predictor, inverse_scaler, probability_flow=False,
                   denoise=True, eps=1e-3, gen_reflow=False, reflow_t=1, nfe_multiplier=1, save_trajectory=False, adaptive_interval=False):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    inverse_scaler: The inverse data normalizer.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    gen_reflow: valid only if rf_sampler.
                Flag to generate reflow images
    reflow_t: Number of reflow time division. Default=1 (No division.)
    nfe_multiplier: 1 if Euler, 2 if Heun.
    save_trajectory: For analysis.
    adaptive_interval: Adaptive interval code.

  Returns:
    A sampling function that takes random states, and a replcated training state and returns samples as well as
    the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          eps=eps)


  intvl = 1 - np.array(np.load('assets/c10_adaptive_index_500.npy', allow_pickle=True)[()][reflow_t]) / 500 if adaptive_interval else None

  # Define RF sampler
  def rf_sampler(rng, state, cond_image=None, **kwargs):

    rng, step_rng = random.split(rng)
    timestep_dict = get_timestep(reflow_t, sde.N, adaptive_interval=intvl)
    vec_t, interval_t = get_sampling_interval(batch_size=shape[0], n_steps=sde.N, timestep_dict=timestep_dict, gen_reflow=gen_reflow, rng=step_rng)
    dt = - (vec_t[:, 1:] - vec_t[:, :-1])

    if save_trajectory:
      def loop_body(i, val):
        rng, x, x_mean, x_all = val
        vec = vec_t[:, i], vec_t[:, i + 1]
        rng, step_rng = random.split(rng)
        new_x, x_mean = predictor_update_fn(step_rng, state, x, vec, interval_t[:, i], total_interval=dt[:, i])
        x_all.append(jnp.expand_dims(new_x, axis=0))
        x_all.pop(0)
        return rng, new_x, x_mean, x_all

      x_all = [jnp.zeros([1, *shape])] * sde.N # Dummy definition of all x

    else:
      def loop_body(i, val):
        rng, x, x_mean, curv_diff = val
        vec = vec_t[:, i], vec_t[:, i + 1]
        rng, step_rng = random.split(rng)
        new_x, x_mean = predictor_update_fn(step_rng, state, x, vec, interval_t[:, i], total_interval=dt[:, i])
        curv_diff.append(jnp.expand_dims(new_x - x, axis=0))
        curv_diff.pop(0)
        return rng, new_x, x_mean, curv_diff

    curv_diff = [jnp.zeros([1, *shape])] * sde.N # Dummy definition of curvatures
    

    if gen_reflow:
      assert cond_image is not None
      rng, step_rng = random.split(rng)
      noise = sde.prior_sampling(step_rng, shape)
      initial_image = batch_mul(1. - vec_t[0], cond_image) + batch_mul(vec_t[0], noise)
      rng, step_rng = random.split(rng)
      assert not save_trajectory
      _, x, x_mean, _ = jax.lax.fori_loop(0, sde.N, loop_body, (rng, initial_image, initial_image, curv_diff))
    else:
      assert sum(timestep_dict['n_steps_interval']) == sde.N
      rng, step_rng = random.split(rng)
      initial_image = sde.prior_sampling(step_rng, shape)
      x = initial_image
      mid_images = [x]
      index_start = 0
      for rf_div in range(reflow_t):
        if save_trajectory:
          _, x, x_mean, x_all = jax.lax.fori_loop(index_start, index_start + timestep_dict['n_steps_interval'][rf_div], loop_body, (rng, x, x, x_all))
        else:
          _, x, x_mean, curv_diff = jax.lax.fori_loop(index_start, index_start + timestep_dict['n_steps_interval'][rf_div], loop_body, (rng, x, x, curv_diff))
        index_start += timestep_dict['n_steps_interval'][rf_div]
        mid_images.append(jax.lax.stop_gradient(x))

    # Calculate statistics using curvature statistics, if required.
    if save_trajectory:
      stats = jnp.concatenate(x_all, axis=0)
    
    else:
      stats = dict()
      if not gen_reflow:
        curv_diff = jnp.concatenate(curv_diff, axis=0)      # array of (x_t' - x_t), (sde.N, B, H, W, C)
        t_diff = jnp.transpose(- dt)               # array of (t' - t),     (sde.N, B)
        lambda_mult = lambda a, b: a * b
        curv_derivative = jax.vmap(jax.vmap(lambda_mult, (0, 0), 0), (1, 1), 1)(curv_diff, 1. / t_diff) # (sde.N, B, H, W, C) of curvature derivative
        
        # Calculate (marginal) straightness.
        marginal_diff = x - initial_image                   # x_0 - x_1,             (B, H, W, C)
        straightness_gap = jnp.sum(jnp.square(- marginal_diff - curv_derivative), axis=(2, 3, 4)) # || (x_1 - x_0) - d/dt x_t  ||_2^2, (sde.N, B)
        straightness = jnp.mean(jnp.sum(batch_mul(- t_diff, straightness_gap), axis=0)) # (1,)
        straightness_by_t = jnp.mean(straightness_gap, axis=1) # (sde.N)

        # Calculate sequential straightness.
        mid_images = jnp.concatenate([jnp.expand_dims(m, axis=0) for m in mid_images], axis=0) # (reflow_t + 1, B, H, W, C)
        seq_diff = mid_images[1:] - mid_images[:-1] # (reflow_t, B, H, W, C)
        seq_straightness_gap = jnp.zeros((0, shape[0]))
        current_index = 0
        for r in range(reflow_t):
          next_index = current_index + timestep_dict['n_steps_interval'][r]
          mid_marginal_diff = seq_diff[r] / (timestep_dict['interval'][r + 1] - timestep_dict['interval'][r]) # (B, H, W, C)
          mid_curv_derivative = curv_derivative[current_index:next_index]                    # (div[r], B, H, W, C)
          part_gap = jnp.sum(jnp.square(mid_marginal_diff - mid_curv_derivative), axis=(2, 3, 4)) # (n_div[r], B)
          seq_straightness_gap = jnp.concatenate([seq_straightness_gap, part_gap], axis=0)        # finally (sde.N, B)
          current_index = next_index

        assert seq_straightness_gap.shape[0] == sde.N
        seq_straightness = jnp.mean(jnp.sum(batch_mul(- t_diff, seq_straightness_gap), axis=0)) # (1,)
        seq_straightness_by_t = jnp.mean(seq_straightness_gap, axis=1) # (sde.N,)

        stats['straightness'] = straightness
        stats['straightness_by_t'] = straightness_by_t
        stats['seq_straightness'] = seq_straightness
        stats['seq_straightness_by_t'] = seq_straightness_by_t
        stats['nfe'] = sde.N * nfe_multiplier

        stats['interval'] = vec_t[0] # (sde.N + 1)

        del curv_diff, t_diff, curv_derivative, marginal_diff, mid_images, seq_diff, seq_straightness_gap, part_gap, mid_marginal_diff, mid_curv_derivative

    return ((
      inverse_scaler(x_mean if denoise else x), inverse_scaler(initial_image)
    ), (
      vec_t[:, -1:], # ending point
      vec_t[:, 0:1], # starting point
    ),
    stats)

  return jax.pmap(rf_sampler, axis_name='batch')


def get_ode_sampler(sde, shape, inverse_scaler, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,
                    gen_reflow=False, reflow_t=1, adaptive_interval=False):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.

  Returns:
    A sampling function that takes random states, and a replicated training state and returns samples
    as well as the number of function evaluations during sampling.

  Note: the ode sampler is not available of generating randomized interval in SeqRF.
  """
  if adaptive_interval:
    optimal_interval = np.load('assets/c10_adaptive_index_500.npy', allow_pickle=True)[()][reflow_t]
    optimal_interval = 1 - np.array(optimal_interval) / optimal_interval[-1] # 1: noise, 0: data.
    assert optimal_interval.shape[0] == reflow_t + 1
    timestep_dict = get_timestep(reflow_t, sde.N, optimal_interval)
  else:
    timestep_dict = get_timestep(reflow_t, sde.N)

  @jax.pmap
  def drift_fn(state, x, t, index):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, state, state.opt_state_ema.ema, train=False, eps=eps)
    return score_fn(x, t, index) # (sde.T - eps) multiplied for bug in the original RF code.
  
  def ode_sampler(prng, pstate, cond_image=None, **kwargs):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      prng: An array of random state. The leading dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      cond_image: conditional image for generating initial `noisy image`. shape (n_tpu, B // n_tpu, H, W, C)
    Returns:
      Samples, and the number of function evaluations.
    """
    def ode_func_(t, x, index):
      """
      Input
        t:     time, float.
        x:     flattened numpy array, shape (B * H * W * C,)
        index: head index, int.

      Return
        Flattened drift function output, shape (B * H * W * C,)
      """
      x = from_flattened_numpy(x, (jax.local_device_count(),) + shape) # (n_tpu, B // n_tpu, H, W, C)
      t = rescale_time(t, to='diffusion') # (n_tpu, B), (eps -> 1) scale to (1 -> 0) scale.
      vec_ = jnp.ones((x.shape[0], x.shape[1])) * t # (n_tpu, B // n_tpu)
      drift = drift_fn(pstate, x, vec_, jnp.ones_like(vec_, jnp.int32) * index)
      print(rescale_time(t, to='rf'), to_flattened_numpy(drift)[0:3])
      return to_flattened_numpy(drift)

    ode_func = []
    for idx in range(reflow_t):
      ode_func.append(functools.partial(ode_func_, index=idx))

    # Initial sample
    rng = flax.jax_utils.unreplicate(prng)
    rng, step_rng = random.split(rng)
    if cond_image is None:
      # generate from noise.
      initial_image = sde.prior_sampling(step_rng, (jax.local_device_count(),) + shape) # shape (n_tpu, B // n_tpu, H, W, C)
    else:
      # initial image is linear interpolation between noise and image. (e.g. noisy image) shape: (n_tpu, B // n_tpu, H, W, C)
      initial_image = jnp.reshape(cond_image, (jax.local_device_count(),) + shape)

    if gen_reflow:
      assert kwargs['batch_idx'] is not None
      assert isinstance(kwargs['batch_idx'], int)
      assert kwargs['batch_idx'] in jnp.arange(reflow_t)
      start_t_batch = timestep_dict['interval'][kwargs['batch_idx']]
      end_t_batch = timestep_dict['interval'][kwargs['batch_idx'] + 1]
      rng, step_rng = random.split(rng)
      noise = sde.prior_sampling(step_rng, (jax.local_device_count(),) + shape)
      initial_image = (1. - start_t_batch) * cond_image + start_t_batch * noise
    else:
      rng, step_rng = random.split(rng)
      initial_image = sde.prior_sampling(step_rng, (jax.local_device_count(),) + shape) # shape (n_tpu, B // n_tpu, H, W, C)

    # Black-box ODE solver for the probability flow ODE
    solution = []
    mid_images = [initial_image]
    n_points_per_seq = []
    interval_resized = rescale_time(timestep_dict['interval'], to='rf')
    if gen_reflow: # generate reflow dataset
      solution.append(integrate.solve_ivp(ode_func[kwargs['batch_idx']], (interval_resized[kwargs['batch_idx']], interval_resized[kwargs['batch_idx'] + 1]),
                                          to_flattened_numpy(initial_image), rtol=rtol, atol=atol, method=method))
      n_points_per_seq.append(solution[-1].t.shape[0]) # (1,)
    else: # sample
      current_image = to_flattened_numpy(initial_image)
      for rf_div in range(reflow_t):
        solution.append(integrate.solve_ivp(ode_func[rf_div], (interval_resized[rf_div], interval_resized[rf_div + 1]),
                                            current_image, rtol=rtol, atol=atol, method=method))

        current_image = solution[-1].y[:, -1]
        mid_images.append(from_flattened_numpy(current_image, (jax.local_device_count(),) + shape)) # (n_tpu, B // n_tpu, H, W, C)
        n_points_per_seq.append(solution[-1].t.shape[0])

    if not gen_reflow:
      nfe = jnp.sum(jnp.asarray([s.nfev for s in solution]))
      n_points_per_seq = jnp.asarray(n_points_per_seq) # (1,) or (reflow_t,), resp.
      
      mid_images = [jnp.expand_dims(m, axis=-1) for m in mid_images]
      mid_images = jnp.concatenate(mid_images, axis=-1) # (n_tpu, B // n_tpu, H, W, C, reflow_t + 1)
      t = jnp.asarray(jnp.concatenate([s.t[:-1] for s in solution])) # (n_points,)
      t = jnp.concatenate([t, jnp.expand_dims(solution[-1].t[-1], -1)])
      x_all = jnp.asarray(jnp.concatenate([s.y[:, :-1] for s in solution], axis=-1)) # (B * H * W * C, n_points)
      x_all = jnp.concatenate([x_all, solution[-1].y[:, -1:]], axis=-1)
      x_all = x_all.reshape((jax.local_device_count(),) + shape + (x_all.shape[1],)) # x_all: jnp.array of the trajectory, shape (n_tpu, B // n_tpu, H, W, C, n_points)
      x = x_all[:, :, :, :, :, -1] # shape (n_tpu, B, H, W, C)

      # Calculate statistics using curvature statistics, if required.
      dt = t[1:] - t[:-1]                                                                          # tnext - t, (rf scale) (n_points - 1,)
      x_all = jnp.transpose(x_all, (5, 0, 1, 2, 3, 4))
      x_all = jnp.reshape(x_all,
        (x_all.shape[0], x_all.shape[1] * x_all.shape[2]) + x_all.shape[3:])                       # (n_points, B, H, W, C)

    else:
      x = solution[-1].y[:, -1:]
      t = solution[-1].t
      del solution
      x = jnp.reshape(x, initial_image.shape)

    stats = dict()
    # In the following cases, B <- n_tpu * B.
    if not gen_reflow:
      curv_diff = x_all[1:] - x_all[:-1]                                                         # x{tnext} - xt,  (n_points - 1, B, H, W, C)
      curv_derivative = batch_mul(curv_diff, 1. / dt)                                            # (x{tnext} - xt) / (tnext - t), (n_points - 1, B, H, W, C)
      
      # Calculate (marginal) straightness.
      marginal_diff = x - initial_image                                                          # x0 - x1, (n_tpu, B, H, W, C)
      marginal_diff = jnp.reshape(marginal_diff, 
        (marginal_diff.shape[0] * marginal_diff.shape[1],) + marginal_diff.shape[2:])            # x0 - x1, (B, H, W, C)
      straightness_gap = jnp.sum(jnp.square(marginal_diff - curv_derivative), axis=(2, 3, 4))    # l2sq((x0 - x1) - d/dt x_t), (n_points - 1, B) # x0 <- x at time 1, x1 <- x at time eps
      straightness = jnp.mean(jnp.sum(batch_mul(dt, straightness_gap), axis=0))                  # (1,)
      straightness_by_t = jnp.mean(straightness_gap, axis=1)                                     # (n_points - 1,)

      # Calculate sequential straightness.
      mid_images = jnp.transpose(mid_images, (5, 0, 1, 2, 3, 4))
      mid_images = jnp.reshape(mid_images,
        (mid_images.shape[0], mid_images.shape[1] * mid_images.shape[2]) + mid_images.shape[3:]) # (reflow_t + 1, B, H, W, C)
      seq_diff = mid_images[1:] - mid_images[:-1]                                                # (reflow_t, B, H, W, C)
      seq_straightness_gap = jnp.zeros((0, jax.local_device_count() * shape[0]))
      current_index = 0
      interval_resized = rescale_time(timestep_dict['interval'], 'rf')
      for r in range(reflow_t):
        next_index = current_index + n_points_per_seq[r]
        mid_marginal_diff = seq_diff[r] / (interval_resized[r + 1] - interval_resized[r])       # (B, H, W, C)
        mid_curv_derivative = curv_derivative[current_index:next_index]                         # (div[r], B, H, W, C)
        part_gap = jnp.sum(
          jnp.square(mid_marginal_diff - mid_curv_derivative), axis=(2, 3, 4))                  # (n_div[r], B)
        seq_straightness_gap = jnp.concatenate([seq_straightness_gap, part_gap], axis=0)        # finally (n_points - 1, B)
        current_index = next_index

      seq_straightness = jnp.mean(jnp.sum(batch_mul(dt, seq_straightness_gap), axis=0))         # (1,)
      seq_straightness_by_t = jnp.mean(seq_straightness_gap, axis=1)                            # (n_points,)

      # For RK45 solver, NFE = (n_points - 1) * 6.
      stats['straightness'] = straightness                   # (1,) (c.f. ODE solver: (n_tpu,) after passing pmap)
      stats['straightness_by_t'] = straightness_by_t         # (n_points - 1,)
      stats['seq_straightness'] = seq_straightness           # (1,)
      stats['seq_straightness_by_t'] = seq_straightness_by_t # (n_points - 1,)
      stats['nfe'] = nfe                                     # (1,)
      stats['interval'] = t                                  # (n_points,)

      # save trajectory if needed
      if ('save_trajectory' in kwargs) and kwargs['save_trajectory']:
        stats['trajectory'] = x_all

      del curv_diff, dt, curv_derivative, marginal_diff, mid_images, seq_diff, seq_straightness_gap, part_gap, mid_marginal_diff, mid_curv_derivative, x_all

    # x, initial_image: shape (n_tpu, B // n_tpu, H, W, C)
    # times: shape (n_tpu, B // n_tpu)
    return ((inverse_scaler(x), inverse_scaler(initial_image)), \
      (jnp.ones((x.shape[0], x.shape[1])) * rescale_time(t[-1], "diffusion"), jnp.ones((x.shape[0], x.shape[1])) * rescale_time(t[0], "diffusion")), stats)

  return ode_sampler
