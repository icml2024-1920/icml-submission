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

"""All functions related to loss computation and optimization.
"""

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from sde_lib import VESDE, VPSDE, I2SBSDE, RFSDE, EDMSDE
from utils import batch_mul, time_to_index
import optax
import numpy as np

from flax.traverse_util import flatten_dict

def get_optimizer(config):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    warmup = config.optim.warmup
    if warmup > 0:
      warmup_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                                           peak_value=config.optim.lr,
                                                           warmup_steps=warmup,
                                                           decay_steps=config.training.n_iters,
                                                           end_value=config.optim.lr)
    optimizer = optax.adamw(learning_rate=warmup_schedule, b1=config.optim.beta1, eps=config.optim.eps, weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(state,
                  grad,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    # Fast parameter update
    if grad_clip >= 0:
      # Compute global gradient norm
      grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad)]))
      # Clip gradient
      clipped_grad = jax.tree_map(
        lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
    else:  # disabling gradient clipping if grad_clip < 0
      clipped_grad = grad
    state = state.apply_gradients(grads=clipped_grad) # variables['params'] updated.

    # Delayed parameter (EMA) update
    initial_count = 0 if config.model.rf_phase == 1 else config.model.initial_count
    if config.model.variable_ema_rate:
      ema_decay = jnp.minimum(config.model.ema_rate, (1 + (initial_count + state.step)) / (10 + (initial_count + state.step)))
    else:
      ema_decay = config.model.ema_rate
    updates, new_opt_state_ema = state.tx_ema.update(
      state.params, state.opt_state_ema, ema_decay
    )
    state = state.replace(opt_state_ema=new_opt_state_ema)

    return state

  return optimize_fn


def get_rf_loss_fn(sde, state, train, reduce_mean=True, reflow=False, distill=False, eps=1e-5, reflow_t=1, adaptive=False, rf_distill=None):
  """
    Create a loss function for training RFs. (rectified flow)
    
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, batch):
    score_fn = mutils.get_score_fn(sde, state, params, train=train, eps=eps)
    """
      data configuration
      Output
        x = (original data, degraded data or noise)
    """
    # adaptive interval
    adaptive_interval = 1 - jnp.array(np.load('assets/c10_adaptive_index_500.npy', allow_pickle=True)[()][reflow_t]) / 500 if adaptive else None

    if reflow or distill:
      x = batch[0]     # tuple of (less noisy data, more noisy data)
      t_ref = batch[1] # tuple of (end_time, start_time) in (1-->0) sampling. ex) (end_time, start_time) = (0.0, 0.25)

      # For patching the bug, we first index the midpoint.
      index = time_to_index((t_ref[0] + t_ref[1]) / 2., reflow_t, adaptive_interval)

      if (reflow and distill):
        # train_rf_distill
        assert rf_distill is not None
        assert reflow_t == 1
        rng, step_rng = random.split(rng)
        t = (random.randint(step_rng, (x[0].shape[0],), minval=0, maxval=rf_distill) + 1) / rf_distill
      elif reflow:
        # train_reflow
        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (x[0].shape[0],), minval=0.0, maxval=1) # interpolation level uniformly from (0, 1)
      elif distill:
        # train_distill
        t = jnp.ones((x[0].shape[0],)) # (1,)
      else:
        raise NotImplementedError()
      
      perturbed_data = batch_mul(x[0], 1 - t) + batch_mul(x[1], t)
      t_ = t_ref[0] * (1 - t) + t_ref[1] * t
      score = score_fn(perturbed_data, t_, index, rng)

      # loss function (l2)
      # score_ref = batch_mul(x[1] - x[0], 1 / (t_ref[1] - t_ref[0]))
      score_ref = - batch_mul(x[0] - x[1], 1 / (t_ref[0] - t_ref[1]))
      losses = jnp.square(score_ref - score)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)

    else:
      data = batch['image']

      rng, step_rng = random.split(rng)
      z = random.normal(step_rng, data.shape)
      x = (data, z)
      
      rng, step_rng = random.split(rng)
      t = random.uniform(step_rng, (x[0].shape[0],), minval=0.0, maxval=sde.T)
      index = time_to_index(t, reflow_t, adaptive_interval)
      perturbed_data = batch_mul(x[0], 1 - t) + batch_mul(x[1], t)
      score = score_fn(perturbed_data, t, index, rng)

      # loss function
      losses = jnp.square((x[0] - x[1]) - score)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)

    loss = jnp.mean(losses)
    return loss

  return loss_fn


def get_step_fn(config, sde, state, train, optimize_fn=None, reduce_mean=False, eps=0.0, reflow_t=1):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    state: `flax.training.TrainState` object.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    reflow_t: number of reflow steps.

  Returns:
    A one-step function for training or evaluation.
  """
  loss_fn = get_rf_loss_fn(sde, state, train, reduce_mean=reduce_mean,
                           reflow=(config.training.reflow_mode in ['train_reflow', 'train_rf_distill'] and config.model.rf_phase > 1),
                           distill=(config.training.reflow_mode in ['train_distill', 'train_rf_distill'] and config.model.rf_phase > 1),
                           eps=eps,
                           reflow_t=reflow_t,
                           adaptive=config.training.adaptive_interval,
                           rf_distill=int(config.model.num_scales // config.training.reflow_t) if config.training.reflow_mode == 'train_rf_distill' else None)

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.training.TrainState` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    # rng = jax.random.fold_in(rng, state.step) # New RNGsplitter.
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1)
    if train: # training
      loss, grad = grad_fn(step_rng, state.params, batch)
      grad = jax.lax.pmean(grad, axis_name='batch')
      state = optimize_fn(state, grad) # Optimize params with (Adam) optimizer
      state = state.replace(step=state.step + 1) # Update step
    else: # evaluation or sampling
      loss = loss_fn(step_rng, state.opt_state_ema.ema, batch)

    loss = jax.lax.pmean(loss, axis_name='batch')
    return (rng, state), loss

  return step_fn


def get_loss_fn_exp(sde, state, train, t_, reduce_mean=True, eps=1e-5, reflow_t=1):
  """
    Create a loss function for training RFs. (rectified flow)
    
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, batch):
    score_fn = mutils.get_score_fn(sde, state, params, train=train, eps=eps)
    """
      data configuration
      Output
        x = (original data, degraded data or noise)
    """
    x = batch # (B, H, W, C), noiseless image.
    t = jnp.full((x.shape[0],), t_) # (B,), should be a fixed time.
    small_z = 0.0001

    # For patching the bug, we first index the midpoint.
    index = time_to_index(t, reflow_t)

    rng, step_rng = random.split(rng)
    n1 = random.normal(step_rng, x.shape)
    rng, step_rng = random.split(rng)
    n2 = random.normal(step_rng, x.shape) # perturb to compare score output.
    perturbed_data = batch_mul(x, 1 - t) + batch_mul(n1, t)
    perturbed_data_again = perturbed_data + n2 * small_z

    score = score_fn(perturbed_data, t, index, rng)
    score_perturbed = score_fn(perturbed_data_again, t, index, rng)

    score_diff = score_perturbed - score
    score_diff = jnp.sqrt(jnp.mean(score_diff ** 2, (1, 2, 3)))
    perturbed_diff = perturbed_data_again - perturbed_data
    x_diff = jnp.sqrt(jnp.mean(perturbed_diff ** 2, (1, 2, 3)))

    lipschitz = score_diff / x_diff

    # loss function (l2)
    score_ref = x - n1
    losses = jnp.square(score_ref - score)
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    loss = jnp.mean(losses)
    return loss, lipschitz

  return loss_fn


def get_step_fn_playground(sde, state, train, reflow_t=1, t=0.0):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    state: `flax.training.TrainState` object.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    reflow_t: number of reflow steps.

  Returns:
    A one-step function for training or evaluation.
  """
  loss_fn = get_loss_fn_exp(sde, state, train, reduce_mean=True, eps=1e-3, reflow_t=reflow_t, t_=t)

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.training.TrainState` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    (_, lipschitz), grad = grad_fn(step_rng, state.opt_state_ema.ema, batch)
    grad = jax.lax.pmean(grad, axis_name='batch')
    grad = flatten_dict(grad)
    # grad_sq_norm = [jnp.linalg.norm(grad[g] ** 2) for g in grad]
    grad_sq_norm = 0
    for g in grad:
      grad_sq_norm += jnp.linalg.norm(grad[g]**2)
    return (rng, state), (lipschitz, grad_sq_norm)

  return step_fn
