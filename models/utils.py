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

"""All functions and modules related to model definition.
"""
from typing import Any, Callable, Optional

import flax
import functools
import jax.numpy as jnp
import sde_lib
import jax
import optax
import numpy as np
from models import wideresnet_noise_conditional
from flax.training import train_state
import orbax.checkpoint
from utils import batch_mul, rescale_time
from losses import optimization_manager, get_optimizer
from flax.core.frozen_dict import FrozenDict, freeze
from flax.traverse_util import flatten_dict, unflatten_dict
import torch
from optax._src import base as obase
from optax._src import utils as outils
from optax._src.transform import EmaState, update_moment, bias_correction

from optax._src.base import GradientTransformation

_MODELS = {}

def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = jnp.exp(
    jnp.linspace(
      jnp.log(config.model.sigma_max), jnp.log(config.model.sigma_min),
      config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def init_train_state(rng, config) -> train_state.TrainState:
  # Get model parameters and definitions
  model_name = config.model.name
  model_def = functools.partial(get_model(model_name), config=config)
  model = model_def()

  input_shape = (jax.local_device_count(), config.data.image_size, config.data.image_size, config.data.num_channels)
  label_shape = input_shape[:1]
  fake_input = jnp.zeros(input_shape)
  fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
  fake_index = jnp.zeros(label_shape, dtype=jnp.int32)
  params_rng, dropout_rng = jax.random.split(rng)
  
  variables = model.init(params_rng, fake_input, fake_label, fake_index, train=False)
  # print(variables)

  # Create optimizer
  optimizer = get_optimizer(config)
  optimizer_ema = variable_ema(initial_count=config.model.initial_count) # customized API for variable rate
  class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    params: FrozenDict[str, Any]
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    tx_ema: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState
    opt_state_ema: optax.OptState
    dropout_rng: jax.Array

    def apply_gradients(self, *, grads, **kwargs):
      """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

      Note that internally this function calls `.tx.update()` followed by a call
      to `optax.apply_updates()` to update `params` and `opt_state`.

      Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

      Returns:
        An updated instance of `self` with `step` incremented by one, `params`
        and `opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
      """
      updates, new_opt_state = self.tx.update(
          grads, self.opt_state, self.params)
      new_params = optax.apply_updates(self.params, updates)
      return self.replace(
          params=new_params,
          opt_state=new_opt_state,
          **kwargs,
      )

    @classmethod
    def create(cls, *, apply_fn, params, tx, tx_ema, dropout_rng, **kwargs):
      """Creates a new instance with `step=0` and initialized `opt_state`."""
      opt_state = tx.init(params)
      opt_state_ema = tx_ema.init(params)
      return cls(
          step=0,
          apply_fn=apply_fn,
          params=params,
          tx=tx,
          tx_ema=tx_ema,
          opt_state=opt_state,
          opt_state_ema=opt_state_ema,
          dropout_rng=dropout_rng,
          **kwargs,
      )
  return TrainState.create(
    apply_fn=model.apply,
    params=variables['params'], # main parameter
    tx=optimizer, # (Adam) Optimizer
    tx_ema=optimizer_ema, # EMA state that includes delayed EMA parameter
    dropout_rng=dropout_rng,
  )


def get_model_fn(state, params, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    state: A `flax.training.TrainState` object that represent the whole training state.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels, index, rng=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      index: A minibatch of sequential index of multi-head timestep embedding. int
      rng: If present, it is the random state for dropout

    Returns:
      model output
    """
    if not train:
      return state.apply_fn({'params': params}, x, labels, index, train=False)
    else:
      return state.apply_fn({'params': params}, x, labels, index, train=True, rngs={'dropout': rng})

  return model_fn


def get_score_fn(sde, state, params, train=False, eps=0.0):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    state: A `flax.training.TrainState` object that represent the whole training state.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(state, params, train=train)

  def score_fn(x, t, index, rng=None):
    """
    Input
      x:     (B, H, W, C), (scaled) image
      t:     (B,), 'diffusion' type input. Sampled with 1-->0 interval.
      index: (B,), head selecting module.
    """
    labels = rescale_time(t, to='rf') * 999
    score = model_fn(x, labels, index, rng)
    return score

  return score_fn



def variable_ema(
  debias: bool = True,
  accumulator_dtype: Optional[Any] = None,
  initial_count: int = 0
) -> obase.GradientTransformation:
  """
    ema with variable ema rate. ema rate is entered in the update_fn rather than being initialiized.
  """
  accumulator_dtype = outils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    return EmaState(
        count=jnp.ones([], jnp.int32) * initial_count,
        ema=jax.tree_util.tree_map(
            lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

  def update_fn(updates, state, decay, params=None):
    del params
    updates = new_ema = update_moment(updates, state.ema, decay, order=1)
    count_inc = outils.safe_int32_increment(state.count)
    if debias:
      updates = bias_correction(new_ema, decay, count_inc)
    state_ema = outils.cast_tree(new_ema, accumulator_dtype)
    return updates, EmaState(count=count_inc, ema=state_ema)

  return obase.GradientTransformation(init_fn, update_fn)


def flax_to_flax_ckpt(baseline_flax_state, flax_state, reflow_t):
  """
  Temporary. 
  """
  if reflow_t == 1:
    return baseline_flax_state

  baseline_flattened_model = flatten_dict(baseline_flax_state.params)
  baseline_flattened_ema = flatten_dict(baseline_flax_state.opt_state_ema.ema)
  baseline_flattened_mu = flatten_dict(baseline_flax_state.opt_state[0].mu)
  baseline_flattened_nu = flatten_dict(baseline_flax_state.opt_state[0].nu)

  flattened_model = flatten_dict(flax_state.params)
  flattened_ema = flatten_dict(flax_state.opt_state_ema.ema)
  flattened_mu = flatten_dict(flax_state.opt_state[0].mu)
  flattened_nu = flatten_dict(flax_state.opt_state[0].nu)

  for k_b, k in zip(baseline_flattened_model, flattened_model):
    assert k_b == k
    if baseline_flattened_model[k_b].shape == flattened_model[k_b].shape:
      flattened_model[k_b] = baseline_flattened_model[k_b]
    else:
      if len(flattened_model[k_b].shape) == 2:
        assert flattened_model[k_b].shape[1] == baseline_flattened_model[k_b].shape[1] * reflow_t
        flattened_model[k_b] = jnp.tile(baseline_flattened_model[k_b], (1, reflow_t,))
      elif len(flattened_model[k_b].shape) == 1:
        assert flattened_model[k_b].shape[0] == baseline_flattened_model[k_b].shape[0] * reflow_t
        flattened_model[k_b] = jnp.tile(baseline_flattened_model[k_b], (reflow_t,))
      else:
        raise ValueError()

  for k_b, k in zip(baseline_flattened_ema, flattened_ema):
    assert k_b == k
    if baseline_flattened_ema[k_b].shape == flattened_ema[k_b].shape:
      flattened_ema[k_b] = baseline_flattened_ema[k_b]
    else:
      if len(flattened_ema[k_b].shape) == 2:
        assert flattened_ema[k_b].shape[1] == baseline_flattened_ema[k_b].shape[1] * reflow_t
        flattened_ema[k_b] = jnp.tile(baseline_flattened_ema[k_b], (1, reflow_t,))
      elif len(flattened_ema[k_b].shape) == 1:
        assert flattened_ema[k_b].shape[0] == baseline_flattened_ema[k_b].shape[0] * reflow_t
        flattened_ema[k_b] = jnp.tile(baseline_flattened_ema[k_b], (reflow_t,))
      else:
        raise ValueError()

  for k_b, k in zip(baseline_flattened_mu, flattened_mu):
    assert k_b == k
    if baseline_flattened_mu[k_b].shape == flattened_mu[k_b].shape:
      flattened_mu[k_b] = baseline_flattened_mu[k_b]
    else:
      if len(flattened_mu[k_b].shape) == 2:
        assert flattened_mu[k_b].shape[1] == baseline_flattened_mu[k_b].shape[1] * reflow_t
        flattened_mu[k_b] = jnp.tile(baseline_flattened_mu[k_b], (1, reflow_t,))
      elif len(flattened_mu[k_b].shape) == 1:
        assert flattened_mu[k_b].shape[0] == baseline_flattened_mu[k_b].shape[0] * reflow_t
        flattened_mu[k_b] = jnp.tile(baseline_flattened_mu[k_b], (reflow_t,))
      else:
        raise ValueError()

  for k_b, k in zip(baseline_flattened_nu, flattened_nu):
    assert k_b == k
    if baseline_flattened_nu[k_b].shape == flattened_nu[k_b].shape:
      flattened_nu[k_b] = baseline_flattened_nu[k_b]
    else:
      if len(flattened_mu[k_b].shape) == 2:
        assert flattened_mu[k_b].shape[1] == baseline_flattened_nu[k_b].shape[1] * reflow_t
        flattened_mu[k_b] = jnp.tile(baseline_flattened_nu[k_b], (1, reflow_t,))
      elif len(flattened_nu[k_b].shape) == 1:
        assert flattened_nu[k_b].shape[0] == baseline_flattened_nu[k_b].shape[0] * reflow_t
        flattened_nu[k_b] = jnp.tile(baseline_flattened_nu[k_b], (reflow_t,))
      else:
        raise ValueError()
  

  # update Adam state
  adam_opt_state = optax._src.transform.ScaleByAdamState(
    count=300000,
    mu=unflatten_dict(flattened_mu),
    nu=unflatten_dict(flattened_nu)
  )
  biascor_opt_state = optax._src.transform.ScaleByScheduleState(
    count=300000
  )
  new_opt_state = (
    adam_opt_state,
    optax._src.base.EmptyState(),
    biascor_opt_state
  )

  # update EMA state
  new_opt_state_ema = optax._src.transform.EmaState(
    count=300000,
    ema=unflatten_dict(flattened_ema)
  )

  flax_state = flax_state.replace(
    opt_state=new_opt_state,
    opt_state_ema=new_opt_state_ema,
    params=unflatten_dict(flattened_model)
  )

  return flax_state



def torch_to_flax_ckpt(torch_ckpt_path, flax_state, reflow_t, initial_count, embedding_type):
  """
    Input
      torch_ckpt_path
        Path of torch checkpoint, type str.
      flax_state
        Flax state of the model, type flax.training.TrainState.
      reflow_t
        number of reflow sequences, int.
      initial_step
        number of initial steps, int.

    Return
      replaced_flax_state
        Ported flax state of the model, type flax.training.TrainState.

    (torch_ckpt documentation)
      'optimizer'
        'state': state of the (AdamW) optimizer, type dict with keys in int.
          i: tensor index, type int
            'step': type int
            'exp_avg': type torch.Tensor
            'exp_avg_sq': type torch.Tensor
        'param_groups': (AdamW) optimizer parameters, type length-one list, that consists of a dict.
      'model': The model parameters, type dict
        'modules.sigma': (dummy) sigma
        'modules.all_modules.*': parameters
      'ema'
        'decay': EMA decay, type float
        'num_updates': number of steps, type int
        'shadow_params': EMA parameters, type list
      'step': number of steps, type int

      torch_ckpt['model'] (except sigmas) --> unflatten_dict(flax_state.params)           (main parameter)
      torch_ckpt['model'] (except sigmas) --> unflatten_dict(flax_state.states['params']) (main parameter)
      torch_ckpt['ema']['shadow_params']  --> unflatten_dict(flax_state.tx_ema['ema'])    (EMA parameter)
  """
  # Initialize torch checkpoint
  torch_ckpt = torch.load(torch_ckpt_path, map_location=torch.device('cpu'))
  torch_ckpt['model'].pop('module.sigmas')

  # validate tree (list) length
  n_leaves = len(flatten_dict(flax_state.params))
  assert n_leaves == len(torch_ckpt['model']), f"flax # leaf: {n_leaves}, torch # leaf: {len(torch_ckpt['model'])}"
  if embedding_type == 'positional':
    assert n_leaves == len(torch_ckpt['ema']['shadow_params']), \
      f"flax ema # leaf: {n_leaves}, torch ema # leaf: {len(torch_ckpt['ema']['shadow_params'])}"
  elif embedding_type == 'fourier':
    assert n_leaves == len(torch_ckpt['ema']['shadow_params']) + 1, \
      f"flax ema # leaf: {n_leaves}, torch ema # leaf: {len(torch_ckpt['ema']['shadow_params'])}"
  else:
    raise NotImplementedError()

  flattened_model = flatten_dict(flax_state.params)
  # port ema
  flattened_ema = flatten_dict(flax_state.opt_state_ema.ema)
  # port AdamW parameters
  flattened_mu = flatten_dict(flax_state.opt_state[0].mu)
  flattened_nu = flatten_dict(flax_state.opt_state[0].nu)

  # timestep lists
  if embedding_type == 'positional':
    timestep_weight_prefix_list = ['module.all_modules.0.weight', 'module.all_modules.1.weight']
    timestep_bias_prefix_list = ['module.all_modules.0.bias', 'module.all_modules.1.bias']
  elif embedding_type == 'fourier':
    timestep_weight_prefix_list = ['module.all_modules.1.weight', 'module.all_modules.2.weight']
    timestep_bias_prefix_list = ['module.all_modules.1.bias', 'module.all_modules.2.bias']
    torch_ckpt['ema']['shadow_params'].insert(0, torch_ckpt['model']['module.all_modules.0.W'])
  else:
    raise ValueError()

  j = 0

  for model_t, model_j, ema_t in zip(torch_ckpt['model'], flattened_model, torch_ckpt['ema']['shadow_params']):
    
    # torch_mu and torch_nu definition.
    if not (j == 0 and embedding_type == 'fourier'):
      # 'fourier' and j=0 --> fourier_W, which is not updated.
      torch_mu = torch_ckpt['optimizer']['state'][j]['exp_avg']
      torch_nu = torch_ckpt['optimizer']['state'][j]['exp_avg_sq']
    # the key 'ema_j' is not aligned correctly: use 'model_j' instead.
    assert len(torch_ckpt['model'][model_t].shape) in [1, 2, 4]
    if len(torch_ckpt['model'][model_t].shape) == 1:
      if model_t in timestep_bias_prefix_list: # duplicate timestep embedding
        # model
        flattened_model[model_j] = jnp.asarray(torch.tile(torch_ckpt['model'][model_t], (reflow_t,)).detach())
        # ema
        flattened_ema[model_j] = jnp.asarray(torch.tile(ema_t, (reflow_t,)).detach())
        if not (j == 0 and embedding_type == 'fourier'):
          # optimizer param (mu)
          flattened_mu[model_j] = jnp.asarray(torch.tile(torch_mu, (reflow_t,)).detach())
          # optimizer param (nu)
          flattened_nu[model_j] = jnp.asarray(torch.tile(torch_nu, (reflow_t,)).detach())
      else:
        assert tuple(torch_ckpt['model'][model_t].shape) == flattened_model[model_j].shape
        # model param
        flattened_model[model_j] = jnp.asarray(torch_ckpt['model'][model_t].detach())
        # ema param
        flattened_ema[model_j] = jnp.asarray(ema_t.detach())
        if not (j == 0 and embedding_type == 'fourier'):
          # optimizer param (mu)
          flattened_mu[model_j] = jnp.asarray(torch_mu.detach())
          # optimizer param (nu)
          flattened_nu[model_j] = jnp.asarray(torch_nu.detach())

    elif len(torch_ckpt['model'][model_t].shape) == 2:
      if model_t in timestep_weight_prefix_list: # duplicate timestep embedding
        # model param
        transposed_tensor = torch_ckpt['model'][model_t].permute(1, 0)
        flattened_model[model_j] = jnp.asarray(torch.tile(transposed_tensor, (1, reflow_t)).detach())
        # ema param
        transposed_tensor = ema_t.permute(1, 0)
        flattened_ema[model_j] = jnp.asarray(torch.tile(transposed_tensor, (1, reflow_t)).detach())
        if not (j == 0 and embedding_type == 'fourier'):
          # optimizer param (mu)
          transposed_tensor = torch_mu.permute(1, 0)
          flattened_mu[model_j] = jnp.asarray(torch.tile(transposed_tensor, (1, reflow_t)).detach())
          # optimizer param (nu)
          transposed_tensor = torch_nu.permute(1, 0)
          flattened_nu[model_j] = jnp.asarray(torch.tile(transposed_tensor, (1, reflow_t)).detach())
      else:
        # model param
        transposed_tensor = torch_ckpt['model'][model_t] if 'NIN' in model_t else torch_ckpt['model'][model_t].permute(1, 0)
        assert tuple(transposed_tensor.shape) == flattened_model[model_j].shape, f"Module {model_t} - {tuple(transposed_tensor.shape)} vs. {flattened_model[model_j].shape}"
        flattened_model[model_j] = jnp.asarray(transposed_tensor.detach())
        # ema param
        transposed_tensor = ema_t if 'NIN' in model_t else ema_t.permute(1, 0)
        flattened_ema[model_j] = jnp.asarray(transposed_tensor.detach())
        if not (j == 0 and embedding_type == 'fourier'):
          # optimizer param (mu)
          transposed_tensor = torch_mu if 'NIN' in model_t else torch_mu.permute(1, 0)
          flattened_mu[model_j] = jnp.asarray(transposed_tensor.detach())
          # optimizer param (nu)
          transposed_tensor = torch_nu if 'NIN' in model_t else torch_nu.permute(1, 0)
          flattened_nu[model_j] = jnp.asarray(transposed_tensor.detach())

    elif len(torch_ckpt['model'][model_t].shape) == 4:
      # model param
      transposed_tensor = torch_ckpt['model'][model_t].permute(2, 3, 1, 0)
      assert tuple(transposed_tensor.shape) == flattened_model[model_j].shape
      flattened_model[model_j] = jnp.asarray(transposed_tensor.detach())
      # ema param
      transposed_tensor = ema_t.permute(2, 3, 1, 0)
      flattened_ema[model_j] = jnp.asarray(transposed_tensor.detach())
      if not (j == 0 and embedding_type == 'fourier'):
        # optimizer param (mu)
        transposed_tensor = torch_mu.permute(2, 3, 1, 0)
        flattened_mu[model_j] = jnp.asarray(transposed_tensor.detach())
        # optimizer param (nu)
        transposed_tensor = torch_nu.permute(2, 3, 1, 0)
        flattened_nu[model_j] = jnp.asarray(transposed_tensor.detach())
    else:
      raise ValueError(f"{[model_j]}")

    j += 1

  # update Adam state
  adam_opt_state = optax._src.transform.ScaleByAdamState(
    count=jnp.ones([], jnp.int32) * initial_count,
    mu=unflatten_dict(flattened_mu),
    nu=unflatten_dict(flattened_nu)
  )
  biascor_opt_state = optax._src.transform.ScaleByScheduleState(
    count=jnp.ones([], jnp.int32) * initial_count
  )
  new_opt_state = (
    adam_opt_state,
    optax._src.base.EmptyState(),
    biascor_opt_state
  )

  # update EMA state
  new_opt_state_ema = optax._src.transform.EmaState(
    count=jnp.ones([], jnp.int32) * initial_count,
    ema=unflatten_dict(flattened_ema)
  )

  flax_state = flax_state.replace(
    opt_state=new_opt_state,
    opt_state_ema=new_opt_state_ema,
    params=unflatten_dict(flattened_model)
  )

  return flax_state


#############################################################################################
# TODO: functions that are not used.
def create_classifier(prng_key, batch_size, ckpt_path, **kwargs):
  """Create a noise-conditional image classifier.

  Args:
    prng_key: A JAX random state.
    batch_size: The batch size of input data.
    ckpt_path: The path to stored checkpoints for this classifier.

  Returns:
    classifier: A `flax.linen.Module` object that represents the architecture of the classifier.
    classifier_params: A dictionary that contains trainable parameters of the classifier.
  """
  input_shape = (batch_size, 32, 32, 3)
  classifier = wideresnet_noise_conditional.WideResnet(
    blocks_per_group=4,
    channel_multiplier=10,
    num_outputs=10
  )
  initial_variables = classifier.init({'params': prng_key, 'dropout': jax.random.PRNGKey(0)},
                                      jnp.ones(input_shape, dtype=jnp.float32),
                                      jnp.ones((batch_size,), dtype=jnp.float32), train=False)
  model_state, init_params = initial_variables.pop('params')
  # classifier_params = checkpoints.restore_checkpoint(ckpt_path, init_params)

  mgr_options = orbax.checkpoint.CheckpointManagerOptions(
    save_interval_steps=kwargs['save_interval_steps'],
    max_to_keep=1,
    create=True)
  ckpt_mgr = orbax.checkpoint.CheckpointManager(
    ckpt_path,
    orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)
  restore_args = flax.training.orbax_utils.restore_args_from_target(init_params, mesh=None)
  classifier_params = ckpt_mgr.restore(step=ckpt_mgr.latest_step(), items=init_params, restore_kwargs={'restore_kwargs': restore_args})

  return classifier, classifier_params


def get_logit_fn(classifier, classifier_params):
  """ Create a logit function for the classifier. """

  def preprocess(data):
    image_mean = jnp.asarray([[[0.49139968, 0.48215841, 0.44653091]]])
    image_std = jnp.asarray([[[0.24703223, 0.24348513, 0.26158784]]])
    return (data - image_mean[None, ...]) / image_std[None, ...]

  def logit_fn(data, ve_noise_scale):
    """Give the logits of the classifier.

    Args:
      data: A JAX array of the input.
      ve_noise_scale: time conditioning variables in the form of VE SDEs.

    Returns:
      logits: The logits given by the noise-conditional classifier.
    """
    data = preprocess(data)
    logits = classifier.apply({'params': classifier_params}, data, ve_noise_scale, train=False, mutable=False)
    return logits

  return logit_fn


def get_classifier_grad_fn(logit_fn):
  """Create the gradient function for the classifier in use of class-conditional sampling. """

  def grad_fn(data, ve_noise_scale, labels):
    def prob_fn(data):
      logits = logit_fn(data, ve_noise_scale)
      prob = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(labels.shape[0]), labels].sum()
      return prob

    return jax.grad(prob_fn)(data)

  return grad_fn
