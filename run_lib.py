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
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any
import copy

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import functools

from flax.training import orbax_utils
import orbax.checkpoint
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, iddpm
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import datetime
import wandb
import matplotlib.pyplot as plt
import jax_smi

from flax.traverse_util import flatten_dict, unflatten_dict
from flax.training import orbax_utils, checkpoints

FLAGS = flags.FLAGS


def train(config, workdir, log_name):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  # ====================================================================================================== #
  # Get logger
  jax_smi.initialise_tracking()

  # wandb_dir: Directory of wandb summaries
  current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  if log_name is None:
    wandb.init(project="projectname", name=f"{config.model.name}-{current_time}", entity="username", resume="allow")
  else:
    wandb.init(project="projectname", name=log_name, entity="username", resume="allow")
  wandb_dir = os.path.join(workdir, "wandb")
  tf.io.gfile.makedirs(wandb_dir)
  wandb.config = config

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)
  rng = jax.random.PRNGKey(config.seed)
  # ====================================================================================================== #
  # Get modes
  """
    train_mode
      'train_baseline':  Train baseline model (1-Rectified flow) before Reflow.
      'gen_reflow_data': Reflow data generating phase before k-RF (k > 1)
      'train_reflow':    Training with reflowed data.
      'train_distill':   Distill the flow matching model with reflowed data.
  """
  if_reflow = False if 'rf_phase' not in config.model else (config.model.rf_phase > 1)
  if config.model.rf_phase == 1:
    train_mode = 'train_baseline'
  else:
    train_mode = config.training.reflow_mode
    assert train_mode in ['gen_reflow', 'train_reflow', 'train_distill', 'train_rf_distill']
    if train_mode in ['train_reflow', 'train_distill', 'train_rf_distill']:
      # TODO: Get the number of dataset we already have
      reflow_batch_idx = 0
      n_total_data = 0
      if train_mode == 'train_rf_distill':
        assert config.model.num_scales % config.training.reflow_t == 0
  # ====================================================================================================== #
  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  state = mutils.init_train_state(step_rng, config)

  # Generate placeholder
  state_dict = {
    'model': copy.deepcopy(state),
  }

  # Set checkpoint directories.
  if train_mode != 'train_baseline':
    if train_mode in ['gen_reflow', 'train_reflow']:
      old_checkpoint_dir = os.path.join(workdir, f"{config.model.rf_phase - 1}_rf", "checkpoints")
    elif train_mode in ['train_distill', 'train_rf_distill']:
      old_checkpoint_dir = os.path.join(workdir, f"{config.model.rf_phase}_rf", "checkpoints")
    else:
      raise NotImplementedError()
  else:
    assert config.model.rf_phase == 1
  
  if train_mode == 'train_distill':
    checkpoint_dir = os.path.join(workdir, f"{config.model.rf_phase}_distill", "checkpoints")
    checkpoint_meta_dir = os.path.join(workdir, f"{config.model.rf_phase}_distill", "checkpoints-meta")
  elif train_mode in ['gen_reflow', 'train_reflow', 'train_baseline']:
    checkpoint_dir = os.path.join(workdir, f"{config.model.rf_phase}_rf", "checkpoints")
    checkpoint_meta_dir = os.path.join(workdir, f"{config.model.rf_phase}_rf", "checkpoints-meta")
  elif train_mode == 'train_rf_distill':
    checkpoint_dir = os.path.join(workdir, f"{config.model.rf_phase}_{config.training.reflow_t}rf_{config.model.num_scales}distill", "checkpoints")
    checkpoint_meta_dir = os.path.join(workdir, f"{config.model.rf_phase}_{config.training.reflow_t}rf_{config.model.num_scales}distill", "checkpoints-meta")
  else:
    raise NotImplementedError()

  # get manager options, and restore checkpoints.
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  mgr_options = orbax.checkpoint.CheckpointManagerOptions(
    save_interval_steps=config.training.snapshot_freq,
    create=True)
  ckpt_mgr = orbax.checkpoint.CheckpointManager(
    checkpoint_dir,
    orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)

  if config.training.import_torch != 'none':
    # cast torch checkpoint if necessary.
    logging.info("Import from pretrained pytorch model.")
    state = mutils.torch_to_flax_ckpt(config.training.import_torch, state, config.training.reflow_t,
                                      config.model.initial_count, config.model.embedding_type)

  elif config.training.import_flax_baseline != 'none':
    logging.info("Import from pretrained flax model.")
    config_reflow_one = copy.deepcopy(config)
    config_reflow_one.training.reflow_t = 1
    rng, step_rng = jax.random.split(rng)
    state_baseline = mutils.init_train_state(step_rng, config_reflow_one)
    state_baseline_dict = {
      'model': copy.deepcopy(state_baseline),
    }
    state_baseline_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(config.training.import_flax_baseline, "default"), target=state_baseline_dict)
    state = mutils.flax_to_flax_ckpt(state_baseline_dict['model'], state, config.training.reflow_t)
    state = state.replace(step=0)
    del state_baseline, state_baseline_dict

  elif train_mode == 'train_baseline':
    logging.info(f"Restore checkpoint from {config.training.reflow_source_ckpt}.")
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(checkpoint_dir, "0", "default"), target=state_dict)
    state = state_dict['model']
    state = state.replace(step=0)

  elif train_mode in ['gen_reflow', 'train_reflow', 'train_distill', 'train_rf_distill']:
    # Resume training when intermediate checkpoints are detected
    logging.info(f"Restore checkpoint from step {config.training.reflow_source_ckpt}.")
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(old_checkpoint_dir, f"{config.training.reflow_source_ckpt}", "default"), target=state_dict)
    state = state_dict['model']
    state = state.replace(step=0)

  else:
    raise NotImplementedError()

  # Resume training when intermediate checkpoints are detected
  mgr_meta_options = orbax.checkpoint.CheckpointManagerOptions(
    save_interval_steps=config.training.snapshot_freq_for_preemption,
    max_to_keep=1,
    create=True)
  ckpt_meta_mgr = orbax.checkpoint.CheckpointManager(
    checkpoint_meta_dir,
    orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_meta_options)
  if ckpt_meta_mgr.latest_step() is not None:
    logging.info(f"Restore checkpoint-meta from step {ckpt_meta_mgr.latest_step()}.")
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(checkpoint_dir, f"{ckpt_meta_mgr.latest_step()}", "default"), target=state_dict)
    state = state_dict['model']

  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  # ====================================================================================================== #
  # Build data iterators
  if train_mode in ['train_baseline', 'gen_reflow']:
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=config.training.n_jitted_steps,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                gen_reflow=(train_mode == 'gen_reflow'))
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  # ====================================================================================================== #
  # Setup SDEs
  if config.training.sde.lower() == 'rfsde':
    sde = sde_lib.RFSDE(N=config.model.num_scales)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  # ====================================================================================================== #
  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  train_step_fn = losses.get_step_fn(config, sde, state, train=True, optimize_fn=optimize_fn, eps=sampling_eps, reflow_t=config.training.reflow_t)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)

  if train_mode != 'gen_reflow':
    eval_step_fn = losses.get_step_fn(config, sde, state, train=False, optimize_fn=optimize_fn, eps=sampling_eps, reflow_t=config.training.reflow_t)
    # Pmap (and jit-compile) multiple evaluation steps together for faster running
    p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

  # Building sampling functions
  if config.training.snapshot_sampling and (train_mode != 'gen_reflow'):
    sampling_shape = (config.eval.batch_size // jax.local_device_count(), config.data.image_size,
                      config.data.image_size, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  
  elif train_mode == 'gen_reflow':
    sampling_shape = (config.training.gen_reflow_size // jax.local_device_count(), config.data.image_size,
                      config.data.image_size, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, gen_reflow=True)

  else:
    logging.info("Sampling function is not required.")

  pstate = flax_utils.replicate(state)
  num_train_steps = config.training.n_iters
  # ====================================================================================================== #
  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.process_index() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.process_index())

  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"
  # ====================================================================================================== #
  # Trigger that update reflow batch
  n_reflow_batch = config.training.n_reflow_data // (config.training.batch_size * config.training.n_jitted_steps) + 1
  reflow_sample_dir = os.path.join(sample_dir, "reflow_batch")

  def gen_reflow_pair(rng, batch, batch_idx):
    """
      Generate reflow pair, with the ODE solver. (performs better than the PC solver.)

      Input
        rng: jax.random.PRNGKey variable.
        batch: input batch, scaled by `scaler`. = centered to 0. shape (n_tpu, B, H, W, C)

      config.training.reflow_t: how many division to divide reflow_t
      For example, divide t into
        1 : [0, 1]
        2 : [0, 0.5], [0.5, 1]
        3 : [0, 1/3], [1/3, 2/3], [2/3, 1]
        4 : [0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1]
      and generate data from latter time to former time, with equal probability

      kwargs
        batch_idx: batch index to generate in current step.
      
      Return:
        (x0, x1), (t0, t1)
        x0: destination x at time t0, scaled to zero-centered. shape (n_tpu, B, H, W, C)
        x1: source x at time t1, scaled to zero-centered. shape (n_tpu, B, H, W, C)
        t0: destination time. shape (n_tpu, B)
        t1: source time. shape (n_tpu, B)
    """
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)
    (x0, x1), (t0, t1), _ = sampling_fn(next_rng, pstate, cond_image=batch, batch_idx=batch_idx)
    return (scaler(jnp.reshape(x0, (x0.shape[0] * x0.shape[1],) + x0.shape[2:])), \
            scaler(jnp.reshape(x1, (x1.shape[0] * x1.shape[1],) + x1.shape[2:]))), \
           (jnp.reshape(t0, (t0.shape[0] * t0.shape[1],)), \
            jnp.reshape(t1, (t1.shape[0] * t1.shape[1],)))

  def set_reflow_batch_fn(rng, train_iter):
    """
      Input
        rng: jax.random.PRNGKey variable.
        train_iter: iterable for the training set.
      Return
        None

      The generated reflow pairs will be saved in "reflow_sample_dir/reflow_batch_*.npz".
        x0: destination x at time t0, inverse_scaled to [0, 1]
        x1: source x at time t1, inverse_scaled to [0, 1]
        t0: destination time
        t1: source time

      Workflow.
        Each step
          - Generate reflow dataset of size `config.training.batch_size * config.training.reflow_batch_size`
          - Generated batch_idx: [0, reflow_t - 1].
          - For each reflow_t generated step, shuffle whole current batch.
          - If stacked data >= n_reflow_batch, then save top [config.training.batch_size * config.training.n_jitted_steps]

    """
    tf.io.gfile.makedirs(reflow_sample_dir)
    reflow_list = tf.io.gfile.glob(os.path.join(reflow_sample_dir, "reflow_batch_*.npz"))
    n_reflow_batch_begin = len(reflow_list) + 1
    # _ = [tf.io.gfile.remove(f) for f in tf.io.gfile.glob(os.path.join(reflow_sample_dir, "reflow_batch_*.npz"))] # Reset
    logging.info(f"Start generating reflow pair of {n_reflow_batch * config.training.batch_size * config.training.n_jitted_steps} data points.")
    total_generated_data = int((n_reflow_batch_begin - 1) * config.training.batch_size * config.training.n_jitted_steps)
    logging.info(f"Totally {total_generated_data} data generated so far.")

    jit_idx = 0
    initial_phase = True
    current_reflow_batch = n_reflow_batch_begin
    while total_generated_data <= config.training.n_reflow_data:
      # Generate a single batch of interval `batch_idx`.
      x0, x1, t0, t1 = [], [], [], []
      for b_idx in range(config.training.reflow_t):
        if jit_idx == 0:
          batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter)) # new batch
        rng, step_rng = jax.random.split(rng)
        (x0_batch, x1_batch), (t0_batch, t1_batch) = gen_reflow_pair(rng=step_rng, batch=batch['image'][:, jit_idx], batch_idx=b_idx)
        x0.append(x0_batch)
        x1.append(x1_batch)
        t0.append(t0_batch)
        t1.append(t1_batch)
        jit_idx = (jit_idx + 1) % config.training.n_jitted_steps

      x0 = jnp.concatenate(x0, axis=0) # (config.training.reflow_t * B, H, W, C)
      x1 = jnp.concatenate(x1, axis=0)
      t0 = jnp.concatenate(t0, axis=0) # (config.training.reflow_t * B,)
      t1 = jnp.concatenate(t1, axis=0) 

      rng, step_rng = jax.random.split(rng)
      perm_idx = jax.random.permutation(step_rng, t0.shape[0])
      x0, x1, t0, t1 = x0[perm_idx], x1[perm_idx], t0[perm_idx], t1[perm_idx]

      if initial_phase:
        x0_all, x1_all, t0_all, t1_all = x0, x1, t0, t1
        initial_phase = False
      else:
        x0_all = jnp.concatenate([x0_all, x0], axis=0)
        x1_all = jnp.concatenate([x1_all, x1], axis=0)
        t0_all = jnp.concatenate([t0_all, t0], axis=0)
        t1_all = jnp.concatenate([t1_all, t1], axis=0)

      logging.info(f"data and noise stacked: {x0_all.shape}")

      # save if there is enough stack.
      while t0_all.shape[0] >= config.training.batch_size * config.training.n_jitted_steps:
        x0_top, x0_all = jnp.split(x0_all, [config.training.batch_size * config.training.n_jitted_steps], axis=0)
        x1_top, x1_all = jnp.split(x1_all, [config.training.batch_size * config.training.n_jitted_steps], axis=0)
        t0_top, t0_all = jnp.split(t0_all, [config.training.batch_size * config.training.n_jitted_steps], axis=0)
        t1_top, t1_all = jnp.split(t1_all, [config.training.batch_size * config.training.n_jitted_steps], axis=0)
        np.savez_compressed(os.path.join(reflow_sample_dir, f"reflow_batch_{current_reflow_batch}.npz"),
                            x0=x0_top,
                            x1=x1_top,
                            t0=t0_top,
                            t1=t1_top)
        total_generated_data += config.training.batch_size * config.training.n_jitted_steps
        logging.info(f"Generated reflow pair {current_reflow_batch}. Generated data points: {total_generated_data}.")

        if current_reflow_batch == n_reflow_batch_begin:
          # Draw sample pair figure
          x0_batch_draw = inverse_scaler(jnp.reshape(x0_top, (-1, *x0_top.shape[-3:]))[0:64])
          x1_batch_draw = inverse_scaler(jnp.reshape(x1_top, (-1, *x1_top.shape[-3:]))[0:64])
          utils.draw_figure_grid(x0_batch_draw, reflow_sample_dir, f"reflow_destination_example")
          utils.draw_figure_grid(x1_batch_draw, reflow_sample_dir, f"reflow_source_example")
          original_batch = jnp.reshape(jnp.swapaxes(batch['image'], 0, 1), (-1, *batch['image'].shape[-3:]))
          utils.draw_figure_grid(inverse_scaler(original_batch)[0:64], reflow_sample_dir, f"reflow_original_example")

        del x0_top, x1_top, t0_top, t1_top

        current_reflow_batch += 1

  # ====================================================================================================== #
  # Main training or generation part
  if train_mode == "gen_reflow":
    assert config.sampling.method == 'ode'
    set_reflow_batch_fn(rng, train_iter)
  else:
    for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
      if train_mode in ["train_reflow", "train_distill", "train_rf_distill"]:
        # Use pre-trained reflow dataset
        if step == initial_step:
          logging.info(f"Already have reflow data with {config.training.reflow_t} divisions.")
          reflow_data_files = tf.io.gfile.glob(os.path.join(reflow_sample_dir, "reflow_batch_*.npz"))
          assert len(reflow_data_files) >= n_reflow_batch, \
            f"Have {len(reflow_data_files)} files; Should have {n_reflow_batch} reflow batches."
          for reflow_data_file in reflow_data_files:
            rf_data_temp = np.load(reflow_data_file)
            n_total_data += rf_data_temp['x0'].shape[0]
          logging.info(f"Have {n_total_data} simulation-driven reflow data.")
        reflow_batch = np.load(os.path.join(reflow_sample_dir, f"reflow_batch_{reflow_batch_idx+1}.npz"))
        batch = (
          (reflow_batch['x0'], reflow_batch['x1']),
          (reflow_batch['t0'], reflow_batch['t1'])
        )
        assert batch[0][0].shape[0] == config.training.n_jitted_steps * config.training.batch_size, \
          f"{batch[0][0].shape[0]} vs. {config.training.n_jitted_steps * config.training.batch_size}"
        batch = (
          jax.tree_map(
          lambda x: jnp.reshape(x, (jax.local_device_count(), config.training.n_jitted_steps, config.training.batch_size // jax.local_device_count(), *x.shape[-3:])),
          batch[0]
          ),
          jax.tree_map(
          lambda x: jnp.reshape(x, (jax.local_device_count(), config.training.n_jitted_steps, config.training.batch_size // jax.local_device_count())),
          batch[1]
          )
        )
      elif train_mode == "train_baseline":
        # Use batch
        batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))
      else:
        raise ValueError("train_mode should be in [`train_baseline`, `train_reflow`, `train_distill`, `train_rf_distill`.]")
      # ====================================================================================================== #
      if (step % config.training.snapshot_freq == 0 or step == num_train_steps):
        if (not config.training.zero_snapshot) and step == 0:
          pass
        else:
          # Generate and save one batch of samples
          if config.training.snapshot_sampling:
            rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
            sample_rng = jnp.asarray(sample_rng)
            (sample, init_noise), _, _ = sampling_fn(sample_rng, pstate)
            image_grid = jnp.reshape(sample, (-1, *sample.shape[-3:]))

            # Draw snapshot figure
            if train_mode == 'train_reflow':
              this_sample_dir = os.path.join(
                sample_dir, "rf_{}_iter_{}_host_{}".format(config.model.rf_phase, step, jax.process_index()))
            elif train_mode == 'train_baseline':
              this_sample_dir = os.path.join(
                sample_dir, "rf_1_iter_host_{}".format(step, jax.process_index()))
            elif train_mode == 'train_distill':
              this_sample_dir = os.path.join(
                sample_dir, "distill_{}_iter_{}_host_{}".format(config.model.rf_phase, step, jax.process_index()))
            elif train_mode == 'train_rf_distill':
              this_sample_dir = os.path.join(
                sample_dir, "rf_distill_{}_iter_{}_host_{}".format(config.model.rf_phase, step, jax.process_index()))
            else:
              raise ValueError()
            tf.io.gfile.makedirs(this_sample_dir)
            utils.draw_figure_grid(image_grid[0:64], this_sample_dir, f"sample_{step}")
          
            # Get statistics
            stats = utils.get_samples_and_statistics(config, rng, sampling_fn, pstate, this_sample_dir, sampling_shape, mode='train', current_step=step)
            logging.info(f"FID = {stats['fid']}")
            logging.info(f"KID = {stats['kid']}")
            logging.info(f"Inception_score = {stats['is']}")
            logging.info(f"NFE (Number of function evaluations) = {stats['nfe']}")
            wandb_statistics_dict = {
              'fid': float(stats['fid']),
              'kid': float(stats['kid']),
              'inception_score': float(stats['is']),
              'nfe': float(stats['nfe']),
              'step': int(step),
              'n_data': int(config.training.snapshot_fid_sample),
            }
            if train_mode == 'train_reflow':
              logging.info(f"straightness = {stats['straightness']['straightness']}")
              logging.info(f"sequential straightness = {stats['straightness']['seq_straightness']}")

              wandb_statistics_dict_new = {
                'straightness': float(stats['straightness']['straightness']),
                'seq_straightness': float(stats['straightness']['seq_straightness'])
              }
              wandb_statistics_dict = {**wandb_statistics_dict, **wandb_statistics_dict_new}
            wandb.log(wandb_statistics_dict, step=step)
      # ====================================================================================================== #
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, pstate), ploss = p_train_step((next_rng, pstate), batch)

      if if_reflow:
        reflow_batch_idx = (reflow_batch_idx + 1) % n_reflow_batch

      # Calculate loss and save
      loss = flax.jax_utils.unreplicate(ploss).mean()
      wandb_log_dict = {'train/loss': float(loss)}
      # Log to console, file and tensorboard on host 0
      if jax.process_index() == 0 and step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e" % (step, loss))
        wandb.log(wandb_log_dict, step=step)

      if step % config.training.snapshot_freq_for_preemption == 0:
        # Save a temporary checkpoint to resume training after pre-emption periodically
        saved_state = flax_utils.unreplicate(pstate)
        state_dict = {
          'model': copy.deepcopy(saved_state),
        }
        save_args = flax.training.orbax_utils.save_args_from_target(state_dict)
        ckpt_meta_mgr.save(step, state_dict, save_kwargs={'save_args': save_args})
        del state_dict

      if step % config.training.snapshot_freq == 0:
        # Save a temporary checkpoint to resume training after pre-emption periodically
        saved_state = flax_utils.unreplicate(pstate)
        state_dict = {
          'model': copy.deepcopy(saved_state),
        }
        save_args = flax.training.orbax_utils.save_args_from_target(state_dict)
        ckpt_mgr.save(step, state_dict, save_kwargs={'save_args': save_args})
        del state_dict
      
      # ====================================================================================================== #
      # Report the loss on an evaluation dataset periodically only in train_baseline case
      if (step % config.training.eval_freq == 0) and (train_mode == "train_baseline"):
        eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)
        
        # Eval loss at the baseline.
        (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)

        eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
        wandb_log_dict = {'eval/loss': float(eval_loss)}
        if jax.process_index() == 0:
          logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
          wandb.log(wandb_log_dict, step=step)
      # ====================================================================================================== #

def evaluate(config, workdir, log_name, eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # ====================================================================================================== #
  # Get logger
  jax_smi.initialise_tracking()

  # wandb_dir: Directory of wandb summaries
  current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  if log_name is None:
    wandb.init(project="projectname", name=f"{config.model.name}-{current_time}", entity="username", resume="allow")
  else:
    wandb.init(project="projectname", name=log_name, entity="username", resume="allow")
  wandb_dir = os.path.join(workdir, "wandb")
  tf.io.gfile.makedirs(wandb_dir)
  wandb.config = config

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)
  rng = jax.random.PRNGKey(config.seed)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  # ====================================================================================================== #
  # Get modes
  """
    eval_mode
      'eval_baseline': Evaluate baseline model. (Original RF)
      'eval_reflow': Evaluate reflow (n-SeqRF) model.
      'eval_distill': Evaluate distill (n-SeqRF-distill) model.
      'eval_rf_distill': distill from baseline reflow model.
  """
  if config.model.rf_phase == 1:
    eval_mode = 'eval_baseline'
  else:
    eval_mode = config.eval.reflow_mode
    assert eval_mode in ['eval_reflow', 'eval_distill', 'eval_rf_distill']
    if eval_mode == 'eval_rf_distill':
      assert config.model.num_scales % config.training.reflow_t == 0
  # ====================================================================================================== #
  # Initialize model
  rng, step_rng = jax.random.split(rng)
  state = mutils.init_train_state(step_rng, config)

  # Generate placeholder.
  state_dict = {
    'model': copy.deepcopy(state),
  }

  if eval_mode == 'eval_reflow':
    checkpoint_dir = os.path.join(workdir, f"{config.model.rf_phase}_rf", "checkpoints")
  elif eval_mode == 'eval_distill':
    checkpoint_dir = os.path.join(workdir, f"{config.model.rf_phase}_distill", "checkpoints")
  elif eval_mode == 'eval_baseline':
    # checkpoint_dir will not be used
    checkpoint_dir = None
  elif eval_mode == 'eval_rf_distill':
    checkpoint_dir = os.path.join(workdir, f"{config.model.rf_phase}_{config.training.reflow_t}rf_{config.model.num_scales}distill", "checkpoints")
  else:
    raise NotImplementedError()
  # ====================================================================================================== #
  # Setup SDEs
  if config.training.sde.lower() == 'rfsde':
    sde = sde_lib.RFSDE(N=config.model.num_scales)
    sampling_eps = 1e-3 # Not used.
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  # ====================================================================================================== #
  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                      config.data.image_size, config.data.image_size,
                      config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  # ====================================================================================================== #
  # Add additional task for evaluation (for example, get gradient statistics) here.
  # ====================================================================================================== #
  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.process_index())

  # TODO: Custom experiment. (Playground)
  if config.eval.custom:
    rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng) # use the same seed
    sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                      config.data.image_size, config.data.image_size,
                      config.data.num_channels)
    if eval_mode == 'eval_baseline':
      logging.info("Import from pretrained pytorch model.")
      assert config.training.import_torch != "none"
      state = mutils.torch_to_flax_ckpt(config.training.import_torch, state, config.training.reflow_t,
                                        config.model.initial_count, config.model.embedding_type)
    else:
      state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(checkpoint_dir, f"100000", "default"), target=state_dict)
      state = state_dict['model']
    pstate = flax.jax_utils.replicate(state)
    logging.info("Evaluate global truncation error, compared to 480-step Euler solver.")
    config.eval.save_trajectory = True

    nfe_set = [480, 240, 120, 48, 24, 12, 8, 6, 4, 2]
    for nfe in nfe_set:
      max_nfe = nfe_set[0]
      # for nfe in [12]:
      if nfe % config.training.reflow_t != 0:
        continue

      logging.info(f"Run for NFE={nfe}.")
      sde = sde_lib.RFSDE(N=nfe)
      config.model.num_scales = nfe
      sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
      (samples, z), _, straightness = sampling_fn(sample_rng, pstate) # fix sample_rng
      samples = samples.reshape((-1, *samples.shape[-3:]))
      utils.draw_figure_grid(samples[0:64], os.path.join("test"), f"{nfe}")
      np.save(f"test/{nfe}.npy", np.array(straightness))
      logging.info(f"{nfe} done.")
      # exit()
      # sample_dict[nfe] = samples
      # trajectory_dict[nfe] = straightness['all_trajectory']

      # sample_dict[nfe] = np.array(jnp.reshape(sample_dict[nfe], (128, 32, 32, 3)))
      # trajectory_dict[nfe] = np.array(jnp.reshape(jnp.transpose(trajectory_dict[nfe], (1, 0, 2, 3, 4, 5)), (nfe, 128, 32, 32, 3)))
      # del samples, straightness

      # if nfe < max_nfe:
      #   total_diff = sample_dict[nfe] - sample_dict[max_nfe]
      #   assert trajectory_dict[nfe].shape[0] == nfe
      #   assert trajectory_dict[max_nfe].shape[0] == max_nfe

      #   current_diff = []
      #   total_gap = np.zeros((nfe,))
      #   for i in range(nfe):
      #     current_gap = trajectory_dict[nfe][i + 1 - 1] - trajectory_dict[max_nfe][int(max_nfe // nfe) * (i + 1) - 1] # (128, 32, 32, 3)
      #     current_gap = np.mean(np.sqrt(np.mean(current_gap ** 2, (1, 2, 3))))
      #     total_gap[i] = current_gap
      #   print(total_gap)
      #   np.save(f"test/{nfe}.npy", total_gap)
      
      #   current_diff = jnp.concatenate([jnp.expand_dims(c, 0) for c in current_diff], axis=0)
      #   print(current_diff.shape)
      #   exit()
      
  elif config.eval.custom_two:
    """
    Two experiments
    (1) Verify variance reduction effect.
    (2) Lipschitz constant effect.

    How to do?
    (1) for each t in [0, 1], sample (x_t, t) randomly from the training set.
    (2) Run the neural network by value_and_grad.
    (3) Calculate the square norm of the gradient.
    (4) Calculate the average flow matching loss.
     --> Multiply (3) and (4) to obtain the upper bound of variance.
    (5) Save the flow function f(x_t, t).
     
    (5) slightly (and randomly) perturb x_t and get (x_t', t)
    (6) Obtain the flow function f(x_t', t) of the perturbed data.
     --> Get average, 99% highest and maximum of M(t) w.r.t. time t, and save this.
    """
    if eval_mode == 'eval_baseline':
      # cast torch checkpoint if necessary.
      logging.info("Import from pretrained pytorch model.")
      assert config.training.import_torch != "none"
      state = mutils.torch_to_flax_ckpt(config.training.import_torch, state, config.training.reflow_t,
                                        config.model.initial_count, config.model.embedding_type)
      pstate = flax.jax_utils.replicate(state)
    elif eval_mode == 'eval_reflow': # we always use 100000 step in this playground.
      logging.info(f"Restore checkpoint from step 100000.")
      state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(checkpoint_dir, f"100000", "default"), target=state_dict)
      state = state_dict['model']
      pstate = flax.jax_utils.replicate(state)
    else:
      raise ValueError("eval_mode should be eval_baseline or eval_reflow.")


    _, eval_ds, _ = datasets.get_dataset(config,
                                         additional_dim=config.training.n_jitted_steps,
                                         uniform_dequantization=False,
                                         gen_reflow=False)
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

    # Now we imported state.
    t_ls = np.linspace(0.05, 0.95, 19)
    n_batch = 1
    loss_jcfm = np.zeros_like(t_ls)
    grad_sq_norm_arr = np.zeros_like(t_ls)
    avg_lipschitzness = np.zeros_like(t_ls)
    top_99_lipschitzness = np.zeros_like(t_ls)
    max_lipschitzness = np.zeros_like(t_ls)


    
    # Eval loss at the baseline.

    for i in range(len(t_ls)):
      t=t_ls[i]
      eval_step_fn = losses.get_step_fn_playground(sde, state, train=False, reflow_t=config.training.reflow_t, t=t)
      p_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)
      grad_sq_norm_sum = 0
      lipschitz_list = []
      for idx in range(n_batch):
        eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)
        
        batch = eval_batch['image']
        # (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)
        (_, _), (lipschitz, grad_sq_norm) = p_step((next_rng, pstate), batch)
        # print(lipschitz.shape) # (8, 5, 16)
        # print(grad_sq_norm.shape) # (8, 5)
        grad_sq_norm_sum += jnp.sum(grad_sq_norm)
        lipschitz_list.append(jnp.expand_dims(lipschitz, 0))

        logging.info(f"{idx + 1} step done.")
      lipschitz_list = np.reshape(jnp.concatenate(lipschitz_list, 0), (-1,))
      grad_sq_norm_sum /= n_batch
      grad_sq_norm_arr[i] = np.array(grad_sq_norm_sum)
      avg_lipschitzness[i] = np.sum(lipschitz_list) / (40 * n_batch)
      max_lipschitzness[i] = np.max(lipschitz_list)
      print(t, grad_sq_norm_arr[i], avg_lipschitzness[i], max_lipschitzness[i], flush=True)
    exit()

  elif eval_mode == 'eval_baseline':
    # cast torch checkpoint if necessary.
    logging.info("Import from pretrained pytorch model.")
    assert config.training.import_torch != "none"
    state = mutils.torch_to_flax_ckpt(config.training.import_torch, state, config.training.reflow_t,
                                      config.model.initial_count, config.model.embedding_type)
    pstate = flax.jax_utils.replicate(state)
    # ====================================================================================================== #
    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      state = jax.device_put(state)
      # Run sample generation for multiple rounds to create enough samples
      # Designed to be pre-emption safe. Automatically resumes when interrupted
      if jax.process_index() == 0:
        logging.info("Sampling -- baseline")
      this_sample_dir = os.path.join(eval_dir, f"baseline_host_{jax.process_index()}")
      stats = utils.get_samples_and_statistics(config, rng, sampling_fn, pstate, this_sample_dir, sampling_shape, mode='eval', current_step=0)
      logging.info(f"FID = {stats['fid']}")
      logging.info(f"KID = {stats['kid']}")
      logging.info(f"Inception_score = {stats['is']}")
      wandb_statistics_dict = {
        'fid': float(stats['fid']),
        'kid': float(stats['kid']),
        'inception_score': float(stats['is']),
        'sample': wandb.Image(os.path.join(this_sample_dir, "sample.png")),
        'nfe': stats['nfe']
      }

      logging.info(f"straightness = {stats['straightness']['straightness']}")
      logging.info(f"sequential straightness = {stats['straightness']['seq_straightness']}")

      wandb.log(wandb_statistics_dict, step=0)
    # ====================================================================================================== #
  elif eval_mode in ['eval_distill', 'eval_reflow', 'eval_rf_distill']:
    for ckpt in range(config.eval.begin_step, config.eval.end_step + 1, config.eval.interval_step):
      logging.info(f"Restore checkpoint from step {ckpt}.")
      state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(checkpoint_dir, f"{ckpt}", "default"), target=state_dict)
      state = state_dict['model']
      pstate = flax.jax_utils.replicate(state)
      # ====================================================================================================== #
      # Generate samples and compute IS/FID/KID when enabled
      if config.eval.enable_sampling:
        state = jax.device_put(state)
        # Run sample generation for multiple rounds to create enough samples
        # Designed to be pre-emption safe. Automatically resumes when interrupted
        if jax.process_index() == 0:
          logging.info("Sampling -- checkpoint step: %d" % (ckpt,))
        this_sample_dir = os.path.join(
          eval_dir, f"step_{ckpt}_host_{jax.process_index()}")
        
        stats = utils.get_samples_and_statistics(config, rng, sampling_fn, pstate, this_sample_dir, sampling_shape,
                                                 mode='eval', current_step=ckpt)
        straightness_dir = os.path.join(this_sample_dir, "straightness")
        logging.info(f"FID = {stats['fid']}")
        logging.info(f"KID = {stats['kid']}")
        logging.info(f"Inception_score = {stats['is']}")
        logging.info(f"NFE (Number of function evaluations) = {stats['nfe']}")
        logging.info(f"straightness = {stats['straightness']['straightness']}")
        logging.info(f"sequential straightness = {stats['straightness']['seq_straightness']}")
        wandb_statistics_dict = {
          'fid': float(stats['fid']),
          'kid': float(stats['kid']),
          'inception_score': float(stats['is']),
          'sample': wandb.Image(os.path.join(this_sample_dir, "sample.png")),
          'nfe': float(stats['nfe']),
          'straightness': float(stats['straightness']['straightness']),
          'seq_straightness': float(stats['straightness']['seq_straightness']),
          'str_fig': wandb.Image(os.path.join(straightness_dir, f'straightness_{ckpt}.png')),
          'seq_str_fig': wandb.Image(os.path.join(straightness_dir, f'seq_straightness_{ckpt}.png')),
          'step': int(ckpt),
          'n_data': int(config.eval.num_samples),
        }
        wandb.log(wandb_statistics_dict, step=ckpt)
      # ====================================================================================================== #
  else:
    raise NotImplementedError("TODO: eval_ada_distill, eval_ada_reflow, rf_distill")
  # ====================================================================================================== #
