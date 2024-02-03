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
"""Utility code for generating and saving image grids and checkpointing.

   The `save_image` code is copied from
   https://github.com/google/flax/blob/master/examples/vae/utils.py,
   which is a JAX equivalent to the same function in TorchVision
   (https://github.com/pytorch/vision/blob/master/torchvision/utils.py)
"""

import math
from typing import Any, Dict, Optional, TypeVar

import flax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import torch
import time
import matplotlib.pyplot as plt

import evaluation
import gc
import io
import logging
import tensorflow_gan as tfgan

from jax.experimental.host_callback import call
import datasets

T = TypeVar("T")


def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)


def load_training_state(filepath, state):
  with tf.io.gfile.GFile(filepath, "rb") as f:
    state = flax.serialization.from_bytes(state, f.read())
  return state


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
  """Make a grid of images and save it into an image file.

  Pixel values are assumed to be within [0, 1].

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C).
    fp: A filename(string) or file object.
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename, this
      parameter should always be used.
  """
  if not (isinstance(ndarray, jnp.ndarray) or isinstance(ndarray, np.ndarray) or
          (isinstance(ndarray, list) and
           all(isinstance(t, jnp.ndarray) for t in ndarray))):
    raise TypeError("array_like of tensors expected, got {}".format(
      type(ndarray)))

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                       padding)
  num_channels = ndarray.shape[3]
  grid = jnp.full(
    (height * ymaps + padding, width * xmaps + padding, num_channels),
    pad_value).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = grid.at[y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width].set(ndarray[k])
      k = k + 1

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
  im = Image.fromarray(np.array(ndarr.copy()))
  im.save(fp, format=format)


def flatten_dict(config):
  """Flatten a hierarchical dict to a simple dict."""
  new_dict = {}
  for key, value in config.items():
    if isinstance(value, dict):
      sub_dict = flatten_dict(value)
      for subkey, subvalue in sub_dict.items():
        new_dict[key + "/" + subkey] = subvalue
    elif isinstance(value, tuple):
      new_dict[key] = str(value)
    else:
      new_dict[key] = value
  return new_dict


def to_flattened_numpy(x):
  """Flatten a JAX array `x` and convert it to numpy."""
  return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
  """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
  return jnp.asarray(x).reshape(shape)


def draw_figure_grid(sample, sample_dir, figname):
  """Draw grid of figures; samples are of [0, 1]-valued numpy arrays."""
  tf.io.gfile.makedirs(sample_dir)
  image_grid = sample.reshape((-1, *sample.shape[-3:]))
  nrow = int(min(np.sqrt(image_grid.shape[0]), 8))
  sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
  max_sample = int(min(nrow ** 2, image_grid.shape[0]))
  sample = sample[0:max_sample]
  with tf.io.gfile.GFile(
      os.path.join(sample_dir, f"{figname}.np"), "wb") as fout:
    np.save(fout, sample)

  with tf.io.gfile.GFile(
      os.path.join(sample_dir, f"{figname}.png"), "wb") as fout:
    save_image(image_grid, fout, nrow=nrow, padding=2)


def get_samples_and_statistics(config, rng, sampling_fn, pstate, sample_dir, sample_shape, mode='train',
                               save_samples=False, save_trajectory=False, current_step=-1):
  """
    Sampling pipeline, including statistics
  """
  if mode == 'train':
    # train mode: snapshot sampling
    n_samples = config.training.snapshot_fid_sample
    b_size = config.eval.batch_size
  elif mode == 'eval':
    # eval mode: sampling for evaluation
    n_samples = config.eval.num_samples
    b_size = config.eval.batch_size
  else:
    raise NotImplementedError()
  num_sampling_rounds = (n_samples - 1) // b_size + 1
  tf.io.gfile.makedirs(sample_dir)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  # Sample
  all_pools = []
  all_straightness_dict = {
    'straightness': [],
    'straightness_by_t': [],
    'seq_straightness': [],
    'seq_straightness_by_t' : [],
    'nfe': [],
    'interval': []
  }

  for i in range(num_sampling_rounds):
    logging.info(f"Round {i + 1} for sampling.")
    rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng)

    (samples, z), _, straightness = sampling_fn(sample_rng, pstate)
    with tf.io.gfile.GFile(
        os.path.join(sample_dir, f"straightness_{i+1}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer,
                          straightness=straightness['straightness'],
                          straightness_by_t=straightness['straightness_by_t'],
                          seq_straightness=straightness['seq_straightness'],
                          seq_straightness_by_t=straightness['seq_straightness_by_t'],
                          nfe=straightness['nfe'])
      fout.write(io_buffer.getvalue())

    gc.collect()
    samples = samples.reshape((-1, *samples.shape[-3:]))

    # Visualize example images in first step
    if i == 0:
      image_grid = samples[0:64]
      draw_figure_grid(image_grid, sample_dir, "sample")

    # Visualize sequential straightness over time.
    # if (i == 0) and (config.sampling.method == 'ode'):
    if config.sampling.method == 'pc':
      straightness['interval'] = straightness['interval'][0]
      straightness['straightness_by_t'] = jnp.mean(straightness['straightness_by_t'], 0)
      straightness['seq_straightness_by_t'] = jnp.mean(straightness['seq_straightness_by_t'], 0)

    if i == 0:
      if current_step != -1:
        straightness_dir = os.path.join(sample_dir, "straightness")
        tf.io.gfile.makedirs(straightness_dir)

        # Plot for straightness
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim = [0, 1], title=f'straightness (step {current_step})', xlabel='t', ylabel='straightness')
        ax.plot(straightness['interval'][:-1], straightness['straightness_by_t'])
        plt.savefig(os.path.join(straightness_dir, f'straightness_{current_step}.png'))
        plt.close(fig)

        # Plot for sequential straightness
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim = [0, 1], title=f'sequential straightness (step {current_step})', xlabel='t', ylabel='seq. straightness')
        ax.plot(straightness['interval'][:-1], straightness['seq_straightness_by_t'])
        plt.savefig(os.path.join(straightness_dir, f'seq_straightness_{current_step}.png'))
        plt.close(fig)

        # Save the value in the first iteration.
        all_straightness_dict['straightness_by_t'] = straightness['straightness_by_t']
        all_straightness_dict['seq_straightness_by_t'] = straightness['seq_straightness_by_t']
        all_straightness_dict['interval'] = straightness['interval']

      # Obtain optimal trajectory
      if config.eval.save_trajectory:
        assert 'all_trajectory' in straightness, "There should be saved trajectory to evaluate optimal one."
        trajectory = straightness['all_trajectory']
        logging.info(f"Evaluating the optimal trajectory for {config.training.reflow_t} segments, "
                      f"using dynamic programming, using {trajectory.shape[0]} data.")

        del all_straightness_dict
        del straightness

        trajectory = jnp.transpose(trajectory, (1, 0, 2, 3, 4, 5))
        trajectory = jnp.reshape(trajectory, trajectory.shape[0:1] + (trajectory.shape[1] * trajectory.shape[2],) + trajectory.shape[3:])
        logging.info(trajectory.shape)

        # trajectory shape: (n_steps + 1, B, H, W, C)
        path, distance = optimal_timestep(trajectory, 13)
        exit()

    # Save images to `samples_*.npz`
    samples_save = np.clip(samples * 255., 0, 255).astype(np.uint8) # [0, 1] --> [0, 255]
    with tf.io.gfile.GFile(
        os.path.join(sample_dir, f"samples_{i+1}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, samples=samples_save)
      fout.write(io_buffer.getvalue())

    # Save stats to `statistics_*.npz
    # Force garbage collection before calling TensorFlow code for Inception network
    gc.collect()
    latents = evaluation.run_inception_distributed(samples_save, inception_model,
                                                    inceptionv3=inceptionv3)
    # Force garbage collection again before returning to JAX code
    gc.collect()
    # Save latent represents of the Inception network to disk or Google Cloud Storage
    with tf.io.gfile.GFile(
        os.path.join(sample_dir, f"statistics_{i+1}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(
        io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
      fout.write(io_buffer.getvalue())

  # Check if there is pretrained inception pool layer statistics
  data_stats, have_stats = evaluation.load_dataset_stats(config)
  if have_stats:
    data_pools = data_stats["pool_3"]

  else:
    # Build training dataset iterators.
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=config.training.n_jitted_steps,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

    # Newly generate dataset statistics.
    train_pools = []
    if not inceptionv3:
      train_logits = []

    train_iter = iter(train_ds)
    for i, batch in enumerate(train_iter):
      train_batch = jax.tree_util.tree_map(lambda x: x._numpy(), batch)
      train_batch_resize = jax.image.resize(train_batch['image'],
                                            (*train_batch['image'].shape[:-3], *sample_shape[-3:]),
                                            method='nearest')
      train_batch_int = np.clip(train_batch_resize * 255., 0, 255).astype(np.uint8)
      train_batch_images = train_batch_int.reshape((-1, *train_batch_int.shape[-3:]))
      train_latents = evaluation.run_inception_distributed(train_batch_images, inception_model,
                                                            inceptionv3=inceptionv3)
      train_pools.append(train_latents['pool_3'])
      if not inceptionv3:
        train_logits.append(train_latents['logits'])
    data_pools = jnp.array(train_pools).reshape(-1, train_pools[0].shape[-1])
    if not inceptionv3:
      data_logits = jnp.array(train_logits).reshape(-1, train_logits[0].shape[-1])
    
    if not inceptionv3:
      np.savez_compressed(data_stats, pool_3=data_pools, logits=data_logits)
    else:
      np.savez_compressed(data_stats, pool_3=data_pools)

  # Compute statistics (FID/KID/IS/straightness)
  all_logits = []
  all_pools = []
  stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))
  wait_message = False

  for stat_file in stats:
    with tf.io.gfile.GFile(stat_file, "rb") as fin:
      stat = np.load(fin)
      if not inceptionv3:
        all_logits.append(stat["logits"])
      all_pools.append(stat["pool_3"])

  if not inceptionv3:
    all_logits = np.concatenate(
      all_logits, axis=0)[:n_samples]
  all_pools = np.concatenate(all_pools, axis=0)[:n_samples]

  if not inceptionv3:
    inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
  else:
    inception_score = -1

  fid = tfgan.eval.frechet_classifier_distance_from_activations(
    data_pools, all_pools)
  # Hack to get tfgan KID work for eager execution.
  tf_data_pools = tf.convert_to_tensor(data_pools)
  tf_all_pools = tf.convert_to_tensor(all_pools)
  kid = tfgan.eval.kernel_classifier_distance_from_activations(
    tf_data_pools, tf_all_pools).numpy()
  del tf_data_pools, tf_all_pools
  gc.collect()

  # Return values
  stats_dict = dict()
  stats_dict["is"] = inception_score
  stats_dict["fid"] = fid
  stats_dict["kid"] = kid
  if 'rf_solver' in config.sampling.predictor:
    straightness_files = tf.io.gfile.glob(os.path.join(sample_dir, "straightness_*.npz"))
    for straightness_file in straightness_files:
      with tf.io.gfile.GFile(straightness_file, "rb") as fin:
        straightness = np.load(straightness_file)
        for k in ['straightness', 'seq_straightness', 'nfe']:
          all_straightness_dict[k].append(straightness[k])

    all_straightness = dict()
    if config.sampling.method == 'pc':
      if config.sampling.predictor == 'rf_solver':
        nfe_multiplier = 1
      elif config.sampling.predictor == 'rf_solver_heun':
        nfe_multiplier = 2
      else:
        raise ValueError()
      stats_dict['nfe'] = config.model.num_scales * nfe_multiplier
      for k in ['straightness', 'seq_straightness']:
        all_straightness[k] = jnp.mean(jnp.asarray([v_ for v_ in all_straightness_dict[k]]))
    elif config.sampling.method == 'ode':
      for k in ['straightness', 'seq_straightness', 'nfe']:
        all_straightness[k] = jnp.mean(jnp.asarray([v_ for v_ in all_straightness_dict[k]]))
      for k in ['straightness_by_t', 'seq_straightness_by_t', 'interval']:
        all_straightness[k] = all_straightness_dict[k]
      stats_dict['nfe'] = all_straightness['nfe']
      all_straightness.pop('nfe')
    else:
      raise NotImplementedError()

    stats_dict['straightness'] = all_straightness

  _ = [tf.io.gfile.remove(f) for f in tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))]
  _ = [tf.io.gfile.remove(f) for f in tf.io.gfile.glob(os.path.join(sample_dir, "straightness_*.npz"))]
  del inception_model

  if not save_samples:
    _ = [tf.io.gfile.remove(f) for f in tf.io.gfile.glob(os.path.join(sample_dir, "samples_*.npz"))] # remove samples
  return stats_dict

def jprint(*args):
  fstring = ""
  arrays = []
  for i, a in enumerate(args):
    if i != 0:
      fstring += " "
    if isinstance(a, str):
      fstring += a
    else:
      fstring += '{}'
      arrays.append(a)

  call(lambda arrays: print(fstring.format(*arrays)), arrays)



def time_to_index(time_cond, reflow_t, adaptive_interval=None):
  """
  Input
    time_cond: jnp.array, shape (B,) of jnp.float32
    reflow_t: float
  Return
    index_cond: jnp.array, shape (B,) of jnp.int32

  Example) non-adaptive, reflow_t = 4 case.
  (0, 0.25] --> return 3
  (0.25, 0.5] --> return 2
  (0.5, 0.75] --> return 1
  (0.75, 1.0] --> return 0
  """
  if adaptive_interval is None: # Uniform interval.
    index_cond = jnp.asarray((1. - time_cond) * reflow_t, jnp.int32)
    index_cond = jnp.asarray(jnp.clip(index_cond, 0.0, reflow_t - 1), jnp.int32)
  else: # If adaptive_interval is present
    assert len(adaptive_interval) == reflow_t + 1
    index_cond = jnp.full(time_cond.shape, -1)
    for i in range(reflow_t):
      logical_and = jnp.logical_and(adaptive_interval[i + 1] < time_cond, index_cond == -1)
      index_cond = jnp.array(logical_and, dtype=jnp.int32) * i + (1 - jnp.array(logical_and, dtype=jnp.int32)) * index_cond
    index_cond = jnp.asarray(jnp.clip(index_cond, 0.0, reflow_t - 1), jnp.int32)
  return index_cond


def get_temb_head(temb, index_cond, reflow_t):
  """
  Input
    temb: jnp.array, shape (B, nf * 4 * reflow_t)
    time_cond: jnp.array, shape (B,) of jnp.int32
    reflow_t: float
  Return
    temb: jnp.array, shape (B, nf * 4)
  """
  B, C = temb.shape
  assert C % reflow_t == 0, f"{C} should be divided by {reflow_t}."
  temb = jnp.reshape(temb, (B, reflow_t, int(C // reflow_t))) # (B, reflow_t, nf * 4)
  index_cond_one_hot = jnp.expand_dims(jax.nn.one_hot(index_cond, reflow_t), 2)   # (B, reflow_t, 1)
  temb = jnp.sum(temb * index_cond_one_hot, axis=1) # (B, nf * 4)
  return temb


def get_timestep(reflow_t, n_steps, adaptive_interval=None):
  """
    Input
      reflow_t: type int, number of reflow fractions
      n_steps: size of the timestep sequence.
      adaptive_interval: None or type np.array of shape (reflow_t + 1,)
    Return
      timestep, shape (reflow_t + 1,) of type jnp.float32

    Flow: sampling
      (1) Divide n_steps into a list of reflow_t integers, whose sum is reflow_t. (n_steps_sequential)
          (Ex.) 13 = 5 + 4 + 4 (reflow_t = 3, n_steps = 13)
      (2) Generate timestep (eps --> 1), divided by
        - reflow_t uniform intervals, then
        - each interval divided uniformly by n_steps_sequential[i].
  """
  assert reflow_t <= n_steps, f"The number of divisions {reflow_t} should be smaller than the number of steps {n_steps}."

  # (1) Get intervals.
  if adaptive_interval is not None: # using adaptive interval
    assert adaptive_interval.shape == (reflow_t + 1,)
    interval = adaptive_interval
  else:
    interval = np.linspace(1.0, 0.0, reflow_t + 1)
  
  # (2) Divide n_steps w.r.t intervals
  n_steps_sequential = [int(n_steps // reflow_t)] * reflow_t
  residual = int(n_steps % reflow_t)
  for i in range(residual):
    n_steps_sequential[i] += 1

  # (3) Generate timestep (can also work in adaptive stepping.)
  timestep = tuple()
  for i in range(reflow_t):
    if i == reflow_t - 1:
      timestep += (np.linspace(interval[i], interval[i + 1], n_steps_sequential[i] + 1),)
    else:
      timestep += (np.linspace(interval[i], interval[i + 1], n_steps_sequential[i] + 1)[:-1],)

  timestep_dict = {
    'timestep'        : timestep,            # tuple of length reflow_t, with each element of shape (n_steps_interval + 1,)
    'interval'        : interval,            # (reflow_t + 1,)
    'n_steps_interval': n_steps_sequential,  # (reflow_t,)
  }

  return timestep_dict


def get_sampling_interval(batch_size, n_steps, timestep_dict, gen_reflow, rng):
  """
  Input
    batch_size: type int, batch size.
    timestep_dict: the dictionary obtained from get_timestep
    gen_reflow: True if reflow, False otherwise.
    rng: jax.random.PRNGKey.

  Return
    vec_t: timestep to be used. Float array with shape (batch_size, n_steps + 1)
    index_t: head index to be used. Integer array with shape (batch_size, n_steps)
  """
  if gen_reflow:
    index_t = jax.random.randint(rng, (batch_size,), minval=0, maxval=len(timestep_dict['n_steps_interval']))
    start_time = jnp.array(timestep_dict['interval'])[index_t]
    end_time = jnp.array(timestep_dict['interval'])[index_t + 1]
    
    vec_t = jnp.linspace(start_time, end_time, n_steps + 1, axis=1)
    index_t = jnp.tile(jnp.expand_dims(index_t, axis=1), (1, n_steps,))
  else:
    assert n_steps == sum(timestep_dict['n_steps_interval'])
    index_t = []
    vec_t = jnp.zeros((0,))
    for i in range(len(timestep_dict['n_steps_interval'])):
      index_t += [i] * timestep_dict['n_steps_interval'][i]
      vec_t = jnp.concatenate([vec_t, timestep_dict['timestep'][i]], axis=0)
    vec_t = jnp.tile(jnp.expand_dims(vec_t, axis=0), (batch_size, 1))
    index_t = jnp.asarray(jnp.tile(jnp.expand_dims(jnp.array(index_t), axis=0), (batch_size, 1)), jnp.int32)

  return vec_t, index_t


def rescale_time(x, to):
  assert to in ['rf', 'diffusion']

  if to == 'rf':
    # (1 -> 0) scale to (eps = 0.001 -> 1) scale
    return (1 - x) * 0.999 + 0.001
  elif to == 'diffusion':
    # (eps = 0.001 -> 1) scale to (1 -> 0) scale
    return 1.0 - (x - 0.001) / 0.999
  else:
    raise NotImplementedError()


# TODO: later for chooosing timesteps for optimal sampling.
# def optimal_timestep_sampling(trajectory, reflow_t, nfe, minimum_distance=0.01):
#   """
#   Obtain the optimal timestep that minimizes the truncation error in the sampling phase,
#   using Dijkstra's algorithm of weighted directed graph with fixed number of edges.

#   Input
#     trajectory: dict.
#       "trajectory": shape (n_steps + 1, B, H, W, C), full trajectory of a minibatch.
#       "t": shape (n_steps + 1,), current time.
#       "segment": shape (reflow_t,), number of function evaluations for each segment. should be summed to `n_steps`.
#   """
#   for k in trajectory:
#     trajectory[k] = np.array(trajectory[k])

#   n_steps = trajectory["t"].shape[0] - 1
#   assert trajectory['segment'].shape[0] == reflow_t
#   assert np.sum(trajectory['segment']) == n_steps
#   assert np.sum(trajectory['trajectory'].shape[0]) == n_steps + 1

#   # Step 1. Get all straightness.
#   trajectory = trajectory[:, 0:128]
#   pass


def optimal_timestep(trajectory, reflow_t, minimum_distance=0.01):
  """
  Obtain the optimal timestep that minimizes the truncation error,
  using `Dijkstra's algorithm of weighted directed graph with fixed number of edges`.

  Input
    trajectory: shape (n_steps + 1, B, H, W, C), full trajectory of a minibatch.

  Output
    optimal_timestep: shape (reflow_t + 1,)

  Workflow.
    - average_gap[i, j]
  """
  assert reflow_t >= 2

  trajectory = trajectory[:, 0:128]

  trajectory = np.array(trajectory)            # (n_steps + 1, B, H, W, C) ex. 401
  curvature = trajectory[1:] - trajectory[:-1] # (n_steps, B, H, W, C) ex. 400
  n_steps_plus_one, B, H, W, C = trajectory.shape
  n_steps = n_steps_plus_one - 1

  straightness_map = np.full((n_steps + 1, n_steps + 1), np.inf) # 401, 401. [i, j]: average straightness from i to j.
  minimum_index = np.ceil(minimum_distance * n_steps).astype(int)

  # Step 1. Get all straightness from start_step to end_step
  # total_curvature:
  #   Integral of straightness from start_step to end_step.
  #   int_{t1}^{t2} || (x_{t2} - x_{t1}) / (t2 - t1) - d/dt curv(t) ||^2 dt
  #   infinity when start_step > end_step (To disable this edge.)
  for start_step in range(n_steps + 1):
    for end_step in range(start_step + 1, n_steps + 1):
      average_diff = (trajectory[end_step] - trajectory[start_step]) / (end_step - start_step)
      curvature_map = curvature[start_step:end_step] # (end_step - start_step, B, H, W, C)
      path_gap = np.expand_dims(average_diff, 0) - curvature_map # (end_step - start_step, B, H, W, C)
      straightness_map[start_step, end_step] = np.mean(np.sum(path_gap ** 2, (2, 3, 4)))
    logging.info(f"{start_step} / {n_steps} complete.")

  np.save(f"assets/cifar10_straightness_map.npy", straightness_map)

  for i in range(n_steps + 1):
    for j in range(n_steps + 1):
      if j > i:
        straightness_map[i, j] *= j - i

  # Step 2. Consider the straightness as weight of a directed graph, where (start_step --> end_step) edges are active.
  # Then, run Dijkstra's algorithm.
  D = np.full((n_steps + 1, reflow_t + 1), np.inf) # D[v, k]: length-k (0 --> v) minimum total weight. (ex. (400 + 1, 12 + 1))
  P = np.full((n_steps + 1, reflow_t + 1), -1) # {k-1}-th element of Minimal length-k (0 --> v) path. (ex. (400 + 1, 12 + 1))

  for k in range(n_steps + 1):
    for v in range(n_steps + 1):
      if k + minimum_index > v:
        straightness_map[k, v] = np.inf

  D[0, 0] = 0 # length k=0, (start, end) = (0, 0)
  for k in range(1, reflow_t + 1):
    for v in range(n_steps + 1):
      # D[u, k - 1]:            Minimum (k-1)-length path of (0 --> u).
      # straightness_map[u, v]: 1-length path of (u --> v)
      total_straightness = D[:, k - 1] + straightness_map[:, v]
      D[v, k] = np.min(total_straightness)    # D[v, k]: achieved shortest-path w.r.t. P[v, k].
      P[v, k] = np.argmin(total_straightness) # P[v, k]: shortest-path is achieved when (k-1)-th vertex is P[v, k].

  # Step 3. Recover the shortest path
  all_path = []
  all_distance = []
  for all_k in range(1, reflow_t + 1): # [1, reflow_t]
    path = [n_steps]
    current_v = n_steps
    for j in range(all_k - 1, 0, -1): # [all_k, all_k - 1, ..., 0]
      path.insert(0, P[current_v, j])
      current_v = P[current_v, j]
    all_path.append(path)
    all_distance.append(D[n_steps, all_k])
  
    # print(all_k)
    # print(all_path[-1], flush=True)
    # print(all_distance[-1], flush=True)
  return all_path, all_distance
