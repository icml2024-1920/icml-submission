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
"""Layers for defining NCSN++.
"""
from typing import Any, Optional, Tuple
from . import layers
from . import up_or_down_sampling
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""
  embedding_size: int = 256
  scale: float = 1.0

  @nn.compact
  def __call__(self, x):
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size,))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""
  method: str = 'cat'

  @nn.compact
  def __call__(self, x, y):
    h = conv1x1(x, y.shape[-1])
    if self.method == 'cat':
      return jnp.concatenate([h, y], axis=-1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""
  skip_rescale: bool = False
  init_scale: float = 0.

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape
    h = nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x)
    q = NIN(C)(h)
    k = NIN(C)(h)
    v = NIN(C)(h)

    w = jnp.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = jnp.reshape(w, (B, H, W, H * W))
    w = jax.nn.softmax(w, axis=-1)
    w = jnp.reshape(w, (B, H, W, H, W))
    h = jnp.einsum('bhwHW,bHWc->bhwc', w, v)
    h = NIN(C, init_scale=self.init_scale)(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):
  out_ch: Optional[int] = None
  with_conv: bool = False
  fir: bool = False
  fir_kernel: Tuple[int] = (1, 3, 3, 1)

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    if not self.fir:
      h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), 'nearest')
      if self.with_conv:
        h = conv3x3(h, out_ch)
    else:
      if not self.with_conv:
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.Conv2d(out_ch,
                                       kernel=3,
                                       up=True,
                                       resample_kernel=self.fir_kernel,
                                       use_bias=True,
                                       kernel_init=default_init())(x)

    assert h.shape == (B, 2 * H, 2 * W, out_ch)
    return h


class Downsample(nn.Module):
  out_ch: Optional[int] = None
  with_conv: bool = False
  fir: bool = False
  fir_kernel: Tuple[int] = (1, 3, 3, 1)

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    if not self.fir:
      if self.with_conv:
        x = conv3x3(x, out_ch, stride=2)
      else:
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
    else:
      if not self.with_conv:
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = up_or_down_sampling.Conv2d(
          out_ch,
          kernel=3,
          down=True,
          resample_kernel=self.fir_kernel,
          use_bias=True,
          kernel_init=default_init())(x)

    assert x.shape == (B, H // 2, W // 2, out_ch)
    return x


class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""
  act: Any
  out_ch: Optional[int] = None
  conv_shortcut: bool = False
  dropout: float = 0.1
  skip_rescale: bool = False
  init_scale: float = 0.

  @nn.compact
  def __call__(self, x, temb=None, train=True):
    B, H, W, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))
    h = conv3x3(h, out_ch)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[:, None, None, :]

    h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
    h = nn.Dropout(self.dropout)(h, deterministic=not train)
    h = conv3x3(h, out_ch, init_scale=self.init_scale)
    if C != out_ch:
      if self.conv_shortcut:
        x = conv3x3(x, out_ch)
      else:
        x = NIN(out_ch)(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp(nn.Module):
  """ResBlock adapted from BigGAN."""
  act: Any
  up: bool = False
  down: bool = False
  out_ch: Optional[int] = None
  dropout: float = 0.1
  fir: bool = False
  fir_kernel: Tuple[int] = (1, 3, 3, 1)
  skip_rescale: bool = True
  init_scale: float = 0.

  @nn.compact
  def __call__(self, x, temb=None, train=True):
    B, H, W, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    h = conv3x3(h, out_ch)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[:, None, None, :]

    h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
    h = nn.Dropout(self.dropout)(h, deterministic=not train)
    h = conv3x3(h, out_ch, init_scale=self.init_scale)
    if C != out_ch or self.up or self.down:
      x = conv1x1(x, out_ch)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockEDM(nn.Module):
  """ResBlock adopted from https://github.com/NVlabs/edm. Difference: This ResNetBlock includes AttnBlock. """
  act: Any
  out_channels: int
  init_scale: float = 0.
  up: bool = False
  down: bool = False
  attention: bool = False
  num_heads: Optional[int] = None
  channels_per_head: int = 64
  dropout: float = 0.0
  skip_scale: float = 1.0
  eps: float = 1e-5
  fir_kernel: Tuple[int] = (1, 1)
  resample_proj: bool = False
  adaptive_scale: bool = True

  @nn.compact
  def __call__(self, x, temb=None, train=True):
    B, H, W, C = x.shape
    in_channels = C
    h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))

    if self.up:
      h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
      x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      # if self.fir:
      #   h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
      #   x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      # else:
      #   h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
      #   x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
      x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      # if self.fir:
      #   h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
      #   x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      # else:
      #   h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
      #   x = up_or_down_sampling.naive_downsample_2d(x, factor=2)
    
    h = conv3x3(h, self.out_channels, init_scale=self.init_scale)
    
    if self.adaptive_scale:
      emb = nn.Dense(self.out_channels * 2, kernel_init=default_init())(temb)
      scale, shift = jnp.split(emb, 2, axis=1)
      h = nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)
      h = self.act(h * (scale + 1.0)[:, None, None, :] + shift[:, None, None, :])
    else:
      emb = nn.Dense(self.out_channels, kernel_init=default_init())(temb)
      h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h + emb[:, None, None, :]))

    h = nn.Dropout(self.dropout)(h, deterministic=not train)
    h = conv3x3(h, self.out_channels, init_scale=self.init_scale)

    if self.out_channels != in_channels or self.up or self.down:
      h += conv1x1(x, self.out_channels) # TODO: Followed from ResnetBlockBigGANpp
    else:
      h += x
    h *= self.skip_scale

    # Additional attention operator, added to ResnetBlockBigGANpp (attention layer after ResBlock)
    num_heads = self.num_heads
    num_heads = 0 if not self.attention else num_heads if num_heads is not None else self.out_channels // self.channels_per_head
    if num_heads:
      h = nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)
      h += nn.SelfAttention(num_heads=num_heads)(h, deterministic=not train)
      h *= self.skip_scale
    return h
