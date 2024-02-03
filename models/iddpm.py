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

from . import utils, layers, layerspp, normalization
import flax.linen as nn
import functools
import jax.numpy as jnp
import numpy as np
import ml_collections

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
ResnetBlockEDM = layerspp.ResnetBlockEDM
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

@utils.register_model(name='iddpm')
class IDDPM(nn.Module):
  """
  IDDPM model. Ported from https://github.com/NVlabs/edm.
  """
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, time_cond, train=True):       # TODO: conditional (add class_labels args.)
    # config parsing
    config = self.config
    img_resolution = config.data.image_size           # Image resolution at input/output.
    out_channels = config.data.num_channels           # Number of color channels at output.
    label_dim = config.model.label_dim                # TODO: Number of class labels, 0 = unconditional.
    augment_dim = config.model.augment_dim            # TODO: Augmentation label dimensionality, 0 = no augmentation.

    nf = config.model.nf                              # Base multiplier for the number of channels.
    ch_mult = config.model.ch_mult                    # Per-resolution multipliers for the number of channels.
    channel_mult_emb = config.model.channel_mult_emb  # Multiplier for the dimensionality of the embedding vector.
    num_res_blocks = config.model.num_res_blocks      # Number of residual blocks per resolution.
    attn_resolutions = config.model.attn_resolutions  # List of resolutions with self-attention.
    dropout = config.model.dropout                    # List of resolutions with self-attention.
    label_dropout = config.model.label_dropout        # TODO: Dropout probability of class labels for classifier-free guidance.
    fir_kernel = config.model.fir_kernel              # Resample filter.

    get_act = layers.get_act
    init_scale = config.model.init_scale
    ResnetBlock = functools.partial(layerspp.ResnetBlockEDM,
                                    act=get_act(config),
                                    init_scale=init_scale,
                                    fir_kernel=fir_kernel)

    # initialize.
    act = get_act(config)
    emb_channels = nf * channel_mult_emb

    if not config.data.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # (1) mapping
    timesteps = time_cond
    emb = layers.get_timestep_embedding(timesteps, nf)
    map_augment = nn.Dense(nf, use_bias=False, kernel_init=default_initializer()) if augment_dim else None
    map_layer0 = nn.Dense(emb_channels, kernel_init=default_initializer())
    map_layer1 = nn.Dense(emb_channels, kernel_init=default_initializer())
    # map_label = nn.Dense(emb_channels, use_bias=False, kernel_init=default_initializer()) if label_dim else None # TODO: conditional diffusion

    if map_augment is not None and augment_labels is not None:
      emb = emb + map_augment(augment_labels)
    emb = act(map_layer0(emb))
    emb = map_layer1(emb)
    """
    TODO: conditional diffusion.
    if map_label is not None:
      tmp = class_labels
      emb = emb + map_label(tmp)
    """
    emb = act(emb)

    # (2) encoder
    h = x
    cout = h.shape[-1]
    hs = []
    for level, mult in enumerate(ch_mult):
      res = img_resolution >> level
      if level == 0:
        cout = nf * mult
        h = conv3x3(h, cout, init_scale=init_scale)
      else:
        h = ResnetBlock(out_channels=cout, down=True)(h, emb)
      hs.append(h)
      for idx in range(num_res_blocks):
        cout = nf * mult
        h = ResnetBlock(out_channels=cout, attention=(res in attn_resolutions))(h, emb)
        hs.append(h)

    # (3) decoder
    for level, mult in reversed(list(enumerate(ch_mult))):
      res = img_resolution >> level
      if level == len(ch_mult) - 1:
        h = ResnetBlock(out_channels=cout, attention=True)(h, emb)
        h = ResnetBlock(out_channels=cout)(h, emb)
      else:
        h = ResnetBlock(out_channels=cout, up=True)(h, emb)
      for idx in range(num_res_blocks + 1):
        cout = nf * mult
        h_cat = jnp.concatenate([h, hs.pop()], axis=3)
        h = ResnetBlock(out_channels=cout, attention=(res in attn_resolutions))(h_cat, emb)
    
    assert len(hs) == 0 # Check that the intermediate output stack is empty.

    # Output layer
    h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
    h = conv3x3(h, x.shape[-1], init_scale=init_scale)
    return h
