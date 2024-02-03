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

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with VP SDE."""
from configs.default_celeba_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'rfsde'
  training.continuous = True
  training.reduce_mean = True
  training.batch_size = 128
  training.n_reflow_data = 10000
  training.refresh_reflow_step = 10000
  training.reflow_t = 1

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'rf_solver'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True
  data.image_size = 32

  # model
  model = config.model
  model.name = 'ncsnpp'
  if model.name == 'iddpm':
    model.ema_rate = 0.9999
    model.nf = 128
    model.ch_mult = (1, 2, 3, 4)
    model.num_res_blocks = 3
    model.attn_resolutions = (32, 16, 8,)
    model.fir_kernel = [1, 1]
    model.init_scale = 0.
    model.nonlinearity = 'swish'
    model.num_scales = 18

    model.label_dim = 0
    model.augment_dim = 0
    model.label_dropout = 0.0
    model.channel_mult_emb = 4

  elif model.name == 'ncsnpp':
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.embedding_type = 'positional'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3
    model.num_scales = 100

  else:
    raise NotImplementedError()

  # RF hyperparameters
  model.rf_task = 'gen'
  model.rf_phase = 1

  # eval
  evaluate = config.eval
  evaluate.begin_ckpt = 18
  evaluate.end_ckpt = 26
  evaluate.enable_loss = False
  evaluate.sample_low_resolution = False

  return config
