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
"""Training NCSN++ on CelebA with rectified flow."""
from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'rfsde'
  training.continuous = True
  training.reduce_mean = True
  training.batch_size = 16
  training.n_reflow_data = 10000
  training.refresh_reflow_step = 300000
  training.reflow_t = 1
  training.reflow_source_ckpt = 26 # source checkpoint for training reflow.

  training.t_start = 0.0 # time t that x_t begins to be included in the training set
  training.t_end = 1.0 # time t that x_t ends to be included in the training set
  training.all_in_one = False # If true, include all four models: [0.0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]
  training.snapshot_save_freq = 50000
  training.have_reflow_data = False # If true, we already have reflow data.
  training.snapshot_fid_sample = 5000
  training.reflow_mode = 'train_reflow' # 'gen_reflow' or 'train_reflow'.
  training.import_torch = 'RF-Checkpoints/CelebA-HQ/checkpoint_10.pth'
  training.gen_reflow_size = 2048
  training.adaptive_interval = False

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'rf_solver'
  sampling.corrector = 'none'
  sampling.timestep_style = 'uniform'
  sampling.tol = 1e-5

  # data
  data = config.data
  data.centered = True
  data.dataset = 'CelebAHQ'

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.initial_count = 600001
  if model.name == 'iddpm':
    model.ema_rate = 0.9999
    model.nf = 128
    model.ch_mult = (1, 2, 3, 4)
    model.num_res_blocks = 3
    model.attn_resolutions = (16,)
    model.fir_kernel = [1, 1]
    model.init_scale = 0.
    model.nonlinearity = 'swish'
    model.num_scales = 18

    model.label_dim = 0
    model.augment_dim = 0
    model.label_dropout = 0.0
    model.channel_mult_emb = 4

  elif model.name == 'ncsnpp':
    model.scale_by_sigma = True
    model.sigma_max = 380
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'output_skip'
    model.progressive_input = 'input_skip'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3
    model.num_scales = 2000

  else:
    raise NotImplementedError()

  # RF hyperparameters
  model.rf_phase = 1

  # eval
  evaluate = config.eval
  evaluate.begin_step = 10000
  evaluate.end_step = 100000
  evaluate.interval_step = 10000
  evaluate.enable_loss = False
  evaluate.reflow_mode = 'eval_reflow'
  evaluate.save_trajectory = False

  return config
