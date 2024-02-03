# Sequential Rectified Flow (projectname)

ICML 2024 Submission, "*Sequential Flow Straightening for Generative Modeling*".

## Overview

Straightening the probability flow of the continuous-time generative models, such as diffusion models or flow-based models, is the key to fast sampling through the numerical solvers, existing methods learn a linear path by directly generating the probability path the joint distribution between the noise and data distribution.
One key reason for the slow sampling speed of the ODE-based solvers that simulate these generative models is the high curvature of the ODE trajectory, which explodes the truncation error of the numerical solvers in the low-NFE regime.
To address this challenge, We propose a novel method called *SeqRF*, a learning technique that straightens the probability flow to reduce the global truncation error and hence enable acceleration of sampling and improve the synthesis quality.
In both theoretical and empirical studies, we first observe the straightening property of our *SeqRF*.
Via *SeqRF* over flow-based generative models, We achieved surpassing results on both CIFAR-10 and LSUN-Church datasets.

Our project is optimized for TPUs based on Google clouds.

## Bash script commands

> Install all required packages
```
./scripts/requirements.sh
```
Note that [wandb](https://wandb.ai/) project initialization is required to record the logs of the whole runs.
Please refer to the homepage for wandb project tutorials.

The configuration filenames corresponds to the path of the configuration file, which are located in [this directory](configs/rf).

> Generating the reflow dataset used for k-SeqRF.
```
NDIV=<Number of time divisions>
PROJECTNAME_GEN=<data-generating subdirectory name>
CONFIG_FN=<configuration filename>
CKPT_FN=<checkpoint filename>
python3.9 main.py \
  --config=$CONFIG_FN \
  --eval_folder=$PROJECTNAME_GEN \
  --mode="train" \
  --workdir="exp/"$PROJECTNAME_GEN \
  --log_name="gen-"$PROJECTNAME_GEN"-"$NDIV"div" \
  --config.model.rf_phase=2 \
  --config.training.reflow_t=$NDIV \
  --config.training.n_reflow_data=1000000 \
  --config.training.reflow_mode='gen_reflow' \
  --config.training.import_torch=$CKPT_FN \
  --config.sampling.method='ode' \
  --config.sampling.tol=1e-4 \
  --config.training.batch_size=<training bs> \
  --config.training.gen_reflow_size=<evaluation bs>
```
Since we have run our experiments on [Google TPU](https://cloud.google.com/tpu) with $\texttt{Jax/Flax}$, we provide some optimal batch sizes to run our project.
* For CIFAR-10 dataset, recommended batch size for (training, evaluation) is
  * (512, 2048) for v3-8 TPU node.
  * (512, 1024) for v2-8 TPU node.
* For LSUN-Church dataset, (or other $256\times256$ dataset), recommended batch size for (training, evaluation) is
  * (64, -) for v3-8 TPU node with train-only mode.
  * (-, 256) for v3-8 TPU node with evaluation-only mode.
* For CIFAR-10 dataset, the number of recommended reflow data size is 1M. We provide ablation study on the number of reflow data size in our main paper.

> Training k-SeqRF code

We assume that the whole checkpoints and reflow data are stored in ```/mnt/disk``` in the external disk and symbolic-linked to the main repository.
Otherwise, please modify the code below.
```
NDIV=<Number of time divisions>
PROJECTNAME_GEN=<data-generating subdirectory name>
PROJECTNAME=<training subdirectory name>
CONFIG_FN=<configuration filename>
CKPT_FN=<checkpoint filename>

mkdir "exp/"$PROJECTNAME
mkdir "exp/"$PROJECTNAME"/samples"
ln -s "/mnt/disk/exp/"$PROJECTNAME_GEN"/samples/reflow_batch"  "/mnt/disk/exp/"$PROJECTNAME"/samples/reflow_batch"
python3.9 main.py \
  --config=$CONFIG_FN \
  --eval_folder=$PROJECTNAME \
  --mode="train" \
  --workdir="exp/"$PROJECTNAME \
  --log_name=$TPUNAME"-"$PROJECTNAME \
  --config.model.rf_phase=2 \
  --config.training.n_iters=100001 \
  --config.training.reflow_mode='train_reflow' \
  --config.training.reflow_t=$NDIV \
  --config.training.n_reflow_data=1000000 \
  --config.training.snapshot_sampling=True \
  --config.training.snapshot_freq=10000 \
  --config.model.ema_rate=0.9999 \
  --config.model.variable_ema_rate=False \
  --config.training.import_torch=$CKPT_FN \
  --config.sampling.method='ode' \
  --config.training.snapshot_fid_sample=10000 \
  --config.training.batch_size=<training bs> \
  --config.sampling.tol=1e-4 \
  --config.eval.batch_size=<evaluation bs>
```
If you want to run for SeqRF-Ada model, please add ```--config.training.adaptive_interval=True``` tag at the end of the bash script command.

> Distilling k-SeqRF $\to$ k-SeqRF-Distill
```
NDIV=<Number of time divisions>
PROJECTNAME_GEN=<data-generating subdirectory name>
PROJECTNAME=<training subdirectory name>
CONFIG_FN=<configuration filename>
CKPT_FN=<checkpoint filename>

mkdir "exp/"$PROJECTNAME
mkdir "exp/"$PROJECTNAME"/samples"
ln -s "/mnt/disk/exp/"$PROJECTNAME_GEN"/samples/reflow_batch"  "/mnt/disk/exp/"$PROJECTNAME"/samples/reflow_batch"
python3.9 main.py \
  --config=$CONFIG_FN \
  --eval_folder=$PROJECTNAME \
  --mode="train" \
  --workdir="exp/"$PROJECTNAME \
  --log_name=$TPUNAME"-"$PROJECTNAME \
  --config.model.rf_phase=2 \
  --config.training.n_iters=100001 \
  --config.training.reflow_mode='train_distill' \
  --config.training.reflow_t=$NDIV \
  --config.training.n_reflow_data=1000000 \
  --config.training.snapshot_sampling=True \
  --config.training.snapshot_freq=10000 \
  --config.model.ema_rate=0.9999 \
  --config.model.variable_ema_rate=False \
  --config.training.import_torch=$CKPT_FN \
  --config.sampling.method='pc' \
  --config.training.snapshot_fid_sample=10000 \
  --config.training.batch_size=<training bs> \
  --config.sampling.tol=1e-4 \
  --config.model.num_scales=$NDIV \
  --config.eval.batch_size=<evaluation bs>
```

> Evaluation with ODE solver.
```
PROJECTNAME=<training subdirectory name>
CONFIG_FN=<configuration filename>

python3.9 main.py \
  --config=$CONFIG_FN \
  --eval_folder="sample-"$NDIV"div_tol_1e-3" \
  --mode="eval" \
  --workdir="exp/"$PROJECTNAME \
  --log_name="sample-"$PROJECTNAME"-tol_1e-3" \
  --config.model.rf_phase=2 \
  --config.training.reflow_t=$NDIV \
  --config.eval.num_samples=50000 \
  --config.sampling.method='ode' \
  --config.sampling.tol=1e-3 \
  --config.eval.reflow_mode='eval_reflow' \
  --config.eval.batch_size=2048 \
  --config.training.import_torch='none' \
  --config.eval.begin_step=0 \
  --config.eval.end_step=100000 \
  --config.eval.interval_step=10000 \
  --config.eval.save_trajectory=True
```

> Evaluation with fixed-interval solver, such as Euler and Heun solver.
```
PROJECTNAME=<training subdirectory name>
CONFIG_FN=<configuration filename>

python3.9 main.py \
  --config="configs/rf/cifar10_rf_continuous.py" \
  --eval_folder="sample_"$NDIV"div_euler_100" \
  --mode="eval" \
  --workdir="exp/"$PROJECTNAME \
  --log_name=$TPUNAME"-"$PROJECTNAME \
  --config.model.rf_phase=2 \
  --config.training.reflow_t=$NDIV \
  --config.eval.num_samples=50000 \
  --config.sampling.method='pc' \
  --config.sampling.predictor='rf_solver' \
  --config.eval.reflow_mode='eval_reflow' \
  --config.eval.batch_size=2048 \
  --config.training.import_torch='none' \
  --config.eval.begin_step=0 \
  --config.eval.end_step=100000 \
  --config.eval.interval_step=10000 \
  --config.model.num_scales=100 \
  --config.eval.save_trajectory=True
```
Please replace ```'rf_solver'``` to ```'rf_solver_heun'``` and half ```config.model.num_scales``` for Heun solver with same NFE.

## Checkpoints
For baseline pytorch checkpoints, please refer to the original [Rectified flow](https://github.com/gnobitab/RectifiedFlow) repository.

Here are the $\texttt{Jax/Flax}$-based checkpoints we have developed for our work.
* CIFAR-10 Dataset, Sequential reflow
  * [2-SeqRF](TODO)
  * [4-SeqRF](TODO)
  * [6-SeqRF](TODO)
  * [8-SeqRF](TODO)
  * [12-SeqRF](TODO)
* CIFAR-10 Dataset, Distillation after sequential reflow
  * [2-SeqRF-Distill](TODO)
  * [4-SeqRF-Distill](TODO)
* LSUN-Church Dataset, Sequential reflow
  * [1-SeqRF](TODO)
  * [2-SeqRF](TODO)
  * [4-SeqRF](TODO)
