# Know Your Boundaries: The Necessity of Explicit Behavior Cloning in Offline RL

This repository contains a code used for experiments reported in the [paper](#anchor)

## Settings

### Prerequisite

```
mujoco
conda
```

- Remember to add the mujoco directory to `LD_LIBRARY_PATH` environment variable.
- All the other dependencies will be handled in the following installation script via conda.

### Install

```
# Conda Libraries (CUDA, cudnn, etc.)
conda env create --file env.yaml --name arq
# in the case of error during creation, use conda update commands:
# conda env update --file env.yaml

# Env Variables
conda activate arq
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
echo '#!/bin/sh\nexport XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo '#!/bin/sh\nunset XLA_FLAGS\n' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
conda deactivate
conda activate arq

# Install Python Libraries
pip install -r requirements.txt --no-deps
```

## Run

1. Training score-based generative model $s_\psi(a|s)$

```
python -m arq.scripts.train_sde_conditional \
    --config_file ./experiments/sde.gin ./experiments/envs/{env}.gin \
    --config_params \
        env_id=\'{env-id}\' \
        num_updates={num-updates} \
        train_sde_conditional.run.Dataset=[@D4RL_Dataset|@RoboMimicDataset] \
        train_sde_conditional.run.SDE=[@VPSDE|@SDE_Ensemble] \
    --log_dir {log-dir}
```

- E.g.
  - locomotion ({hopper,walker2d,halfcheetah}-{medium,expert,medium-expert,medium-replay}-v2)
    ```
    python -m arq.scripts.train_sde_conditional \
        --config_file ./experiments/sde.gin ./experiments/envs/hopper.gin \
        --config_params \
            env_id=\'hopper-medium-v2\' \
        --log_dir ./log/hopper-medium-v2
    ```
  - kitchen (kitchen-{complete,partial,mixed}-v0)
    ```
    python -m arq.scripts.train_sde_conditional \
        --config_file ./experiments/sde.gin ./experiments/envs/kitchen.gin \
        --config_params \
            env_id=\'kitchen-complete-v0\' \
            num_updates=300000 \
        --log_dir ./log/kitchen-complete-v0
    ```
  - Adroit ({pen,door,relocate,hammer}-{human,cloned}-v0)
    ```
    python -m arq.scripts.train_sde_conditional \
        --config_file ./experiments/sde.gin ./experiments/envs/pen.gin \
        --config_params \
            env_id=\'pen-human-v0\' \
            num_updates=300000 \
            train_sde_conditional.run.SDE=@SDE_Ensemble \
        --log_dir ./log/pen-human-v0
    ```
  - Antmaze (antmaze-{umaze,umaze-diverse,medium-play,medium-diverse,large-play,large-diverse}-v0)
    ```
    python -m arq.scripts.train_sde_conditional \
        --config_file ./experiments/sde.gin ./experiments/envs/antmaze.gin \
        --config_params \
            env_id=\'antmaze-umaze-v0\' \
        --log_dir ./log/antmaze-umaze-v0
    ```
  - Robomimic Machine Generated. ({lift,can}-low-mg-sparse-v0)
    ```
    python -m arq.scripts.train_sde_conditional \
        --config_file ./experiments/sde.gin ./experiments/envs/lift.gin
        --config_params \
            env_id=\'lift-low-mg-sparse-v0\' \
            train_sde_conditional.run.Dataset=@RoboMimicDataset \
            train_sde_conditional.run.SDE=@SDE_Ensemble \
        --log_dir ./log/lift-low-mg-sparse-v0
    ```
  - Robomimic Human ({lift,can,square,transport,toolhang}-low-{mh,ph}-v0)
    ```
    python -m arq.scripts.train_sde_conditional \
        --config_file ./experiments/sde.gin ./experiments/envs/can.gin \
        --config_params \
            env_id=\'can-low-ph-v0\' \
            num_updates=150000 \
            train_sde_conditional.run.Dataset=@RoboMimicDataset \
            train_sde_conditional.run.SDE=@SDE_Ensemble \
        --log_dir ./log/can-low-ph-v0
    ```

2. Pregenerate Samples (It would take a while...)

```
python -m arq.scripts.sampling \
    --config_file {path-to-sde}/config.gin ./experiments/sde_sampling.gin \
    --config_params SDE_chkpt=\'{path-to-sde}/model.tf\'
    --log_dir {out-dir}
```

- E.g.
    ```
    python -m arq.scripts.sampling \
        --config_file ./log/pen-human-v0/config.gin ./experiments/sde_sampling.gin \
        --config_params SDE_chkpt=\'./log/pen-human-v0/model.tf\' \
        --log_dir ./log/pen-human-v0/pregen
    ```

3. ARQ Training $Q_\theta$

```
PREGEN={out-dir}; \
python -m arq.scripts.train_arq \
    --config_file ./experiments/arq.gin ./experiments/envs/{env}.gin \
    --config_params \
        env_id=\'{env-id}\' \
        pregen_ac=\'$PREGEN/candidate_actions.pkl\' \
        pregen_ll=\'$PREGEN/candidate_actions_log_prob.pkl\' \
        pregen_ll_original=\'$PREGEN/action_log_prob.pkl\' \
        lr={learning-rate} \
        K={K} \
        Dataset.BaseDataset=[@D4RL_Dataset|@RoboMimicDataset] \
    --log_dir {log-dir}
```

- E.g.
  - Locomotion
    ```
    PREGEN=./log/hopper-medium-v2/pregen; \
    python -m arq.scripts.train_arq \
        --config_file ./experiments/arq.gin ./experiments/envs/hopper.gin \
        --config_params \
            env_id=\'hopper-medium-v2\' \
            pregen_ac=\'$PREGEN/candidate_actions.pkl\' \
            pregen_ll=\'$PREGEN/candidate_actions_log_prob.pkl\' \
            pregen_ll_original=\'$PREGEN/action_log_prob.pkl\' \
        --log_dir ./log/hopper-medium-v2/arq
    ```
  - Kitchen, Adroit
    ```
    PREGEN=./log/pen-human-v0/pregen; \
    python -m arq.scripts.train_arq \
        --config_file ./experiments/arq.gin ./experiments/envs/pen.gin \
        --config_params \
            env_id=\'pen-human-v0\' \
            pregen_ac=\'$PREGEN/candidate_actions.pkl\' \
            pregen_ll=\'$PREGEN/candidate_actions_log_prob.pkl\' \
            pregen_ll_original=\'$PREGEN/action_log_prob.pkl\' \
            lr=1e-4 \
        --log_dir ./log/pen-human-v0/arq
    ```
  - Antmaze
    ```
    PREGEN=./log/antmaze-umaze-v0/pregen; \
    python -m arq.scripts.train_arq \
        --config_file ./experiments/arq.gin ./experiments/envs/antmaze.gin \
        --config_params \
            env_id=\'antmaze-umaze-v0\' \
            pregen_ac=\'$PREGEN/candidate_actions.pkl\' \
            pregen_ll=\'$PREGEN/candidate_actions_log_prob.pkl\' \
            pregen_ll_original=\'$PREGEN/action_log_prob.pkl\' \
            K=3 \
        --log_dir ./log/antmaze-umaze-v0/arq
    ```
  - Robomimic
    ```
    PREGEN=./log/lift-low-ph-v0/pregen; \
    python -m arq.scripts.train_arq \
        --config_file ./experiments/arq.gin ./experiments/envs/lift.gin \
        --config_params \
            env_id=\'lift-low-ph-v0\' \
            pregen_ac=\'$PREGEN/candidate_actions.pkl\' \
            pregen_ll=\'$PREGEN/candidate_actions_log_prob.pkl\' \
            pregen_ll_original=\'$PREGEN/action_log_prob.pkl\' \
            lr=1e-4 \
            Dataset.BaseDataset=@RoboMimicDataset \
        --log_dir ./log/lift-low-ph-v0/arq
    ```

4. ARQ (or $Q^\beta$) + $s_\psi$

```
python -m arq.scripts.sde_policy \
    --config_file {path-to-sde}/config.gin ./experiments/sde_policy.gin ./experiments/envs/{gin}.gin \
    --config_params \
        SDE_chkpt=\'{path-to-sde}/model.tf\' \
        Q_chkpt=\'{path-to-q}/q0.tf\' \
        env_id=\'{env_id}\' \
        alpha={alpha} \
    --log_dir {log-dir}
```

- E.g.
  ```
  python -m arq.scripts.sde_policy \
    --config_file ./log/pen-human-v0/config.gin ./experiments/sde_policy.gin ./experiments/envs/pen.gin \
    --config_params \
        SDE_chkpt=\'./log/pen-human-v0/model.tf\' \
        Q_chkpt=\'./log/pen-human-v0/arq/q0.tf\' \
        env_id=\'pen-human-v0\' \
        alpha=10.0 \
    --log_dir ./log/pen-human-v0/arq_s
  ```

5. ARQ + $\pi_\phi$

```
PREGEN={out-dir}; \
python -m arq.scripts.bc \
    --config_file ./experiments/pi.gin ./experiments/envs/{gin}.gin \
    --config_params \
        env_id=\'{env_id}\' \
        Q_chkpt=\'{path-to-q}/q0.tf\' \
        pregen_ac=\'$PREGEN/candidate_actions.pkl\' \
        pregen_ll=\'$PREGEN/candidate_actions_log_prob.pkl\' \
        pregen_ll_original=\'$PREGEN/action_log_prob.pkl\' \
        alpha={alpha} \
        num_updates={num-updates} \
        bc.train_pi.Policy=[@StateIndependentStochasticPolicy|@det_small/DeterministicPolicy,@det_large/DeterministicPolicy] \
    --log_dir {log-dir}
```

- E.g.
  - Locomotion, Kitchen, Adroit
    ```
    PREGEN=./log/pen-human-v0/pregen;\
    python -m arq.scripts.bc \
        --config_file ./experiments/pi.gin ./experiments/envs/pen.gin \
        --config_params \
            env_id=\'pen-human-v0\' \
            Q_chkpt=\'./log/pen-human-v0/arq/q0.tf\' \
            pregen_ac=\'$PREGEN/candidate_actions.pkl\' \
            pregen_ll=\'$PREGEN/candidate_actions_log_prob.pkl\' \
            pregen_ll_original=\'$PREGEN/action_log_prob.pkl\' \
            alpha=10.0 \
        --log_dir ./log/pen-human-v0/arq_pi
    ```
  - AntMaze
    ```
    PREGEN=./log/antmaze-umaze-v0/pregen;\
    python -m arq.scripts.bc \
        --config_file ./experiments/pi.gin ./experiments/envs/antmaze.gin \
        --config_params \
            env_id=\'antmaze-umaze-v0\' \
            Q_chkpt=\'./log/antmaze-umaze-v0/arq/q0.tf\' \
            pregen_ac=\'$PREGEN/candidate_actions.pkl\' \
            pregen_ll=\'$PREGEN/candidate_actions_log_prob.pkl\' \
            pregen_ll_original=\'$PREGEN/action_log_prob.pkl\' \
            alpha=10.0 \
            bc.train_pi.Policy=@det_small/DeterministicPolicy \
        --log_dir ./log/antmaze-umaze-v0/arq_pi
    ```
  - Robomimic
    ```
    PREGEN=./log/lift-low-ph-v0/pregen;\
    python -m arq.scripts.bc \
        --config_file ./experiments/pi.gin ./experiments/envs/lift.gin \
        --config_params \
            env_id=\'lift-low-ph-v0\' \
            Q_chkpt=\'./log/lift-low-ph-v0/arq/q0.tf\' \
            pregen_ac=\'$PREGEN/candidate_actions.pkl\' \
            pregen_ll=\'$PREGEN/candidate_actions_log_prob.pkl\' \
            pregen_ll_original=\'$PREGEN/action_log_prob.pkl\' \
            alpha=0.1 \
            num_iterations=300000 \
            bc.train_pi.Policy=@det_large/DeterministicPolicy \
        --log_dir ./log/lift-low-ph-v0/arq_pi
    ```

6. Ablation Study: $Q^\beta$ training

```
python -m arq.scripts.policy_evaluation \
    --config_file ./experiments/Q_beta.gin ./experiments/envs/{env}.gin \
    --config_params \
        env_id=\'{env_id}\' \
        policy_evaluation.run.Dataset=[@D4RL_Dataset|@RoboMimicDataset] \
    --log_dir {log-dir}
```

- E.g.
  - D4RL
    ```
    python -m arq.scripts.policy_evaluation \
        --config_file ./experiments/Q_beta.gin ./experiments/envs/hopper.gin \
        --config_params env_id=\'hopper-medium-replay-v2\' \
        --log_dir ./log/hopper-medium-replay-v2/q_beta
    ```
  - Robomimic
    ```
    python -m arq.scripts.policy_evaluation \
        --config_file ./experiments/Q_beta.gin ./experiments/envs/lift.gin \
        --config_params \
            env_id=\'lift-low-ph-v0\' \
            policy_evaluation.run.Dataset=@RoboMimicDataset \
        --log_dir ./log/lift-low-ph-v0/q_beta
    ```

## Citation

```
@article{Goo2022ARQ,
  title = {Know Your Boundaries: The Necessity of Explicit Behavior Cloning in Offline RL},
  author={Goo, Wonjoon and Niekum, Scott},
  journal={arXiv preprint arXiv:2206.00695},
  year={2022}
}
```
