# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class TaskNameRslRlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for the RSL-RL PPO runner."""
    # TODO: Define training parameters
    # Example:
    # max_iterations = 1000  # TODO
    # save_interval = 50  # TODO
    # experiment_name = "task_name_ppo"  # TODO
    # run_name = "task_name_ppo_run"  # TODO
    # load_run = -1  # TODO
    # checkpoint = "model_*.pt"  # TODO
    # num_steps_per_env = 24  # TODO
    # max_episode_length = 1000  # TODO
    # seed = 1  # TODO
    # num_envs = 4096  # TODO
    # resume = False  # TODO
    # ... (add more as needed)

    policy = RslRlPpoActorCriticCfg(
        # TODO: Fill in actor/critic config
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        # TODO: Fill in algorithm config
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ) 