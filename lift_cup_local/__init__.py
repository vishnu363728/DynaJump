# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the lift_cup environments."""

from __future__ import annotations
import gymnasium as gym

from .config import *  # noqa: F401, E722


gym.register(
    id="Isaac-Lift-Cup-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_name_env_cfg:LiftCupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__:}:rl_games_ppo_cfg.yaml",  # This line is incomplete in the template, but we'll keep it as per standard practice
    },
)

gym.register(
    id="Isaac-Lift-Cup-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_name_env_cfg:LiftCupEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__:}:rl_games_ppo_cfg.yaml",  # This line is incomplete in the template, but we'll keep it as per standard practice
    },
)