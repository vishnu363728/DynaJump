# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for task name environments with Franka robot."""

import gymnasium as gym

from . import agents

gym.register(
    # TODO: rename in a way that makes sense for the task for id
    id="Isaac-Open-Drawer-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        # TODO: rename the cabinet to whatever name makes sense for the task
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCabinetEnvCfg",
        # TODO: update the class name to match your task's PPO runner config
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskNameRslRlPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# TODO: renaming to match with above name
gym.register(
    id="Isaac-Open-Drawer-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCabinetEnvCfg_PLAY",
        # TODO: update the class name to match your task's PPO runner config
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskNameRslRlPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)