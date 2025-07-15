# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # Approach reward to cup
    approach_cup = RewTerm(
        func=lambda env, threshold=0.15: 1.0 / (1.0 + ((cup_ee_distance(env) - threshold) ** 2)),
        params={"threshold": 0.15},
        weight=5.0,
    )
    
    # Grasp reward
    grasp_cup = RewTerm(
        func=lambda env, asset_cfg: torch.where(
            (env.scene["cup"].data.contact_forces[asset_cfg.joint_names] > 0).any(dim=-1),
            -torch.sum(env.scene["panda_arm"].data.joint_pos[:, asset_cfg.joint_names], dim=-1) * 5,
            0.0
        ),
        params={"asset_cfg": SceneEntityCfg("cup")},
        weight=2.0,
    )
    
    # Lift reward (when cup is lifted above initial position)
    lift_cup = RewTerm(
        func=lambda env, threshold=0.1: torch.where(
            env.scene["cup"].data.root_pos_w[:, 2] > -0.4 + threshold,
            1.0 / (1.0 + ((env.scene["cup"].data.root_pos_w[:, 2] + 0.4) ** 2)),
            0.0
        ),
        params={"threshold": 0.1},
        weight=3.0,
    )

def action_rate_l2(env: ManagerBasedRLEnv, coefficient=-1e-4) -> torch.Tensor:
    """Penalty for high control frequency."""
    return env.scene["panda_arm"].data.action_rate_l2 * coefficient