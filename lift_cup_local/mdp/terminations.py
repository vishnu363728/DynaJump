# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    # Time out termination
    time_out = DoneTerm(func=lambda env, max_time=5.0: mdp.time_out(env), params={"time_out": True})
    
    # Failure conditions (cup falls)
    cup_fallen = DoneTerm(
        func=lambda env, min_height=-0.3: env.scene["cup"].data.root_pos_w[:, 2] < min_height,
        params={"min_height": -0.3},
        time_out=True
    )
    
    # Success condition (cup lifted above threshold)
    cup_lifted_success = DoneTerm(
        func=lambda env, lift_threshold=0.1: env.scene["cup"].data.root_pos_w[:, 2] > -0.4 + lift_threshold,
        params={"lift_threshold": 0.1},
        time_out=False
    )

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high control frequency."""
    return env.scene["panda_arm"].data.action_rate_l2