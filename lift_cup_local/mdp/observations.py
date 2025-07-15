# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Standard robot observations
        joint_pos = ObsTerm(func=lambda env: env.scene["panda_arm"].data.joint_pos_rel)
        joint_vel = ObsTerm(func=lambda env: env.scene["panda_arm"].data.joint_vel_rel)

        # End-effector observations
        ee_pos = ObsTerm(func=lambda env: env.scene["ee_frame"].data.target_pos_w[:, 0, :3])
        ee_quat = ObsTerm(func=lambda env: env.scene["ee_frame"].data.target_quat_w[:, 0, :4])

        # Object observations
        cup_position = ObsTerm(
            func=lambda env: env.scene["cup"].data.root_pos_w - env.scene.env_origins,
            params={"std": 0.1}
        )
        
        cup_orientation = ObsTerm(
            func=lambda env: env.scene["cup"].data.root_quat_w,
            params={"make_unique": True, "std": 0.3}
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


def object_position_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the cup in the robot's root frame."""
    robot = env.scene["panda_arm"]
    cup = env.scene["cup"]
    # Get cup position and orientation
    cup_pos_w = cup.data.root_pos_w[:, :3]
    cup_quat_w = cup.data.root_quat_w
    
    # Transform to robot frame
    cup_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cup_pos_w)
    
    return cup_pos_b


def ee_fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data = env.scene["ee_frame"].data
    # Get fingertip positions from frame transformer (if available)
    if hasattr(ee_tf_data, 'target_pos_w'):
        return ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

def cup_ee_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the cup."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :3]
    cup_pos = env.scene["cup"].data.root_pos_w[:, :3]
    
    return torch.norm(cup_pos - ee_tcp_pos, dim=-1)

def cup_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The height of the cup above the ground."""
    cup_pos_z = env.scene["cup"].data.root_pos_w[:, 2]
    return cup_pos_z