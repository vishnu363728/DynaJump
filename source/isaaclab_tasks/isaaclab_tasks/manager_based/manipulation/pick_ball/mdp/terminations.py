# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the pick_ball task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.
    
    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("ball").
        
    Returns:
        Boolean tensor indicating whether each environment should terminate.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, *_ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return distance < threshold


def ball_lifted_success(
    env: ManagerBasedRLEnv,
    threshold: float = 0.1,
    target_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Termination condition for completing the main objective of ball lifting.
    
    Args:
        env: The environment.
        threshold: Height threshold for success criteria.
        target_cfg: Configuration for the ball object.
        
    Returns:
        Boolean tensor indicating whether each environment should terminate.
    """
    target_object: RigidObject = env.scene[target_cfg.name]
    
    # Ball is successfully lifted if it's above the threshold height
    ball_height = target_object.data.root_pos_w[:, 2]
    table_height = 0.8  # Approximate table height
    
    success_condition = ball_height > (table_height + threshold)
    
    return success_condition


def ball_dropped_failure(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    position_threshold: float = 0.05,
    joint_threshold: float = 0.01,
) -> torch.Tensor:
    """Termination condition for ball dropping failure.
    
    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        object_cfg: Configuration for the ball object.
        position_threshold: Threshold for position-based criteria.
        joint_threshold: Threshold for joint-based criteria.
        
    Returns:
        Boolean tensor indicating whether each environment should terminate.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball_object: RigidObject = env.scene[object_cfg.name]
    
    # Check if ball fell below table level
    ball_height = ball_object.data.root_pos_w[:, 2]
    ground_level = 0.0
    
    condition_1 = ball_height < ground_level
    
    # Additional failure condition: ball moved too far from robot
    robot_pos = robot.data.root_pos_w
    ball_pos = ball_object.data.root_pos_w
    distance = torch.norm(ball_pos[:, :2] - robot_pos[:, :2], dim=1)
    condition_2 = distance > 2.0  # 2 meters from robot base
    
    return torch.logical_or(condition_1, condition_2)


def task_failed_conditions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_ee_height: float = 1.0,
    min_ee_height: float = 0.1,
    max_distance_from_origin: float = 2.0,
) -> torch.Tensor:
    """Termination condition for task failure scenarios.
    
    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        max_ee_height: Maximum allowed end-effector height.
        min_ee_height: Minimum allowed end-effector height.
        max_distance_from_origin: Maximum allowed distance from origin.
        
    Returns:
        Boolean tensor indicating whether each environment should terminate due to failure.
    """
    # Get end-effector position
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    
    # Check end-effector bounds
    ee_too_high = ee_pos[:, 2] > max_ee_height
    ee_too_low = ee_pos[:, 2] < min_ee_height
    ee_too_far = torch.norm(ee_pos[:, :2], dim=1) > max_distance_from_origin
    
    # Combine failure conditions
    failed = torch.logical_or(ee_too_high, ee_too_low)
    failed = torch.logical_or(failed, ee_too_far)
    
    return failed