# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData, RigidObjectData
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerData
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """The position of the ball in the robot's root frame.
    
    This function computes the position of the main interaction object relative to the robot's
    base frame, which is useful for understanding spatial relationships independent of the
    robot's position in the world.
    
    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot entity. Defaults to SceneEntityCfg("robot").
        object_cfg: Configuration for the main object entity. Defaults to SceneEntityCfg("ball").
        
    Returns:
        The position of the ball in the robot's root frame as a tensor of shape (num_envs, 3).
    """
    robot = env.scene[robot_cfg.name]
    main_object = env.scene[object_cfg.name]
    object_pos_w = main_object.data.root_pos_w[:, :3]
    object_pos_b, *_ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the main interaction object.
    
    This observation provides the relative position vector from the end-effector to the
    main object, which is crucial for approach and manipulation behaviors.
    
    Args:
        env: The environment instance.
        
    Returns:
        The distance vector from end-effector to object as a tensor of shape (num_envs, 3).
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["ball"].data
    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins.
    
    This function provides the absolute positions of both fingertips in the environment,
    which is useful for fine manipulation tasks and grasp analysis.
    
    Args:
        env: The environment instance.
        
    Returns:
        The flattened positions of all fingertips as a tensor of shape (num_envs, 6).
        The tensor contains [left_finger_x, left_finger_y, left_finger_z, right_finger_x, right_finger_y, right_finger_z].
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)
    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins.
    
    This provides the absolute position of the robot's end-effector in the environment,
    which is fundamental for most manipulation tasks.
    
    Args:
        env: The environment instance.
        
    Returns:
        The end-effector position as a tensor of shape (num_envs, 3).
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins
    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.
    
    This provides the orientation of the robot's end-effector, which is important for
    tasks that require specific approach angles or orientations.
    
    Args:
        env: The environment instance.
        make_quat_unique: If True, the quaternion is made unique by ensuring the real part is positive.
        
    Returns:
        The end-effector quaternion as a tensor of shape (num_envs, 4).
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat


def ball_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the ball relative to the environment origins.
    
    Args:
        env: The environment instance.
        
    Returns:
        The ball position as a tensor of shape (num_envs, 3).
    """
    object_data = env.scene["ball"].data
    return object_data.root_pos_w - env.scene.env_origins


def ball_orientation(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the ball in the environment frame.
    
    Args:
        env: The environment instance.
        make_quat_unique: If True, the quaternion is made unique by ensuring the real part is positive.
        
    Returns:
        The ball quaternion as a tensor of shape (num_envs, 4).
    """
    object_data = env.scene["ball"].data
    object_quat = object_data.root_quat_w
    return math_utils.quat_unique(object_quat) if make_quat_unique else object_quat


def gripper_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The gripper joint positions.
    
    Args:
        env: The environment instance.
        
    Returns:
        The gripper joint positions as a tensor of shape (num_envs, 2).
    """
    robot_data: ArticulationData = env.scene["robot"].data
    return robot_data.joint_pos[:, -2:]