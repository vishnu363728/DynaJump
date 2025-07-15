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


# TODO: Add task-specific observation functions below
# Follow the existing patterns for consistency

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("{MAIN_OBJECT_NAME}"),  # TODO: Replace with main interaction object
) -> torch.Tensor:
    """The position of the {MAIN_OBJECT_NAME} in the robot's root frame.
    
    This function computes the position of the main interaction object relative to the robot's
    base frame, which is useful for understanding spatial relationships independent of the
    robot's position in the world.
    
    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot entity. Defaults to SceneEntityCfg("robot").
        object_cfg: Configuration for the main object entity. Defaults to SceneEntityCfg("{MAIN_OBJECT_NAME}").
        
    Returns:
        The position of the {MAIN_OBJECT_NAME} in the robot's root frame as a tensor of shape (num_envs, 3).
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
    object_data: ArticulationData = env.scene["{MAIN_OBJECT_NAME}"].data  # TODO: Replace with main object name
    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


# TODO: Add task-specific distance functions
# Example template for additional objects:
def rel_ee_{SECONDARY_OBJECT_NAME}_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the {SECONDARY_OBJECT_NAME}.
    
    Args:
        env: The environment instance.
        
    Returns:
        The distance vector from end-effector to {SECONDARY_OBJECT_NAME} as a tensor of shape (num_envs, 3).
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    secondary_object_data: FrameTransformerData = env.scene["{SECONDARY_OBJECT_FRAME_NAME}"].data  # TODO: Replace with secondary object frame name
    return secondary_object_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


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


# TODO: Add task-specific state observation functions
# Examples of common patterns:

# def {OBJECT_NAME}_position(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """The position of the {OBJECT_NAME} relative to the environment origins.
#     
#     Args:
#         env: The environment instance.
#         
#     Returns:
#         The {OBJECT_NAME} position as a tensor of shape (num_envs, 3).
#     """
#     object_data = env.scene["{OBJECT_NAME}"].data
#     return object_data.root_pos_w - env.scene.env_origins

# def {OBJECT_NAME}_orientation(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
#     """The orientation of the {OBJECT_NAME} in the environment frame.
#     
#     Args:
#         env: The environment instance.
#         make_quat_unique: If True, the quaternion is made unique by ensuring the real part is positive.
#         
#     Returns:
#         The {OBJECT_NAME} quaternion as a tensor of shape (num_envs, 4).
#     """
#     object_data = env.scene["{OBJECT_NAME}"].data
#     object_quat = object_data.root_quat_w
#     return math_utils.quat_unique(object_quat) if make_quat_unique else object_quat

# def {JOINT_NAME}_joint_state(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """The joint state (position and velocity) of {JOINT_NAME}.
#     
#     Args:
#         env: The environment instance.
#         
#     Returns:
#         The joint state as a tensor of shape (num_envs, 2) containing [position, velocity].
#     """
#     articulation_data: ArticulationData = env.scene["{ARTICULATION_NAME}"].data
#     joint_pos = articulation_data.joint_pos[:, {JOINT_INDEX}]  # TODO: Replace with correct joint index
#     joint_vel = articulation_data.joint_vel[:, {JOINT_INDEX}]  # TODO: Replace with correct joint index
#     return torch.stack([joint_pos, joint_vel], dim=-1)


# TODO: Configuration Instructions for LLM:
# 
# When populating this template, replace the following placeholders:
# 
# OBJECT-SPECIFIC:
# - {MAIN_OBJECT_NAME}: Primary interaction object name (e.g., "object", "cabinet", "door")
# - {SECONDARY_OBJECT_NAME}: Secondary object name in function names (e.g., "drawer", "handle", "button")
# - {SECONDARY_OBJECT_FRAME_NAME}: Frame name for secondary object (e.g., "cabinet_frame", "door_frame")
# 
# TASK-SPECIFIC PATTERNS:
# - {OBJECT_NAME}: Replace in template functions with specific object names
# - {JOINT_NAME}: Replace with specific joint names for articulated objects
# - {ARTICULATION_NAME}: Name of articulated asset in scene
# - {JOINT_INDEX}: Integer index of the joint in the articulation
# 
# Notes:
# 1. Keep the existing functions (ee_pos, ee_quat, fingertips_pos) as they're universal
# 2. Customize object_position_in_robot_root_frame and rel_ee_object_distance for your main object
# 3. Add additional rel_ee_*_distance functions for secondary interaction objects
# 4. Uncomment and customize template functions as needed for your specific task
# 5. Remove TODO comments after populating
# 6. Ensure object/frame names match your environment configuration
# 7. Add proper docstrings explaining what each observation represents
# 8. Consider the observation space requirements of your RL algorithm