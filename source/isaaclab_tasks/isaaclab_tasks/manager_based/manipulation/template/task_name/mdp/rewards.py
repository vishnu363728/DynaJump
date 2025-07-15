# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# TODO: Add task-specific reward functions below
# Use the provided examples as templates and customize for your specific task

# =============================================================================
# EXAMPLE 1: Cabinet Task Functions (Reference Implementation)
# These show patterns for approach, alignment, grasping, and manipulation rewards
# =============================================================================

def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    r"""Reward the robot for reaching the drawer handle using inverse-square law.

    It uses a piecewise function to reward the robot for reaching the handle.

    .. math::

        reward = \begin{cases}
            2 * (1 / (1 + distance^2))^2 & \text{if } distance \leq threshold \\
            (1 / (1 + distance^2))^2 & \text{otherwise}
        \end{cases}

    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]

    # Compute the distance of the end-effector to the handle
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
    """
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["cabinet_frame"].data.target_quat_w[..., 0, :]

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)

    # get current x and y direction of the handle
    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    # get current x and z direction of the gripper
    ee_tcp_x, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 2]

    # make sure gripper aligns with the handle
    # in this case, the z direction of the gripper should be close to the -x direction of the handle
    # and the x direction of the gripper should be close to the -y direction of the handle
    # dot product of z and x should be large
    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def align_grasp_around_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus for correct hand orientation around the handle.

    The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    # bonus if left finger is above the drawer handle and right below
    return is_graspable


def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.04) -> torch.Tensor:
    """Reward the robot's gripper reaching the drawer handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Compute the distance of each finger from the handle
    lfinger_dist = torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])
    rfinger_dist = torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))


def grasp_handle(
    env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold

    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the drawer given by the joint position of the drawer.

    The bonus is given when the drawer is open. If the grasp is around the handle, the bonus is doubled.
    """
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = align_grasp_around_handle(env).float()

    return (is_graspable + 1.0) * drawer_pos


def multi_stage_open_drawer(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Multi-stage bonus for opening the drawer.

    Depending on the drawer's position, the reward is given in three stages: easy, medium, and hard.
    This helps the agent to learn to open the drawer in a controlled manner.
    """
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = align_grasp_around_handle(env).float()

    open_easy = (drawer_pos > 0.01) * 0.5
    open_medium = (drawer_pos > 0.2) * is_graspable
    open_hard = (drawer_pos > 0.3) * is_graspable

    return open_easy + open_medium + open_hard


# =============================================================================
# EXAMPLE 2: Pick-and-Place Task Functions (Reference Implementation)
# These show patterns for object manipulation, lifting, and goal tracking
# =============================================================================

def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, *_ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# =============================================================================
# TODO: Task-Specific Reward Functions
# Replace the template functions below with your specific task requirements
# =============================================================================

def approach_ee_{TARGET_OBJECT}(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Reward the robot for reaching the {TARGET_OBJECT} using inverse-square law.
    
    TODO: Customize this function for your specific target object.
    
    Args:
        env: The environment instance.
        threshold: Distance threshold for enhanced reward.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    target_pos = env.scene["{TARGET_OBJECT_FRAME}"].data.target_pos_w[..., 0, :]  # TODO: Replace with correct frame name

    # Compute the distance of the end-effector to the target
    distance = torch.norm(target_pos - ee_tcp_pos, dim=-1, p=2)

    # Reward the robot for reaching the target
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def {TASK_SPECIFIC_ACTION}_success(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for successfully performing {TASK_SPECIFIC_ACTION}.
    
    TODO: Define what constitutes success for your specific task action.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the relevant asset.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    # TODO: Implement success criteria for your task
    # Example patterns:
    # - Joint position criteria: asset.data.joint_pos[:, joint_idx] > threshold
    # - Distance criteria: torch.norm(pos_a - pos_b) < threshold  
    # - Boolean success: torch.where(condition, 1.0, 0.0)
    
    # Placeholder implementation
    asset_data = env.scene[asset_cfg.name].data
    # TODO: Replace with actual success logic
    success_condition = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return success_condition.float()


def {MANIPULATION_TARGET}_position_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    target_cfg: SceneEntityCfg = SceneEntityCfg("{MANIPULATION_TARGET}"),
) -> torch.Tensor:
    """Reward for tracking the desired position of {MANIPULATION_TARGET}.
    
    TODO: Customize this for your specific manipulation target.
    
    Args:
        env: The environment instance.
        std: Standard deviation for tanh-kernel.
        target_cfg: Configuration for the manipulation target.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    target_object: RigidObject = env.scene[target_cfg.name]
    
    # TODO: Define desired position (could be from command, fixed target, etc.)
    # desired_pos = env.command_manager.get_command("{COMMAND_NAME}")[:, :3]  # From command
    # desired_pos = torch.tensor([x, y, z], device=env.device).expand(env.num_envs, -1)  # Fixed target
    
    # Placeholder: assuming fixed target position
    desired_pos = torch.zeros(env.num_envs, 3, device=env.device)  # TODO: Replace with actual target
    
    # Distance to desired position
    distance = torch.norm(target_object.data.root_pos_w - desired_pos, dim=1)
    return 1 - torch.tanh(distance / std)


# TODO: Configuration Instructions for LLM:
# 
# When populating this template, replace the following placeholders:
# 
# OBJECT/TARGET NAMES:
# - {TARGET_OBJECT}: Main interaction object name (e.g., "handle", "button", "object")
# - {TARGET_OBJECT_FRAME}: Frame name for target object (e.g., "cabinet_frame", "door_frame")
# - {MANIPULATION_TARGET}: Object being manipulated (e.g., "drawer", "door", "object")
# 
# TASK-SPECIFIC ACTIONS:
# - {TASK_SPECIFIC_ACTION}: Main task action (e.g., "open_door", "press_button", "insert_peg")
# 
# REWARD FUNCTION PATTERNS TO USE:
# 1. **Approach Rewards**: Use inverse-square law for smooth approach behavior
# 2. **Alignment Rewards**: Use dot products of rotation matrices for orientation alignment
# 3. **Grasp Rewards**: Check finger positions and joint states for proper grasping
# 4. **Success Rewards**: Binary rewards for task completion
# 5. **Distance Rewards**: Use tanh-kernel for smooth distance-based rewards
# 6. **Multi-stage Rewards**: Progressive rewards for complex tasks
# 
# COMMON REWARD PATTERNS:
# - Distance-based: `1 - torch.tanh(distance / std)` or `1.0 / (1.0 + distance^2)`
# - Binary success: `torch.where(condition, 1.0, 0.0)`
# - Joint position: `asset.data.joint_pos[:, joint_idx]`
# - Orientation alignment: `torch.bmm(vec_a, vec_b)` with rotation matrices
# - Conditional rewards: `condition.float() * reward_value`
# 
# STEPS TO CUSTOMIZE:
# 1. Identify your main interaction objects and their frame names
# 2. Define what constitutes task success
# 3. Choose appropriate reward shaping (approach, alignment, manipulation)
# 4. Use the cabinet and pick-and-place examples as reference patterns
# 5. Remove TODO comments and unused template functions
# 6. Ensure scene entity names match your environment configuration
# 7. Test reward magnitudes to ensure proper learning behavior