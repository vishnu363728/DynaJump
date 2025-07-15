# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the {TASK_NAME} task.

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


# TODO: Add task-specific termination functions below
# Use the provided examples as templates and customize for your specific task

# =============================================================================
# EXAMPLE 1: Goal-Reaching Termination (Reference Implementation)
# Shows pattern for object reaching target position with command-based goals
# =============================================================================

def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.
    
    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        
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


# =============================================================================
# EXAMPLE 2: Complex Multi-Object Termination (Reference Implementation)  
# Shows pattern for multiple objects with spatial relationships and gripper state
# =============================================================================

def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    atol=0.0001,
    rtol=0.0001,
):
    """Termination condition for successfully stacking three cubes.
    
    Checks that cubes are properly aligned in x-y plane and stacked with correct height differences,
    and that the gripper is in the open position.
    
    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        cube_1_cfg: Configuration for the first cube.
        cube_2_cfg: Configuration for the second cube.
        cube_3_cfg: Configuration for the third cube.
        xy_threshold: Maximum allowed x-y distance between cube centers.
        height_threshold: Tolerance for height differences.
        height_diff: Expected height difference between stacked cubes.
        gripper_open_val: Expected gripper joint positions when open.
        atol: Absolute tolerance for gripper position check.
        rtol: Relative tolerance for gripper position check.
        
    Returns:
        Boolean tensor indicating whether each environment should terminate.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    
    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w
    
    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)
    
    # Compute cube height difference
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)
    
    # Check cube positions
    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
    stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
    stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)
    
    # Check gripper positions
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol), stacked
    )
    
    return stacked


# =============================================================================
# TODO: Task-Specific Termination Functions
# Replace the template functions below with your specific task requirements
# =============================================================================

def {MAIN_OBJECTIVE}_completed(
    env: ManagerBasedRLEnv,
    threshold: float = 0.02,
    target_cfg: SceneEntityCfg = SceneEntityCfg("{PRIMARY_OBJECT}"),
) -> torch.Tensor:
    """Termination condition for completing the main objective of {MAIN_OBJECTIVE}.
    
    TODO: Define what constitutes successful completion of your main task objective.
    
    Args:
        env: The environment.
        threshold: Distance/angle threshold for success criteria.
        target_cfg: Configuration for the primary target object.
        
    Returns:
        Boolean tensor indicating whether each environment should terminate.
    """
    target_object: RigidObject = env.scene[target_cfg.name]
    
    # TODO: Implement success criteria for your specific task
    # Common patterns:
    # - Position-based: torch.norm(current_pos - target_pos, dim=1) < threshold
    # - Joint-based: torch.abs(joint_pos - target_joint_pos) < threshold
    # - Height-based: object.data.root_pos_w[:, 2] > min_height
    # - Velocity-based: torch.norm(object.data.root_lin_vel_w, dim=1) < vel_threshold
    
    # Placeholder implementation - replace with actual logic
    success_condition = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # Example: Object reached target height
    # success_condition = target_object.data.root_pos_w[:, 2] > threshold
    
    return success_condition


def {SECONDARY_OBJECTIVE}_achieved(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("{SECONDARY_OBJECT}"),
    position_threshold: float = 0.05,
    joint_threshold: float = 0.01,
) -> torch.Tensor:
    """Termination condition for achieving {SECONDARY_OBJECTIVE}.
    
    TODO: Define criteria for secondary objectives (e.g., proper grasp, alignment, etc.).
    
    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        object_cfg: Configuration for the secondary object.
        position_threshold: Threshold for position-based criteria.
        joint_threshold: Threshold for joint-based criteria.
        
    Returns:
        Boolean tensor indicating whether each environment should terminate.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    secondary_object: RigidObject = env.scene[object_cfg.name]
    
    # TODO: Implement secondary objective criteria
    # Examples:
    # - Proper grasp: Check gripper joint positions and object proximity
    # - Alignment: Check relative orientation between objects
    # - Stability: Check object velocities are near zero
    # - Contact: Check force/torque sensor readings
    
    # Placeholder - combine multiple conditions with logical_and
    condition_1 = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)  # TODO: Replace
    condition_2 = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)  # TODO: Replace
    
    return torch.logical_and(condition_1, condition_2)


def {MANIPULATION_TARGET}_in_final_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("{MANIPULATION_TARGET}"),
    final_joint_pos: float = 0.4,
    joint_threshold: float = 0.02,
) -> torch.Tensor:
    """Termination condition for {MANIPULATION_TARGET} reaching its final state.
    
    TODO: Define the final state criteria for articulated objects (e.g., doors, drawers).
    
    Args:
        env: The environment.
        asset_cfg: Configuration for the articulated asset.
        final_joint_pos: Target joint position for the final state.
        joint_threshold: Tolerance for joint position.
        
    Returns:
        Boolean tensor indicating whether each environment should terminate.
    """
    manipulated_asset: Articulation = env.scene[asset_cfg.name]
    
    # TODO: Check joint positions for articulated objects
    # Examples:
    # - Drawer fully open: joint_pos > final_joint_pos - joint_threshold
    # - Door closed: torch.abs(joint_pos - 0.0) < joint_threshold
    # - Valve rotated: torch.abs(joint_pos - target_angle) < angle_threshold
    
    current_joint_pos = manipulated_asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    
    # Check if joint reached target position
    at_target = torch.abs(current_joint_pos - final_joint_pos) < joint_threshold
    
    return at_target


def task_failed_conditions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_ee_height: float = 1.0,
    min_ee_height: float = 0.1,
    max_distance_from_origin: float = 2.0,
) -> torch.Tensor:
    """Termination condition for task failure scenarios.
    
    TODO: Define failure conditions that should terminate the episode early.
    
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
    
    # TODO: Define failure conditions
    # Common failure patterns:
    # - End-effector out of bounds
    # - Object dropped/fell
    # - Robot in unsafe configuration
    # - Excessive forces/torques
    # - Time limit exceeded (handled elsewhere usually)
    
    # Check end-effector bounds
    ee_too_high = ee_pos[:, 2] > max_ee_height
    ee_too_low = ee_pos[:, 2] < min_ee_height
    ee_too_far = torch.norm(ee_pos[:, :2], dim=1) > max_distance_from_origin
    
    # Combine failure conditions
    failed = torch.logical_or(ee_too_high, ee_too_low)
    failed = torch.logical_or(failed, ee_too_far)
    
    return failed


# TODO: Configuration Instructions for LLM:
# 
# When populating this template, replace the following placeholders:
# 
# TASK-SPECIFIC NAMES:
# - {TASK_NAME}: Task name for documentation (e.g., "cabinet opening", "pick and place")
# - {MAIN_OBJECTIVE}: Primary task objective (e.g., "drawer_opened", "object_lifted", "door_closed")
# - {SECONDARY_OBJECTIVE}: Secondary objectives (e.g., "proper_grasp", "stable_hold", "aligned_approach")
# - {MANIPULATION_TARGET}: Articulated object being manipulated (e.g., "drawer", "door", "valve")
# 
# OBJECT NAMES:
# - {PRIMARY_OBJECT}: Main object of interaction (e.g., "object", "handle", "button")
# - {SECONDARY_OBJECT}: Additional objects (e.g., "target", "goal", "container")
# 
# TERMINATION PATTERNS TO USE:
# 1. **Success Terminations**: Task completed successfully
#    - Position-based: Object reached target location
#    - State-based: Articulated object in desired configuration
#    - Multi-condition: Multiple criteria must be satisfied
# 
# 2. **Failure Terminations**: Task failed or unsafe conditions
#    - Safety bounds: Robot/objects outside safe regions
#    - Physical constraints: Objects dropped, excessive forces
#    - Time limits: Task taking too long (usually handled by environment)
# 
# 3. **Complex Terminations**: Multiple objects with relationships
#    - Spatial relationships: Objects properly aligned/stacked
#    - Temporal conditions: Sequence of events completed
#    - State combinations: Multiple objects in correct states
# 
# COMMON TERMINATION PATTERNS:
# - Distance check: `torch.norm(pos_a - pos_b, dim=1) < threshold`
# - Joint position: `torch.abs(joint_pos - target_pos) < tolerance`
# - Height check: `object.data.root_pos_w[:, 2] > min_height`
# - Velocity check: `torch.norm(velocity, dim=1) < vel_threshold`
# - Multi-condition: `torch.logical_and(condition_a, condition_b)`
# - Boolean conversion: `condition.float()` for reward compatibility
# 
# STEPS TO CUSTOMIZE:
# 1. Define what constitutes successful task completion
# 2. Identify failure conditions that should end episodes early
# 3. Use appropriate thresholds for your task scale
# 4. Combine conditions with logical operators as needed
# 5. Remove TODO comments and unused template functions
# 6. Test termination conditions to ensure proper episode management
# 7. Consider computational efficiency for real-time performance