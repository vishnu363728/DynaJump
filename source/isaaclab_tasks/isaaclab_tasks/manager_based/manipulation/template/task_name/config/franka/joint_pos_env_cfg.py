# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
# TODO: Import task-specific MDP module
# Example: from isaaclab_tasks.manager_based.{TASK_CATEGORY}.{TASK_NAME} import mdp
# TODO: Import task-specific environment config
# Example: from isaaclab_tasks.manager_based.{TASK_CATEGORY}.{TASK_NAME}.{TASK_NAME}_env_cfg import (
#     FRAME_MARKER_SMALL_CFG,
#     {TASK_NAME_PASCAL}EnvCfg,
# )
# Added imports for rigid objects and props
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg, PreviewSurfaceCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class Franka{TASK_NAME_PASCAL}EnvCfg({TASK_NAME_PASCAL}EnvCfg):
    """Configuration for Franka Panda robot performing {TASK_NAME} task."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Set franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Set Actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # TODO: Implement the objects in the scene and their positions that we defined as MISSING in the environment config file
        #Here is an example of such an object:
        '''
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        '''

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        # No need tgo change this
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )


@configclass
class Franka{TASK_NAME_PASCAL}EnvCfg_PLAY(Franka{TASK_NAME_PASCAL}EnvCfg):
    """Play/demo configuration for Franka Panda robot performing {TASK_NAME} task with reduced complexity."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # TODO: Configure scene for play/demo mode
        self.scene.num_envs = {PLAY_NUM_ENVS}  # e.g., 50
        self.scene.env_spacing = {PLAY_ENV_SPACING}  # e.g., 2.5
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# TODO: Configuration Instructions for LLM:
# 
# When populating this template, replace the following placeholders:
# 
# TASK-SPECIFIC:
# - {TASK_CATEGORY}: Task category (e.g., manipulation, locomotion, navigation)
# - {TASK_NAME}: Task name in lowercase (e.g., cabinet, pick_and_place, door_opening)
# - {TASK_NAME_PASCAL}: Task name in PascalCase (e.g., Cabinet, PickAndPlace, DoorOpening)
# 
# REWARD PARAMETERS (uncomment and populate specific reward overrides):
# - {REWARD_NAME}: Name of reward component to override (e.g., approach_gripper_handle)
# - {GRASP_REWARD}: Name of grasp-related reward component (e.g., grasp_handle)
# - {OFFSET_VALUE}: Offset value for reward calculation (float, e.g., 0.04)
# 
# PLAY CONFIGURATION:
# - {PLAY_NUM_ENVS}: Number of environments for play mode (int, e.g., 50)
# - {PLAY_ENV_SPACING}: Spacing between environments in meters (float, e.g., 2.5)
# 
# Notes:
# 1. Remove TODO comments after populating
# 2. Add task-specific reward overrides in the rewards section
# 3. The Franka Panda robot configuration (joints, links, offsets) remains constant
# 4. Only task-specific parameters need to be customized
# 5. Import the correct task MDP module
# 6. Ensure reward parameter names match the task's reward configuration