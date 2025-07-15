# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.lift_cup.mdp import *  # noqa: F401, E402


##
# Pre-defined configs
##

FRAME_MARKER_CFG = sim_utils.MarkerCfg(
    prim_path="/Visuals/FrameMarker",
    mesh_name="frame_marker",
    scale=(0.05, 0.05, 0.05),
    color=(0.8, 0.1, 0.1),
)

FRAME_MARKER_CUP_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_CUP_CFG.color = (0.1, 0.8, 0.1)


##
# Scene definition
##

@configclass
class LiftCupSceneCfg(sim_utils.Scene):
    """Configuration for the lift_cup scene with a robot and task-specific objects.
    
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames.
    """

    # robots: will be populated by agent env cfg
    panda_arm = sim_utils.ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Panda/planar_panda/tabletop_gripper/tabletop_gripper_instanceable.usd"),
        init_state=sim_utils.ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1.0, 0, 0, 0)
        ),
    )

    # end-effector sensor: will be populated by agent env cfg
    ee_frame = sim_utils.FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            sim_utils.FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
                name="ee_tcp",
                offset=sim_utils.OffsetCfg(pos=(0.0, 0.0, 0.1034), rot=(0.7071, 0.0, -0.7071, 0.0))
            ],
    )

    # Main interaction object (cup)
    cup = sim_utils.RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Cups/cylinder_usd/cylinder.usd"),
        init_state=sim_utils.ArticulationCfg.InitialStateCfg(
            pos=(0, 0, -0.4),
            rot=(1.0, 0, 0, 0)
        ),
    )

    # Table (if needed for manipulation tasks)
    table = sim_utils.AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=sim_utils.AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, -1.0]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Ground plane
    ground_plane = sim_utils.AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=sim_utils.AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Lighting
    light = sim_utils.AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    # No commands needed for this task
    
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Will be set by agent env cfg
    arm_action: sim_utils.JointPositionActionCfg = MISSING
    gripper_action: sim_utils.BinaryJointPositionActionCfg = MISSING


##
# Environment configuration
##

@configclass
class LiftCupEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lift_cup environment."""

    # Scene settings
    scene: LiftCupSceneCfg = LiftCupSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings (add/remove based on your task needs)
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        
        # General settings
        self.decimation = 2  # Control frequency divider
        self.episode_length_s = 5.0  # Episode length in seconds
        
        # Viewer settings (for visualization)
        self.viewer.eye = (-1.0, 1.0, 1.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, -0.4)  # Camera target

# Note: The rest of the template files are incomplete and need to be filled with specific task details.
# We've generated only the necessary parts for this task.