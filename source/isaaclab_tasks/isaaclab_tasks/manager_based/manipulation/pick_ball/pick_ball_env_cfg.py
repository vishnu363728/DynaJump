# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##

@configclass
class PickBallSceneCfg(InteractiveSceneCfg):
    """Configuration for the pick_ball scene with a robot and task-specific objects.
    
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames.
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    
    # Main interaction object
    ball: RigidObjectCfg = MISSING  # will be populated in joint_pos_env file
    
    # Table (if needed for manipulation tasks)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Standard robot observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # End-effector observations
        eef_pos = ObsTerm(func=mdp.ee_pos)
        eef_quat = ObsTerm(func=mdp.ee_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        # Object observations
        ball_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)

        # Distance observations
        rel_ee_ball_distance = ObsTerm(func=mdp.rel_ee_object_distance)

        # Command observations
        target_ball_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Standard reset events
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Object position randomization
    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.35), "y": (-0.25, 0.25), "z": (0.05, 0.1)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ball", body_names="ball"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task-specific reward terms
    approach_ee_ball = RewTerm(func=mdp.approach_ee_ball, weight=2.0, params={"threshold": 0.2})
    
    # Ball manipulation rewards
    ball_lifting = RewTerm(
        func=mdp.ball_lifting_success,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("ball")},
    )

    # Distance rewards
    reaching_ball = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # Goal tracking rewards
    ball_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    # Standard penalty terms
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Standard terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Task-specific termination conditions
    ball_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("ball")},
    )

    # Success terminations
    success = DoneTerm(func=mdp.ball_lifted_success)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##

@configclass
class PickBallEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick_ball environment."""

    # Scene settings
    scene: PickBallSceneCfg = PickBallSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        
        # General settings
        self.decimation = 2  # Control frequency divider
        self.episode_length_s = 5.0  # Episode length in seconds
        
        # Viewer settings (for visualization)
        self.viewer.eye = (-2.0, 2.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # Camera target
        
        # Simulation settings
        self.sim.dt = 0.01  # 100Hz physics simulation
        self.sim.render_interval = self.decimation  # Rendering frequency
        
        # Physics settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625