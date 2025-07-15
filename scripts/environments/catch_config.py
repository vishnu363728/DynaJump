# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from omegaconf import DictConfig
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import the lift MDP functions to reuse them
from isaaclab_tasks.manager_based.manipulation.lift import mdp

# Custom termination function for velocity-based episode ending
def object_velocity_below_threshold(env, threshold: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Terminate episode when object velocity is below threshold (object stopped moving)."""
    import torch
    # Get the object asset
    object_asset = env.scene[asset_cfg.name]
    # Get linear velocity magnitude
    lin_vel = object_asset.data.root_lin_vel_w
    vel_magnitude = torch.norm(lin_vel, dim=-1)
    # Return True when velocity is below threshold (episode should end)
    return vel_magnitude < threshold

# Custom reward function to encourage active reaching behavior
def active_interception_reward(env, std: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Reward robot for actively moving toward the falling object."""
    import torch
    
    # Get object and end-effector positions
    object_asset = env.scene[asset_cfg.name]
    ee_frame = env.scene["ee_frame"]
    
    object_pos = object_asset.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # End-effector position
    
    # Calculate distance
    distance = torch.norm(object_pos - ee_pos, dim=-1)
    
    # Higher reward for being closer (exponential reward for very close distances)
    reward = torch.exp(-distance / std)
    
    # Bonus if object is moving downward (falling) and robot is close
    object_vel = object_asset.data.root_lin_vel_w
    is_falling = object_vel[:, 2] < -0.1  # Moving downward
    close_to_object = distance < 0.2  # Within 20cm
    
    # Extra reward for intercepting while falling
    interception_bonus = (is_falling & close_to_object).float() * 5.0
    
    return reward + interception_bonus
    """Terminate episode when object velocity is below threshold (object stopped moving)."""
    import torch
    # Get the object asset
    object_asset = env.scene[asset_cfg.name]
    # Get linear velocity magnitude
    lin_vel = object_asset.data.root_lin_vel_w
    vel_magnitude = torch.norm(lin_vel, dim=-1)
    # Return True when velocity is below threshold (episode should end)
    return vel_magnitude < threshold

# Custom function to get end-effector position from frame transformer
def ee_frame_position(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Get end-effector position from frame transformer (like in lift environment)."""
    import torch
    # Get the frame transformer
    ee_frame = env.scene[asset_cfg.name]
    # Get the target position in world coordinates (first target frame, which is our end_effector)
    # Shape: (num_envs, num_target_frames, 3) -> we want (num_envs, 3)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # First (and only) target frame
    return ee_pos_w

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class CatchBlockSceneCfg(InteractiveSceneCfg):
    """Configuration for the catch block scene with a robot and a thrown block."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


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

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Distance reward - encourage robot to move towards block
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # Catch reward - high reward when block is lifted
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    # Goal tracking reward
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    # Optional: Reward for intercepting moving objects (encourages fast reactions)
    # intercept_bonus = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=2.0)

    # REMOVED penalties that hurt catching performance:
    # - No action_rate penalty (we WANT fast reactions!)
    # - No joint_vel penalty (we WANT fast movements!)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Use our custom velocity-based termination
    object_stopped = DoneTerm(
        func=object_velocity_below_threshold,
        params={
            "threshold": 0.05,  # Block moving slower than 0.05 m/s ends episode
            "asset_cfg": SceneEntityCfg("object")
        }
    )


@configclass
class EventsCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Drop block directly above end-effector (pure gripper timing task)
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Position: right above the gripper/end-effector area
            "pose_range": {"x": (0.45, 0.55), "y": (-0.05, 0.05), "z": (0.6, 0.8)},
            # Velocity: pure drop (no horizontal movement)
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class CatchBlockEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the catch block environment."""

    # Scene settings
    scene: CatchBlockSceneCfg = CatchBlockSceneCfg(num_envs=64, env_spacing=3.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 4.0  # Shorter episodes - robot must act quickly!
        
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        
        # physics settings - optimized for dynamic movement
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        
        # Enable faster simulation for dynamic catching
        self.sim.physx.solver_type = 1  # TGS solver (faster)
        self.sim.physx.min_position_iteration_count = 4
        self.sim.physx.max_position_iteration_count = 16


@configclass
class FrankaCatchBlockEnvCfg(CatchBlockEnvCfg):
    """Franka-specific configuration for the catch block environment."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["panda_joint.*"], 
            scale=0.5, 
            use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object - using similar size to lift task
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0, 0.055],  # Start on table like lift task
                rot=[1, 0, 0, 0],
                lin_vel=[0.0, 0.0, 0.0],
                ang_vel=[0.0, 0.0, 0.0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),  # Same size as lift task - much more visible!
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

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCatchBlockEnvCfg_PLAY(FrankaCatchBlockEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


def get_catch_block_franka_config() -> DictConfig:
    """Get configuration for catch block environment with Franka robot."""
    return FrankaCatchBlockEnvCfg()