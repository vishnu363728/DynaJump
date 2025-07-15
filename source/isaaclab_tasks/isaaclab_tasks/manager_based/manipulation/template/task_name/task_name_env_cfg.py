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
class {TASK_NAME_PASCAL}SceneCfg(InteractiveSceneCfg):
    """Configuration for the {TASK_NAME} scene with a robot and task-specific objects.
    
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames.
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    
    # TODO: Add task-specific objects below
    # Choose appropriate object types based on your task requirements

    # Example: Main interaction object (customize based on task)
    {PRIMARY_OBJECT}: RigidObjectCfg | ArticulationCfg = MISSING  # TODO: Define objects, but keep them equal to MISSING here, as it will be populated in joint_pos_env file
    # TODO: Add additional objects as needed
    # Example patterns:
    # DO NOT USE USD FILES AT ALL FOR OBJECTS!!!, if you don't see an example for an object, please construct it to the best of your ability
    # - Fixed objects: Use RigidObjectCfg for simple objects
    # - Articulated objects: Use ArticulationCfg for objects with joints (doors, drawers, etc.)
    # - Deformable objects: Use DeformableObjectCfg for soft objects
    
    # Table (if needed for manipulation tasks)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # TODO: Add frame transformers for interaction points
    # Example for articulated objects with handles/interaction points:
    # {OBJECT_NAME}_frame = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/{OBJECT_NAME}/{BASE_LINK}",
    #     debug_vis=True,
    #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/{OBJECT_NAME}FrameTransformer"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/{OBJECT_NAME}/{INTERACTION_LINK}",
    #             name="{INTERACTION_POINT_NAME}",
    #             offset=OffsetCfg(
    #                 pos=({OFFSET_X}, {OFFSET_Y}, {OFFSET_Z}),
    #                 rot=({ROT_W}, {ROT_X}, {ROT_Y}, {ROT_Z}),  # align with end-effector frame
    #             ),
    #         ),
    #     ],
    # )

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
    
    # TODO: Add commands if your task requires goal-directed behavior
    # Example for pose commands:
    # {COMMAND_NAME} = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=({MIN_X}, {MAX_X}), pos_y=({MIN_Y}, {MAX_Y}), pos_z=({MIN_Z}, {MAX_Z}),
    #         roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )
    pass  # Remove this if adding commands


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

        # TODO: Add task-specific observations
        # Common patterns from examples:
        
        # End-effector observations
        # eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        # eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # gripper_pos = ObsTerm(func=mdp.gripper_pos)

        # Object observations
        # {OBJECT_NAME}_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # {OBJECT_NAME}_orientation = ObsTerm(func=mdp.{OBJECT_NAME}_orientations_in_world_frame)

        # Distance observations
        # rel_ee_{OBJECT_NAME}_distance = ObsTerm(func=mdp.rel_ee_{OBJECT_NAME}_distance)

        # Joint observations for articulated objects
        # {ARTICULATED_OBJECT}_joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel,
        #     params={"asset_cfg": SceneEntityCfg("{ARTICULATED_OBJECT}", joint_names=["{JOINT_NAME}"])},
        # )
        # {ARTICULATED_OBJECT}_joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("{ARTICULATED_OBJECT}", joint_names=["{JOINT_NAME}"])},
        # )

        # Command observations (if using commands)
        # target_{OBJECT_NAME}_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "{COMMAND_NAME}"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # TODO: Add additional observation groups if needed
    # Example for multi-modal observations:
    # @configclass
    # class RGBCameraPolicyCfg(ObsGroup):
    #     """Observations for policy group with RGB images."""
    #     
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False

    # @configclass
    # class SubtaskCfg(ObsGroup):
    #     """Observations for subtask completion."""
    #     
    #     {SUBTASK_1} = ObsTerm(
    #         func=mdp.{SUBTASK_1_FUNCTION},
    #         params={
    #             "robot_cfg": SceneEntityCfg("robot"),
    #             "object_cfg": SceneEntityCfg("{OBJECT_NAME}"),
    #         },
    #     )
    #     
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False

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

    # TODO: Add task-specific reset events
    # Example for object position randomization:
    # reset_{OBJECT_NAME}_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": ({MIN_X}, {MAX_X}), "y": ({MIN_Y}, {MAX_Y}), "z": ({MIN_Z}, {MAX_Z})},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("{OBJECT_NAME}", body_names="{BODY_NAME}"),
    #     },
    # )

    # TODO: Add domain randomization events
    # Example for physics material randomization:
    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 1.25),
    #         "dynamic_friction_range": (0.8, 1.25),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # TODO: Add task-specific reward terms
    # Use the reward function patterns from the reward functions template
    
    # Example approach rewards:
    # approach_ee_{TARGET} = RewTerm(func=mdp.approach_ee_{TARGET}, weight=2.0, params={"threshold": 0.2})
    
    # Example manipulation rewards:
    # {MANIPULATION_REWARD} = RewTerm(
    #     func=mdp.{MANIPULATION_FUNCTION},
    #     weight=5.0,
    #     params={"asset_cfg": SceneEntityCfg("{TARGET_ASSET}", joint_names=["{JOINT_NAME}"])},
    # )

    # Example distance rewards:
    # reaching_{OBJECT} = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # Example goal tracking rewards (if using commands):
    # {OBJECT}_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "{COMMAND_NAME}"},
    #     weight=16.0,
    # )

    # Standard penalty terms
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Standard terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # TODO: Add task-specific termination conditions
    # Use the termination function patterns from the termination functions template

    # Example failure terminations:
    # {OBJECT}_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("{OBJECT_NAME}")},
    # )

    # Example success terminations:
    # success = DoneTerm(func=mdp.{SUCCESS_FUNCTION})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # TODO: Add curriculum terms if needed for progressive training
    # Example for gradually increasing penalty weights:
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )


##
# Environment configuration
##

@configclass
class {TASK_NAME_PASCAL}EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the {TASK_NAME} environment."""

    # Scene settings
    scene: {TASK_NAME_PASCAL}SceneCfg = {TASK_NAME_PASCAL}SceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings (add/remove based on your task needs)
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    # TODO: Uncomment if using commands
    # commands: CommandsCfg = CommandsCfg()
    
    # TODO: Uncomment if using curriculum
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # TODO: Adjust these settings based on your task requirements
        
        # General settings
        self.decimation = 2  # Control frequency divider
        self.episode_length_s = 5.0  # Episode length in seconds
        
        # Viewer settings (for visualization)
        self.viewer.eye = (-2.0, 2.0, 2.0)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # Camera target
        
        # Simulation settings
        self.sim.dt = 0.01  # 100Hz physics simulation
        self.sim.render_interval = self.decimation  # Rendering frequency
        
        # Physics settings (usually don't need to change these)
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


# TODO: Configuration Instructions for LLM:
# 
# When populating this template, replace the following placeholders:
# 
# TASK NAMING:
# - {TASK_NAME}: Task name in lowercase (e.g., "cabinet", "pick_and_place", "door_opening")
# - {TASK_NAME_PASCAL}: Task name in PascalCase (e.g., "Cabinet", "PickAndPlace", "DoorOpening")
# 
# OBJECT CONFIGURATION:
# - {PRIMARY_OBJECT}: Main interaction object name (e.g., "object", "cabinet", "door")
# - {OBJECT_NAME}: Generic object name for templates
# - {ARTICULATED_OBJECT}: Name of articulated objects with joints
# - {TARGET_ASSET}: Asset being manipulated
# 
# INTERACTION POINTS:
# - {INTERACTION_LINK}: Link name for interaction points (e.g., "handle", "button")
# - {INTERACTION_POINT_NAME}: Name for the interaction frame
# - {BASE_LINK}: Base link of articulated objects
# 
# COORDINATE OFFSETS:
# - {OFFSET_X}, {OFFSET_Y}, {OFFSET_Z}: Position offsets for frame transformers
# - {ROT_W}, {ROT_X}, {ROT_Y}, {ROT_Z}: Rotation quaternion for frame alignment
# 
# JOINT CONFIGURATION:
# - {JOINT_NAME}: Specific joint names for articulated objects
# - {JOINT_INDEX}: Joint indices for observations
# 
# COMMAND CONFIGURATION:
# - {COMMAND_NAME}: Name of command terms
# - {MIN_X}, {MAX_X}, etc.: Range limits for commands and randomization
# 
# FUNCTION NAMES:
# - {SUBTASK_1_FUNCTION}: Function names for subtask observations
# - {MANIPULATION_FUNCTION}: Function names for manipulation rewards
# - {SUCCESS_FUNCTION}: Function name for success termination
# 
# CONFIGURATION PATTERNS TO FOLLOW:
# 
# 1. **Simple Object Manipulation** (like lift task):
#    - Use RigidObjectCfg for objects
#    - Add object position/orientation observations
#    - Include distance-based rewards
#    - Add command-based goal tracking
# 
# 2. **Articulated Object Manipulation** (like cabinet task):
#    - Use ArticulationCfg for objects with joints
#    - Add frame transformers for interaction points
#    - Include joint position/velocity observations
#    - Add approach, grasp, and manipulation rewards
# 
# 3. **Multi-Object Tasks** (like stacking):
#    - Multiple object configurations
#    - Subtask observation groups
#    - Complex success termination conditions
#    - Multi-stage reward structures
# 
# STEPS TO CUSTOMIZE:
# 1. Choose appropriate scene objects and their types
# 2. Define interaction points and frame transformers
# 3. Add relevant observations for your task
# 4. Configure appropriate reward structure
# 5. Set up termination conditions
# 6. Add domain randomization events as needed
# 7. Adjust simulation parameters for your task requirements
# 8. Remove TODO comments and unused sections
# 9. Test configuration with your robot-specific implementation

'''

ArticulateCfg:
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .articulation import Articulation


@configclass
class ArticulationCfg(AssetBaseCfg):
    """Configuration parameters for an articulation."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the articulation."""

        # root velocity
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

        # joint state
        joint_pos: dict[str, float] = {".*": 0.0}
        """Joint positions of the joints. Defaults to 0.0 for all joints."""
        joint_vel: dict[str, float] = {".*": 0.0}
        """Joint velocities of the joints. Defaults to 0.0 for all joints."""

    ##
    # Initialize configurations.
    ##

    class_type: type = Articulation

    articulation_root_prim_path: str | None = None
    """Path to the articulation root prim in the USD file.

    If not provided will search for a prim with the ArticulationRootAPI. Should start with a slash.
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    soft_joint_pos_limit_factor: float = 1.0
    """Fraction specifying the range of joint position limits (parsed from the asset) to use. Defaults to 1.0.

    The soft joint position limits are scaled by this factor to specify a safety region within the simulated
    joint position limits. This isn't used by the simulation, but is useful for learning agents to prevent the joint
    positions from violating the limits, such as for termination conditions.

    The soft joint position limits are accessible through the :attr:`ArticulationData.soft_joint_pos_limits` attribute.
    """

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names."""


AssetBaseCfg:
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass

from .asset_base import AssetBase


@configclass
class AssetBaseCfg:
    """The base configuration class for an asset's parameters.

    Please see the :class:`AssetBase` class for more information on the asset class.
    """

    @configclass
    class InitialStateCfg:
        """Initial state of the asset.

        This defines the default initial state of the asset when it is spawned into the simulation, as
        well as the default state when the simulation is reset.

        After parsing the initial state, the asset class stores this information in the :attr:`data`
        attribute of the asset class. This can then be accessed by the user to modify the state of the asset
        during the simulation, for example, at resets.
        """

        # root position
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) of the root in simulation world frame.
        Defaults to (1.0, 0.0, 0.0, 0.0).
        """

    class_type: type[AssetBase] = None
    """The associated asset class. Defaults to None, which means that the asset will be spawned
    but cannot be interacted with via the asset class.

    The class should inherit from :class:`isaaclab.assets.asset_base.AssetBase`.
    """

    prim_path: str = MISSING
    """Prim path (or expression) to the asset.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot`` will be replaced with ``/World/envs/env_.*/Robot``.
    """

    spawn: SpawnerCfg | None = None
    """Spawn configuration for the asset. Defaults to None.

    If None, then no prims are spawned by the asset class. Instead, it is assumed that the
    asset is already present in the scene.
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose."""

    collision_group: Literal[0, -1] = 0
    """Collision group of the asset. Defaults to ``0``.

    * ``-1``: global collision group (collides with all assets in the scene).
    * ``0``: local collision group (collides with other assets in the same environment).
    """

    debug_vis: bool = False
    """Whether to enable debug visualization for the asset. Defaults to ``False``."""


ArticulationCfg



DEFORMABLE OBJECT:


RIGID OBJECT:
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils

from ..asset_base import AssetBase
from .rigid_object_data import RigidObjectData

if TYPE_CHECKING:
    from .rigid_object_cfg import RigidObjectCfg


class RigidObject(AssetBase):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` attribute.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: RigidObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: RigidObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

    """
    Properties
    """

    @property
    def data(self) -> RigidObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        prim_paths = self.root_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """Rigid body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = slice(None)
        # reset external wrench
        self._external_force_b[env_ids] = 0.0
        self._external_torque_b[env_ids] = 0.0

    def write_data_to_sim(self):
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_b.view(-1, 3),
                torque_data=self._external_torque_b.view(-1, 3),
                position_data=None,
                indices=self._ALL_INDICES,
                is_global=False,
            )

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Operations - Write to simulation.
    """

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_link_pose_w[env_ids] = root_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]
        if self._data._root_com_state_w.data is not None:
            expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
                self._data.root_link_pose_w[env_ids, :3],
                self._data.root_link_pose_w[env_ids, 3:7],
                self.data.body_com_pos_b[env_ids, 0, :],
                self.data.body_com_quat_b[env_ids, 0, :],
            )
            self._data.root_com_state_w[env_ids, :3] = expected_com_pos
            self._data.root_com_state_w[env_ids, 3:7] = expected_com_quat
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_link_pose_w.clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # set into simulation
        self.root_physx_view.set_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_com_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids

        # set into internal buffers
        self._data.root_com_pose_w[local_env_ids] = root_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[local_env_ids, :7] = self._data.root_com_pose_w[local_env_ids]

        # get CoM pose in link frame
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        com_quat_b = self.data.body_com_quat_b[local_env_ids, 0, :]
        # transform input CoM pose to link frame
        root_link_pos, root_link_quat = math_utils.combine_frame_transforms(
            root_pose[..., :3],
            root_pose[..., 3:7],
            math_utils.quat_apply(math_utils.quat_inv(com_quat_b), -com_pos_b),
            math_utils.quat_inv(com_quat_b),
        )
        root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)

        # write transformed pose in link frame to sim
        self.write_root_link_pose_to_sim(root_link_pose, env_ids=env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        self.write_root_com_velocity_to_sim(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_com_vel_w[env_ids] = root_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        # make the acceleration zero to prevent reporting old values
        self._data.body_com_acc_w[env_ids] = 0.0
        # set into simulation
        self.root_physx_view.set_velocities(self._data.root_com_vel_w, indices=physx_env_ids)

    def write_root_link_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids

        # set into internal buffers
        self._data.root_link_vel_w[local_env_ids] = root_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[local_env_ids, 7:] = self._data.root_link_vel_w[local_env_ids]

        # get CoM pose in link frame
        quat = self.data.root_link_quat_w[local_env_ids]
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        # transform input velocity to center of mass frame
        root_com_velocity = root_velocity.clone()
        root_com_velocity[:, :3] += torch.linalg.cross(
            root_com_velocity[:, 3:], math_utils.quat_apply(quat, com_pos_b), dim=-1
        )

        # write transformed velocity in CoM frame to sim
        self.write_root_com_velocity_to_sim(root_com_velocity, env_ids=env_ids)

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step.

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
        """
        if forces.any() or torques.any():
            self.has_external_wrench = True
        else:
            self.has_external_wrench = False
            # to be safe, explicitly set value to zero
            forces = torques = 0.0

        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = slice(None)
        # -- body_ids
        if body_ids is None:
            body_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and body_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._external_force_b[env_ids, body_ids] = forces
        self._external_torque_b[env_ids, body_ids] = torques

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find rigid root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a rigid body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'USD RigidBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single rigid body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one rigid body in the prim path tree."
            )

        articulation_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        if len(articulation_prims) != 0:
            if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                raise RuntimeError(
                    f"Found an articulation root when resolving '{self.cfg.prim_path}' for rigid objects. These are"
                    f" located at: '{articulation_prims}' under '{template_prim_path}'. Please disable the articulation"
                    " root in the USD or from code by setting the parameter"
                    " 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the spawn configuration."
                )

        # resolve root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_expr.replace(".*", "*"))

        # check if the rigid body was created
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create rigid body at: {self.cfg.prim_path}. Please check PhysX logs.")

        # log information about the rigid body
        omni.log.info(f"Rigid body initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        omni.log.info(f"Number of instances: {self.num_instances}")
        omni.log.info(f"Number of bodies: {self.num_bodies}")
        omni.log.info(f"Body names: {self.body_names}")

        # container for data access
        self._data = RigidObjectData(self.root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

        # external forces and torques
        self.has_external_wrench = False
        self._external_force_b = torch.zeros((self.num_instances, self.num_bodies, 3), device=self.device)
        self._external_torque_b = torch.zeros_like(self._external_force_b)

        # set information about rigid body into data
        self._data.body_names = self.body_names
        self._data.default_mass = self.root_physx_view.get_masses().clone()
        self._data.default_inertia = self.root_physx_view.get_inertias().clone()

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._root_physx_view = None

SCENE ENTITY CFG:
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from dataclasses import MISSING

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass


@configclass
class SceneEntityCfg:
    """Configuration for a scene entity that is used by the manager's term.

    This class is used to specify the name of the scene entity that is queried from the
    :class:`InteractiveScene` and passed to the manager's term function.
    """

    name: str = MISSING
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """

    joint_names: str | list[str] | None = None
    """The names of the joints from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to joint indices on initialization of the manager and passed to the term
    function as a list of joint indices under :attr:`joint_ids`.
    """

    joint_ids: list[int] | slice = slice(None)
    """The indices of the joints from the asset required by the term. Defaults to slice(None), which means
    all the joints in the asset (if present).

    If :attr:`joint_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    fixed_tendon_names: str | list[str] | None = None
    """The names of the fixed tendons from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to fixed tendon indices on initialization of the manager and passed to the term
    function as a list of fixed tendon indices under :attr:`fixed_tendon_ids`.
    """

    fixed_tendon_ids: list[int] | slice = slice(None)
    """The indices of the fixed tendons from the asset required by the term. Defaults to slice(None), which means
    all the fixed tendons in the asset (if present).

    If :attr:`fixed_tendon_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    body_names: str | list[str] | None = None
    """The names of the bodies from the asset required by the term. Defaults to None.

    The names can be either body names or a regular expression matching the body names.

    These are converted to body indices on initialization of the manager and passed to the term
    function as a list of body indices under :attr:`body_ids`.
    """

    body_ids: list[int] | slice = slice(None)
    """The indices of the bodies from the asset required by the term. Defaults to slice(None), which means
    all the bodies in the asset.

    If :attr:`body_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    object_collection_names: str | list[str] | None = None
    """The names of the objects in the rigid object collection required by the term. Defaults to None.

    The names can be either names or a regular expression matching the object names in the collection.

    These are converted to object indices on initialization of the manager and passed to the term
    function as a list of object indices under :attr:`object_collection_ids`.
    """

    object_collection_ids: list[int] | slice = slice(None)
    """The indices of the objects from the rigid object collection required by the term. Defaults to slice(None),
    which means all the objects in the collection.

    If :attr:`object_collection_names` is specified, this is filled in automatically on initialization of the manager.
    """

    preserve_order: bool = False
    """Whether to preserve indices ordering to match with that in the specified joint, body, or object collection names.
    Defaults to False.

    If False, the ordering of the indices are sorted in ascending order (i.e. the ordering in the entity's joints,
    bodies, or object in the object collection). Otherwise, the indices are preserved in the order of the specified
    joint, body, or object collection names.

    For more details, see the :meth:`isaaclab.utils.string.resolve_matching_names` function.

    .. note::
        This attribute is only used when :attr:`joint_names`, :attr:`body_names`, or :attr:`object_collection_names` are specified.

    """

    def resolve(self, scene: InteractiveScene):
        """Resolves the scene entity and converts the joint and body names to indices.

        This function examines the scene entity from the :class:`InteractiveScene` and resolves the indices
        and names of the joints and bodies. It is an expensive operation as it resolves regular expressions
        and should be called only once.

        Args:
            scene: The interactive scene instance.

        Raises:
            ValueError: If the scene entity is not found.
            ValueError: If both ``joint_names`` and ``joint_ids`` are specified and are not consistent.
            ValueError: If both ``fixed_tendon_names`` and ``fixed_tendon_ids`` are specified and are not consistent.
            ValueError: If both ``body_names`` and ``body_ids`` are specified and are not consistent.
            ValueError: If both ``object_collection_names`` and ``object_collection_ids`` are specified and are not consistent.
        """
        # check if the entity is valid
        if self.name not in scene.keys():
            raise ValueError(f"The scene entity '{self.name}' does not exist. Available entities: {scene.keys()}.")

        # convert joint names to indices based on regex
        self._resolve_joint_names(scene)

        # convert fixed tendon names to indices based on regex
        self._resolve_fixed_tendon_names(scene)

        # convert body names to indices based on regex
        self._resolve_body_names(scene)

        # convert object collection names to indices based on regex
        self._resolve_object_collection_names(scene)

    def _resolve_joint_names(self, scene: InteractiveScene):
        # convert joint names to indices based on regex
        if self.joint_names is not None or self.joint_ids != slice(None):
            entity: Articulation = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.joint_names is not None and self.joint_ids != slice(None):
                if isinstance(self.joint_names, str):
                    self.joint_names = [self.joint_names]
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                joint_ids, _ = entity.find_joints(self.joint_names, preserve_order=self.preserve_order)
                joint_names = [entity.joint_names[i] for i in self.joint_ids]
                if joint_ids != self.joint_ids or joint_names != self.joint_names:
                    raise ValueError(
                        "Both 'joint_names' and 'joint_ids' are specified, and are not consistent."
                        f"\n\tfrom joint names: {self.joint_names} [{joint_ids}]"
                        f"\n\tfrom joint ids: {joint_names} [{self.joint_ids}]"
                        "\nHint: Use either 'joint_names' or 'joint_ids' to avoid confusion."
                    )
            # -- from joint names to joint indices
            elif self.joint_names is not None:
                if isinstance(self.joint_names, str):
                    self.joint_names = [self.joint_names]
                self.joint_ids, _ = entity.find_joints(self.joint_names, preserve_order=self.preserve_order)
                # performance optimization (slice offers faster indexing than list of indices)
                # only all joint in the entity order are selected
                if len(self.joint_ids) == entity.num_joints and self.joint_names == entity.joint_names:
                    self.joint_ids = slice(None)
            # -- from joint indices to joint names
            elif self.joint_ids != slice(None):
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                self.joint_names = [entity.joint_names[i] for i in self.joint_ids]

    def _resolve_fixed_tendon_names(self, scene: InteractiveScene):
        # convert tendon names to indices based on regex
        if self.fixed_tendon_names is not None or self.fixed_tendon_ids != slice(None):
            entity: Articulation = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.fixed_tendon_names is not None and self.fixed_tendon_ids != slice(None):
                if isinstance(self.fixed_tendon_names, str):
                    self.fixed_tendon_names = [self.fixed_tendon_names]
                if isinstance(self.fixed_tendon_ids, int):
                    self.fixed_tendon_ids = [self.fixed_tendon_ids]
                fixed_tendon_ids, _ = entity.find_fixed_tendons(
                    self.fixed_tendon_names, preserve_order=self.preserve_order
                )
                fixed_tendon_names = [entity.fixed_tendon_names[i] for i in self.fixed_tendon_ids]
                if fixed_tendon_ids != self.fixed_tendon_ids or fixed_tendon_names != self.fixed_tendon_names:
                    raise ValueError(
                        "Both 'fixed_tendon_names' and 'fixed_tendon_ids' are specified, and are not consistent."
                        f"\n\tfrom joint names: {self.fixed_tendon_names} [{fixed_tendon_ids}]"
                        f"\n\tfrom joint ids: {fixed_tendon_names} [{self.fixed_tendon_ids}]"
                        "\nHint: Use either 'fixed_tendon_names' or 'fixed_tendon_ids' to avoid confusion."
                    )
            # -- from fixed tendon names to fixed tendon indices
            elif self.fixed_tendon_names is not None:
                if isinstance(self.fixed_tendon_names, str):
                    self.fixed_tendon_names = [self.fixed_tendon_names]
                self.fixed_tendon_ids, _ = entity.find_fixed_tendons(
                    self.fixed_tendon_names, preserve_order=self.preserve_order
                )
                # performance optimization (slice offers faster indexing than list of indices)
                # only all fixed tendon in the entity order are selected
                if (
                    len(self.fixed_tendon_ids) == entity.num_fixed_tendons
                    and self.fixed_tendon_names == entity.fixed_tendon_names
                ):
                    self.fixed_tendon_ids = slice(None)
            # -- from fixed tendon indices to fixed tendon names
            elif self.fixed_tendon_ids != slice(None):
                if isinstance(self.fixed_tendon_ids, int):
                    self.fixed_tendon_ids = [self.fixed_tendon_ids]
                self.fixed_tendon_names = [entity.fixed_tendon_names[i] for i in self.fixed_tendon_ids]

    def _resolve_body_names(self, scene: InteractiveScene):
        # convert body names to indices based on regex
        if self.body_names is not None or self.body_ids != slice(None):
            entity: RigidObject = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.body_names is not None and self.body_ids != slice(None):
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                body_ids, _ = entity.find_bodies(self.body_names, preserve_order=self.preserve_order)
                body_names = [entity.body_names[i] for i in self.body_ids]
                if body_ids != self.body_ids or body_names != self.body_names:
                    raise ValueError(
                        "Both 'body_names' and 'body_ids' are specified, and are not consistent."
                        f"\n\tfrom body names: {self.body_names} [{body_ids}]"
                        f"\n\tfrom body ids: {body_names} [{self.body_ids}]"
                        "\nHint: Use either 'body_names' or 'body_ids' to avoid confusion."
                    )
            # -- from body names to body indices
            elif self.body_names is not None:
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                self.body_ids, _ = entity.find_bodies(self.body_names, preserve_order=self.preserve_order)
                # performance optimization (slice offers faster indexing than list of indices)
                # only all bodies in the entity order are selected
                if len(self.body_ids) == entity.num_bodies and self.body_names == entity.body_names:
                    self.body_ids = slice(None)
            # -- from body indices to body names
            elif self.body_ids != slice(None):
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                self.body_names = [entity.body_names[i] for i in self.body_ids]

    def _resolve_object_collection_names(self, scene: InteractiveScene):
        # convert object names to indices based on regex
        if self.object_collection_names is not None or self.object_collection_ids != slice(None):
            entity: RigidObjectCollection = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.object_collection_names is not None and self.object_collection_ids != slice(None):
                if isinstance(self.object_collection_names, str):
                    self.object_collection_names = [self.object_collection_names]
                if isinstance(self.object_collection_ids, int):
                    self.object_collection_ids = [self.object_collection_ids]
                object_ids, _ = entity.find_objects(self.object_collection_names, preserve_order=self.preserve_order)
                object_names = [entity.object_names[i] for i in self.object_collection_ids]
                if object_ids != self.object_collection_ids or object_names != self.object_collection_names:
                    raise ValueError(
                        "Both 'object_collection_names' and 'object_collection_ids' are specified, and are not"
                        " consistent.\n\tfrom object collection names:"
                        f" {self.object_collection_names} [{object_ids}]\n\tfrom object collection ids:"
                        f" {object_names} [{self.object_collection_ids}]\nHint: Use either 'object_collection_names' or"
                        " 'object_collection_ids' to avoid confusion."
                    )
            # -- from object names to object indices
            elif self.object_collection_names is not None:
                if isinstance(self.object_collection_names, str):
                    self.object_collection_names = [self.object_collection_names]
                self.object_collection_ids, _ = entity.find_objects(
                    self.object_collection_names, preserve_order=self.preserve_order
                )
            # -- from object indices to object names
            elif self.object_collection_ids != slice(None):
                if isinstance(self.object_collection_ids, int):
                    self.object_collection_ids = [self.object_collection_ids]
                self.object_collection_names = [entity.object_names[i] for i in self.object_collection_ids]

'''