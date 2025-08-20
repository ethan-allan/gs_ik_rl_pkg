
# Environment for UR5 Inverse Kinematics Solver with RL 
# Observations: Joint pos, joint vel, end effector pos, end effector orientation, target pos, target orientation
# Actions: Joint positions for the arm 
# Rewards: Distance to target

import torch
import math
from typing import Literal
import genesis as gs
from genesis.utils.geom import xyz_to_quat, transform_quat_by_quat, transform_by_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class UR5IKEnv:
    def __init__(
            self,
            num_envs,
            env_cfg,
            reward_cfg,
            robot_cfg,
            command_cfg,
            show_viewer=True,
    ):
        
        # == Initialise learning environment  ==
        self.num_envs = num_envs
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.command_cfg = command_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(0.0, 0.0, 0.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(num_envs)),
                show_world_frame=True),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # == Setup physical environment == (safe to edit)
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        # add target
        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                scale=0.02,
                fixed=False,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.5, 0.5),
                ),
            ),
        )

         # == add camera ==
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0, 0, 0.2),
                fov=50,
                GUI=True,
            )

         # == further setup ==  
        self.scene.build(n_envs=num_envs, env_spacing=(2.0, 2.0))
        # set pd gains (must be called after scene.build)
        self.robot.set_pd_gains()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)        
        
        # == init buffers ==
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reward_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.goal_pos = torch.zeros(self.num_envs, 3,device=gs.device)

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=gs.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=gs.device)

        self.extras = dict()
        self.extras["observations"] = dict()
    
    def resample_target(self, envs_idx):
        self.goal_pos[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), gs.device)
        self.goal_pos[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), gs.device)
        self.goal_pos[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), gs.device)
        self.target.set_pos(self.goal_pos, zero_velocity=True)

    
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0 

         # reset robot
        self.robot.reset(envs_idx)
        self.resample_target(envs_idx)
         # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        obs, self.extras = self.get_observations()
        return obs, None
    
    def step(self, actions):

        # == update time ==
        self.episode_length_buf += 1

        # == apply action based on task ==
        actions = self.rescale_action(actions)
        self.robot.apply_action(actions)
        self.scene.step()

        # == check termination ==
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # == compute rewards ==
        reward = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs, self.extras = self.get_observations()

        return obs, reward, self.reset_buf, self.extras
    
    def get_observations(self):
        # Current end-effector pose
        
        target_pos= self.target.get_pos()

        joint_pos =self.robot.get_links_pos()
        joint_vel = self.robot.get_links_vel()

        obs_components = [
            target_pos,  # 3D position difference
            joint_pos, # current orientation (4D quaternion)
            joint_vel  # goal pose (7D: pos + quat)
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor, self.extras


    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        shifted_action = action - 0.5  # shift to [-0.5, 0.5]
        rescaled_action = shifted_action * self.action_scales
        self.actions = rescaled_action
        return rescaled_action
    
    def is_episode_complete(self):
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # check if the ee is in the valid position
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]


    def _to_world_frame(
        self,
        position: torch.Tensor,  # [B, 3]
        quaternion: torch.Tensor,  # [B, 4]
        keypoints_offset: torch.Tensor,  # [B, 7, 3]
    ) -> torch.Tensor:
        world = torch.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(keypoints_offset[:, k], quaternion)
        return world   

    def _reward_distance(self):
        ee_pos = self.robot.get_ee_pos()  # [B, 3]
        target_pos = self.target.get_pos()  # [B, 3]
        distance = torch.norm(ee_pos - target_pos, dim=-1)  # [B]
        return -distance


## ------------ robot ----------------
class Manipulator:
    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.MJCF = gs.morphs.MJCF(
            file="assets/universal_robots_ur5e/ur5e.xml",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        self._robot_entity: gs.Entity = scene.add_entity(material=material, morph=morph)


        # == some buffer initialization ==
        self._init()

    def set_pd_gains(self):
        # set control gains
        # Note: the following values are tuned for achieving best behavior with Franka
        # Typically, each new robot would have a different set of parameters.
        # Sometimes high-quality URDF or XML file would also provide this and will be parsed.
        self._robot_entity.set_dofs_kp(
            torch.tensor([4500, 4500, 3500, 3500, 2000, 2000]),
        )
        self._robot_entity.set_dofs_kv(
            torch.tensor([450, 450, 350, 350, 200, 200]),
        )
        self._robot_entity.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12]),
            torch.tensor([87, 87, 87, 87, 12, 12]),
        )

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs # total number of arm: joints
        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
       
    
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._default_joint_angles = self._args["default_arm_dof"]

    def reset(self, envs_idx: torch.IntTensor):
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: torch.IntTensor | None = None):
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def apply_action(self, action: torch.Tensor) -> None:
        
        self._robot_entity.control_dofs_position(position=action)

    def base_pos(self):
        return self._robot_entity.get_pos()

    def get_ee_pos(self):
        """
        The end-effector pose (the hand pose)
        """
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return pos
    
    def get_links_pos(self):
        """
        Get the positions of all links in the robot.
        """
        return self._robot_entity.get_dofs_position()
    
    def get_links_vel(self):
        """
        Get the velocities of all links in the robot.
        """
        return self._robot_entity.get_dofs_velocity()