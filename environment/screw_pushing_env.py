"""
螺丝推动强化学习环境
"""

import mujoco
import numpy as np
import random
import os
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from utils.kinematics import create_chain_from_mjcf, solve_inverse_kinematics, get_end_effector_pose
from utils.xml_utils import add_challenge_screws_to_xml


class ScrewPushingEnv(gym.Env):
    """螺丝推动强化学习环境"""

    def __init__(self, xml_file="push-grasp-scene.xml", num_screws=3, min_screw_distance=0.15, max_episode_steps=500):
        super().__init__()

        self.xml_file = xml_file
        self.num_screws = num_screws
        self.min_screw_distance = min_screw_distance
        self.max_episode_steps = max_episode_steps

        # 动作空间：末端执行器的xyz和rpy控制
        self.action_space = spaces.Box(
            low=np.array([-0.01, -0.01, -0.01, -0.05, -0.05, -0.05]),
            high=np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05]),
            dtype=np.float32
        )

        # 观察空间：末端位置(3) + 末端姿态(3) + 所有螺丝位置(3*3) + 螺丝间距(3) + 最小间距信息(1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )

        # 机械臂执行器名称
        self.arm_actuator_names = [
            'e_helfin_joint1_actuator', 'e_helfin_joint2_actuator', 'e_helfin_joint3_actuator',
            'e_helfin_joint4_actuator', 'e_helfin_joint5_actuator', 'e_helfin_joint6_actuator'
        ]

        # 初始化环境
        self._setup_environment()
        self.episode_steps = 0

    def _setup_environment(self):
        """设置MuJoCo环境"""
        print("=== 设置训练环境 ===")
        self.temp_xml, self.screw_positions = add_challenge_screws_to_xml(self.xml_file, self.num_screws)
        
        self.model = mujoco.MjModel.from_xml_path(self.temp_xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = [0, 0, -9.8]
        
        try:
            self.arm_chain = create_chain_from_mjcf("xacro-to-urdf-to-mjcf-converter/mjcf_models/elfin15/elfin15.xml")
            print("运动学链创建成功")
        except Exception as e:
            print(f"运动学链创建失败: {e}")
            self.arm_chain = None
        
        self.arm_actuator_ids = [self.model.actuator(name).id for name in self.arm_actuator_names]
        arm_joint_names = [name.replace('_actuator', '') for name in self.arm_actuator_names]
        self.arm_joint_ids = [self.model.joint(name).id for name in arm_joint_names]
        
        # 应用home位置
        home_joint_positions = [0, 0, -1.57, 0, -1.57, 0]
        for i, joint_id in enumerate(self.arm_joint_ids):
            if i < len(home_joint_positions):
                self.data.qpos[joint_id] = home_joint_positions[i]
                self.data.ctrl[self.arm_actuator_ids[i]] = home_joint_positions[i]

    def _get_end_effector_pose(self):
        """获取末端执行器位置和姿态"""
        if self.arm_chain is not None:
            current_joints = self.data.qpos[self.arm_joint_ids]
            return get_end_effector_pose(self.arm_chain, current_joints)
        else:
            return np.zeros(3), np.zeros(3)

    def _get_screw_positions(self):
        """获取所有螺丝的位置"""
        screw_positions = []
        for i in range(self.num_screws):
            try:
                screw_body_id = self.model.body(f'screw_{i + 1}').id
                pos = self.data.xpos[screw_body_id].copy()
                screw_positions.append(pos)
            except:
                screw_positions.append(np.zeros(3))
        return screw_positions

    def _get_observation(self):
        """获取观察"""
        ee_pos, ee_euler = self._get_end_effector_pose()
        screw_positions = self._get_screw_positions()

        # 填充螺丝位置到固定长度
        padded_screw_positions = []
        for i in range(3):
            if i < len(screw_positions):
                padded_screw_positions.extend(screw_positions[i])
            else:
                padded_screw_positions.extend([0, 0, 0])

        # 计算螺丝间距
        screw_distances = []
        min_distance = float('inf')
        for i in range(len(screw_positions)):
            for j in range(i + 1, len(screw_positions)):
                distance = np.linalg.norm(screw_positions[i] - screw_positions[j])
                screw_distances.append(distance)
                min_distance = min(min_distance, distance)

        while len(screw_distances) < 3:
            screw_distances.append(0.0)
        screw_distances = screw_distances[:3]

        min_distance_normalized = min_distance if min_distance != float('inf') else 0.0

        obs = np.concatenate([
            ee_pos,  # 3
            ee_euler,  # 3
            padded_screw_positions,  # 9
            screw_distances,  # 3
            [min_distance_normalized]  # 1
        ])

        return obs.astype(np.float32)

    def _calculate_reward(self):
        """计算奖励"""
        screw_positions = self._get_screw_positions()
        ee_pos, _ = self._get_end_effector_pose()

        screw_distances = []
        min_distance = float('inf')
        for i in range(len(screw_positions)):
            for j in range(i + 1, len(screw_positions)):
                distance = np.linalg.norm(screw_positions[i] - screw_positions[j])
                screw_distances.append(distance)
                min_distance = min(min_distance, distance)

        if min_distance >= self.min_screw_distance:
            distance_reward = 50.0 + (min_distance - self.min_screw_distance) * 20.0
        else:
            distance_reward = -100.0 * (self.min_screw_distance - min_distance)

        success_reward = 0.0
        all_distances_satisfied = all(d >= self.min_screw_distance for d in screw_distances)
        if all_distances_satisfied:
            success_reward = 200.0

        closest_screw_pair_center = None
        closest_pair_distance = float('inf')
        for i, distance in enumerate(screw_distances):
            if distance < self.min_screw_distance:
                pair_idx = 0
                for si in range(len(screw_positions)):
                    for sj in range(si + 1, len(screw_positions)):
                        if pair_idx == i:
                            center = (screw_positions[si] + screw_positions[sj]) / 2
                            if distance < closest_pair_distance:
                                closest_pair_distance = distance
                                closest_screw_pair_center = center
                            break
                        pair_idx += 1

        proximity_reward = 0.0
        if closest_screw_pair_center is not None:
            ee_to_pair = np.linalg.norm(ee_pos - closest_screw_pair_center)
            proximity_reward = -ee_to_pair * 2.0

        time_penalty = -0.1
        total_reward = distance_reward + success_reward + proximity_reward + time_penalty

        return total_reward, all_distances_satisfied

    def step(self, action):
        """执行一步动作"""
        self.episode_steps += 1

        ee_pos, ee_euler = self._get_end_effector_pose()
        target_pos = ee_pos + action[:3]
        target_euler = ee_euler + action[3:]

        current_joints = self.data.qpos[self.arm_joint_ids]
        target_joints = solve_inverse_kinematics(self.arm_chain, target_pos, target_euler, current_joints)

        for i, actuator_id in enumerate(self.arm_actuator_ids):
            if i < len(target_joints):
                self.data.ctrl[actuator_id] = target_joints[i]

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        reward, success = self._calculate_reward()
        terminated = success
        truncated = self.episode_steps >= self.max_episode_steps

        obs = self._get_observation()
        info = {
            'success': success,
            'episode_steps': self.episode_steps,
            'min_screw_distance': min([np.linalg.norm(p1 - p2) for i, p1 in enumerate(self._get_screw_positions()) for j, p2 in enumerate(self._get_screw_positions()) if i < j], default=0.0)
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._setup_environment()
        self.episode_steps = 0

        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_observation()
        info = {
            'min_screw_distance': min([np.linalg.norm(p1 - p2) for i, p1 in enumerate(self._get_screw_positions()) for j, p2 in enumerate(self._get_screw_positions()) if i < j], default=0.0),
            'episode_steps': self.episode_steps
        }
        return obs, info

    def render(self, mode='human'):
        pass

    def close(self):
        if hasattr(self, 'temp_xml') and os.path.exists(self.temp_xml):
            os.remove(self.temp_xml) 