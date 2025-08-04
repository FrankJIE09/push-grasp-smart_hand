#!/usr/bin/env python3
"""
改进的螺丝推动强化学习训练脚本
解决训练过程中得分下降的问题
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import os
import gymnasium as gym
from gymnasium import spaces
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import ikpy.chain
import ikpy.link
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, LearningRateScheduler, CallbackList
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from tqdm import tqdm
import yaml
import json


def load_config(config_file="config.yaml"):
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class ImprovedScrewPushingEnv(gym.Env):
    """改进的螺丝推动强化学习环境"""
    
    def __init__(self, config=None, render_mode=None):
        super().__init__()
        
        # 加载配置
        if config is None:
            config = load_config()
        
        # 从配置中读取环境参数
        env_config = config['environment']
        self.xml_file = env_config['xml_file']
        self.num_screws = env_config['num_screws']
        self.min_screw_distance = env_config['min_screw_distance']
        self.max_episode_steps = env_config['max_episode_steps']
        
        # 从配置中读取动作空间参数
        action_config = config['action_space']
        position_step = action_config['position_step']
        self.action_space = spaces.Box(
            low=np.array([-position_step, -position_step]),
            high=np.array([position_step, position_step]),
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
        
        # 保存配置
        self.config = config
        
        # 初始化环境
        self._setup_environment()
        
        # 重置计数器
        self.episode_steps = 0
        
        # 初始化奖励历史
        self.reward_history = []
        self.distance_history = []
        
    def _setup_environment(self):
        """设置MuJoCo环境"""
        # 创建包含训练用螺丝的临时XML
        self.temp_xml, self.screw_positions = self._add_screws_to_xml()
        
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(self.temp_xml)
        self.data = mujoco.MjData(self.model)
        
        # 设置重力
        self.model.opt.gravity[:] = [0, 0, -9.8]
        
        # 获取执行器ID
        try:
            self.arm_actuator_ids = [self.model.actuator(name).id for name in self.arm_actuator_names]
        except Exception as e:
            print(f"❌ 获取机械臂执行器ID失败: {e}")
            return
        
        # 获取机械臂关节ID
        arm_joint_names = [name.replace('_actuator', '') for name in self.arm_actuator_names]
        self.arm_joint_ids = [self.model.joint(name).id for name in arm_joint_names]
        
        # 获取灵巧手执行器ID
        try:
            self.hand_actuator_names = [
                'e_hhand_index_finger', 'e_hhand_middle_finger', 'e_hhand_ring_finger',
                'e_hhand_little_finger', 'e_hhand_thumb_flex', 'e_hhand_thumb_rot'
            ]
            self.hand_actuator_ids = [self.model.actuator(name).id for name in self.hand_actuator_names]
        except Exception as e:
            print(f"❌ 获取灵巧手执行器ID失败: {e}")
            self.hand_actuator_ids = []
    
    def _add_screws_to_xml(self):
        """添加螺丝到XML文件"""
        config = self.config
        screw_config = config['screw']
        mass = screw_config['mass']
        L = screw_config['length']
        r = screw_config['radius']
        target_distance = screw_config['target_distance']
        
        Ixx = Iyy = (1 / 12) * mass * (3 * r ** 2 + L ** 2)
        Izz = (1 / 2) * mass * r ** 2
        com_offset = [0, 0, 0.01]
        
        with open(self.xml_file, 'r') as f:
            xml_content = f.read()
        
        # 移除keyframe定义
        import re
        xml_content = re.sub(r'<keyframe>.*?</keyframe>', '', xml_content, flags=re.DOTALL)
        
        # 添加mesh资源
        if '<asset>' in xml_content:
            xml_content = xml_content.replace('</asset>',
                                            '    <mesh name="screw_mesh" file="stl/screw.STL" scale="0.001 0.001 0.001"/>\n  </asset>')
        else:
            asset_section = '''
  <compiler coordinate="local" inertiafromgeom="auto"/>
  <option timestep="0.002" iterations="50" tolerance="1e-10" solver="Newton" jacobian="dense"/>
  <asset>
    <mesh name="screw_mesh" file="stl/screw.STL" scale="0.001 0.001 0.001"/>
  </asset>'''
            xml_content = xml_content.replace('<mujoco model="push_grasp_scene">',
                                            '<mujoco model="push_grasp_scene">\n' + asset_section)
        
        # 生成螺丝位置
        if self.num_screws == 3:
            center_x, center_y = -0.75, 0.0
            positions = [
                (center_x - target_distance, center_y - target_distance / 2, 0.25, 0, 0, 0),
                (center_x + target_distance, center_y - target_distance / 2, 0.25, 0, 0, 0),
                (center_x, center_y + target_distance, 0.25, 0, 0, 0)
            ]
        else:
            positions = []
            for i in range(self.num_screws):
                x = random.uniform(-0.76, -0.74)
                y = random.uniform(-0.03, 0.03)
                z = 0.25
                roll = random.uniform(0, 2 * np.pi)
                pitch = random.uniform(0, 2 * np.pi)
                yaw = random.uniform(0, 2 * np.pi)
                positions.append((x, y, z, roll, pitch, yaw))
        
        all_screw_bodies = ""
        for i, (x, y, z, roll, pitch, yaw) in enumerate(positions):
            screw_body = f'''
    <body name="screw_{i + 1}" pos="{x} {y} {z}" euler="{roll} {pitch} {yaw}">
      <freejoint name="screw_{i + 1}_freejoint"/>
      <inertial pos="{com_offset[0]} {com_offset[1]} {com_offset[2]}" mass="{mass}" diaginertia="{Ixx:.6f} {Iyy:.6f} {Izz:.6f}"/>
      <geom name="screw_{i + 1}_geom" type="mesh" mesh="screw_mesh" 
            rgba="0.8 0.8 0.8 1" 
            friction="0.8 0.2 0.1" 
            density="7850"
            solref="0.01 0.8" 
            solimp="0.8 0.9 0.01"
            margin="0.001"
            condim="3"/>
    </body>
'''
            all_screw_bodies += screw_body
        
        xml_content = xml_content.replace('</worldbody>', all_screw_bodies + '\n  </worldbody>')
        temp_xml = "temp_improved_rl_scene.xml"
        with open(temp_xml, 'w') as f:
            f.write(xml_content)
        return temp_xml, positions
    
    def _apply_home_position(self):
        """应用home位置"""
        try:
            # 设置机械臂的home位置
            home_joints = [-0.000, 0.67, -2.21, -0.000, -0.27, 0.000]
            home_hand_ctrl = [0.3, -1.27, -1.27, -1.27, -0.5, 1.2]
            
            # 应用到机械臂关节
            for i, joint_id in enumerate(self.arm_joint_ids):
                if i < len(home_joints):
                    self.data.qpos[joint_id] = home_joints[i]
                    self.data.ctrl[self.arm_actuator_ids[i]] = home_joints[i]
            
            # 应用到灵巧手执行器
            for i, actuator_id in enumerate(self.hand_actuator_ids):
                if i < len(home_hand_ctrl):
                    self.data.ctrl[actuator_id] = home_hand_ctrl[i]
            
            mujoco.mj_forward(self.model, self.data)
            
        except Exception as e:
            print(f"❌ 应用home位置失败: {e}")
            default_joints = [0, -0.5, 1.0, 0, -0.5, 0]
            for i, joint_id in enumerate(self.arm_joint_ids):
                if i < len(default_joints):
                    self.data.qpos[joint_id] = default_joints[i]
                    self.data.ctrl[self.arm_actuator_ids[i]] = default_joints[i]
    
    def _get_end_effector_pose(self):
        """获取末端执行器位置和姿态"""
        try:
            # 直接从MuJoCo获取
            ee_body_id = self.model.body('elfin_link6').id
            ee_pos = self.data.xpos[ee_body_id].copy()
            ee_quat = self.data.xquat[ee_body_id].copy()
            ee_euler = Rotation.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]).as_euler('xyz')
            return ee_pos, ee_euler
        except Exception as e:
            print(f"❌ 获取末端执行器姿态失败: {e}")
            return np.zeros(3), np.zeros(3)
    
    def _get_screw_positions(self):
        """获取所有螺丝的位置"""
        screw_positions = []
        for i in range(self.num_screws):
            try:
                screw_body_id = self.model.body(f'screw_{i + 1}').id
                pos = self.data.xpos[screw_body_id].copy()
                screw_positions.append(pos)
            except Exception as e:
                print(f"❌ 获取螺丝{i + 1}位置失败: {e}")
                screw_positions.append(np.zeros(3))
        return screw_positions
    
    def _get_min_screw_distance(self):
        """获取螺丝间的最小距离"""
        screw_positions = self._get_screw_positions()
        min_distance = float('inf')
        
        for i in range(len(screw_positions)):
            for j in range(i + 1, len(screw_positions)):
                distance = np.linalg.norm(screw_positions[i] - screw_positions[j])
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def _get_observation(self):
        """获取观察"""
        # 末端执行器位置和姿态
        ee_pos, ee_euler = self._get_end_effector_pose()
        
        # 所有螺丝位置
        screw_positions = self._get_screw_positions()
        
        # 确保有3个螺丝位置（不足的用零填充）
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
        
        # 填充距离信息到固定长度
        while len(screw_distances) < 3:
            screw_distances.append(0.0)
        screw_distances = screw_distances[:3]
        
        min_distance_normalized = min_distance if min_distance != float('inf') else 0.0
        
        # 组合观察
        obs = np.concatenate([
            ee_pos,  # 3 - 末端位置
            ee_euler,  # 3 - 末端姿态
            padded_screw_positions,  # 9 - 所有螺丝位置(3x3)
            screw_distances,  # 3 - 螺丝间距
            [min_distance_normalized]  # 1 - 最小间距
        ])
        
        return obs.astype(np.float32)
    
    def _calculate_reward(self):
        """计算改进的奖励函数"""
        screw_positions = self._get_screw_positions()
        ee_pos, _ = self._get_end_effector_pose()
        
        # 计算所有螺丝间的距离
        screw_distances = []
        min_distance = float('inf')
        
        for i in range(len(screw_positions)):
            for j in range(i + 1, len(screw_positions)):
                distance = np.linalg.norm(screw_positions[i] - screw_positions[j])
                screw_distances.append(distance)
                min_distance = min(min_distance, distance)
        
        # 从配置中读取奖励参数
        reward_config = self.config['reward']
        distance_base_reward = reward_config['distance_base_reward']
        distance_coefficient = reward_config['distance_coefficient']
        distance_penalty_coefficient = reward_config['distance_penalty_coefficient']
        success_reward_value = reward_config['success_reward']
        proximity_coefficient = reward_config['proximity_coefficient']
        time_penalty_value = reward_config['time_penalty']
        index_finger_coef = reward_config.get('index_finger_coef', 1.0)
        
        # 改进的距离奖励
        if min_distance >= self.min_screw_distance:
            distance_reward = distance_base_reward + (min_distance - self.min_screw_distance) * distance_coefficient
        else:
            # 使用更温和的惩罚
            distance_reward = -distance_penalty_coefficient * (self.min_screw_distance - min_distance) * 0.1
        
        # 任务完成奖励
        success_reward = 0.0
        all_distances_satisfied = all(d >= self.min_screw_distance for d in screw_distances)
        if all_distances_satisfied:
            success_reward = success_reward_value
        
        # 改进的接近奖励
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
        
        # 使用指数衰减的接近奖励
        proximity_reward = 0.0
        if closest_screw_pair_center is not None:
            ee_to_pair = np.linalg.norm(ee_pos - closest_screw_pair_center)
            proximity_reward = -np.exp(-ee_to_pair) * proximity_coefficient
        
        # 改进的食指奖励
        try:
            index_finger_pos = self._get_index_finger_pos()
            if len(screw_positions) > 0:
                screws_center = np.mean(screw_positions, axis=0)
                finger_dist = np.linalg.norm(index_finger_pos[:2] - screws_center[:2])
                index_finger_reward = -finger_dist * index_finger_coef
            else:
                index_finger_reward = 0.0
        except:
            index_finger_reward = 0.0
        
        # 大幅降低时间惩罚
        time_penalty = -time_penalty_value
        
        # 添加进度奖励
        progress_reward = 0.0
        if hasattr(self, 'last_min_distance'):
            if min_distance > self.last_min_distance:
                progress_reward = 5.0
            elif min_distance < self.last_min_distance:
                progress_reward = -1.0
        self.last_min_distance = min_distance
        
        # 添加探索奖励
        exploration_reward = 0.0
        if hasattr(self, 'last_ee_pos'):
            movement = np.linalg.norm(ee_pos - self.last_ee_pos)
            exploration_reward = movement * 0.1  # 鼓励移动
        self.last_ee_pos = ee_pos.copy()
        
        total_reward = (distance_reward + success_reward + proximity_reward + 
                       index_finger_reward + time_penalty + progress_reward + exploration_reward)
        
        # 记录奖励历史
        self.reward_history.append(total_reward)
        self.distance_history.append(min_distance)
        
        return total_reward, all_distances_satisfied
    
    def _get_index_finger_pos(self):
        """获取食指头的世界坐标"""
        try:
            idx_body_id = self.model.body('e_hhand_if_distal_link').id
            pos = self.data.xpos[idx_body_id].copy()
            return pos
        except Exception as e:
            return np.zeros(3)
    
    def step(self, action):
        """执行一步动作"""
        self.episode_steps += 1
        
        # 获取当前末端执行器位置
        ee_pos, ee_euler = self._get_end_effector_pose()
        
        # 处理action
        target_pos = ee_pos.copy()
        target_pos[0] += action[0]
        target_pos[1] += action[1]
        target_pos[2] = self.fixed_z  # 强制z为home高度
        target_euler = self.fixed_euler  # 姿态强制为home
        
        # 简单的逆运动学（直接设置关节角度）
        current_joints = self.data.qpos[self.arm_joint_ids]
        
        # 应用控制信号
        for i, actuator_id in enumerate(self.arm_actuator_ids):
            if i < len(current_joints):
                self.data.ctrl[actuator_id] = current_joints[i]
        
        # 执行物理仿真步
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        # 计算奖励
        reward, success = self._calculate_reward()
        
        # 检查终止条件
        terminated = success
        truncated = self.episode_steps >= self.max_episode_steps
        
        # 获取新观察
        obs = self._get_observation()
        
        # 信息字典
        info = {
            'success': success,
            'episode_steps': self.episode_steps,
            'min_screw_distance': self._get_min_screw_distance(),
            'reward': reward
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 重新生成螺丝位置
        self._setup_environment()
        
        # 应用home位置
        self._apply_home_position()
        
        # 重置步数计数器
        self.episode_steps = 0
        
        # 记录 home_z
        self.home_ee_pos, self.home_ee_euler = self._get_end_effector_pose()
        self.fixed_z = self.home_ee_pos[2]
        self.fixed_euler = self.home_ee_euler.copy()
        
        # 执行几步物理仿真以稳定环境
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_observation()
        info = {
            'min_screw_distance': self._get_min_screw_distance(),
            'episode_steps': self.episode_steps
        }
        return obs, info
    
    def render(self, mode='human'):
        """渲染环境"""
        pass
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'temp_xml') and os.path.exists(self.temp_xml):
            os.remove(self.temp_xml)


class ImprovedTrainingCallback(BaseCallback):
    """改进的训练回调函数"""
    
    def __init__(self, eval_freq=100, save_freq=1000, demo_freq=1000, verbose=1,
                 results_dir=None, models_dir=None, checkpoints_dir=None, evaluations_dir=None):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.demo_freq = demo_freq
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.checkpoints_dir = checkpoints_dir
        self.evaluations_dir = evaluations_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.last_demo_step = 0
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'timesteps': []
        }
    
    def _on_step(self) -> bool:
        # 记录每个episode的信息
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_info = info['episode']
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
                
                # 记录训练统计信息
                self.training_stats['episode_rewards'].append(episode_info['r'])
                self.training_stats['episode_lengths'].append(episode_info['l'])
                self.training_stats['timesteps'].append(self.n_calls)
                
                # 计算成功率
                if len(self.episode_rewards) >= 100:
                    recent_successes = sum(1 for info in self.locals.get('infos', [])[-100:]
                                           if info.get('success', False))
                    success_rate = recent_successes / 100.0
                    self.success_rate.append(success_rate)
                    self.training_stats['success_rate'].append(success_rate)
        
        # 定期保存模型和检查点
        if self.n_calls % self.save_freq == 0:
            if self.checkpoints_dir:
                checkpoint_path = os.path.join(self.checkpoints_dir, f"improved_screw_pushing_checkpoint_{self.n_calls}")
                self.model.save(checkpoint_path)
                print(f"💾 检查点已保存: {checkpoint_path}")
                
                # 保存训练统计信息
                if self.results_dir:
                    stats_path = os.path.join(self.results_dir, 'improved_training_stats.json')
                    with open(stats_path, 'w', encoding='utf-8') as f:
                        json.dump(self.training_stats, f, indent=2, ensure_ascii=False)
        
        return True


def train_improved_screw_pushing_agent(config=None):
    """训练改进的螺丝推动智能体"""
    if config is None:
        config = load_config()
    
    print("🚀 开始训练改进的螺丝推动强化学习智能体...")
    
    # 检测GPU可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️  GPU不可用，使用CPU训练")
    
    # 创建环境
    env = ImprovedScrewPushingEnv(config)
    
    # 检查环境
    try:
        check_env(env)
    except Exception as e:
        print(f"❌ 环境检查失败: {e}")
        return None, env
    
    # 从配置中读取训练参数
    training_config = config['training']
    callback_config = config['callback']
    
    # 创建训练结果文件夹
    results_dir = training_config['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 创建训练结果文件夹: {results_dir}")
    
    # 创建子文件夹
    models_dir = os.path.join(results_dir, "improved_models")
    logs_dir = os.path.join(results_dir, "improved_logs")
    checkpoints_dir = os.path.join(results_dir, "improved_checkpoints")
    evaluations_dir = os.path.join(results_dir, "improved_evaluations")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(evaluations_dir, exist_ok=True)
    
    print(f"📁 创建子文件夹:")
    print(f"   - 模型文件夹: {models_dir}")
    print(f"   - 日志文件夹: {logs_dir}")
    print(f"   - 检查点文件夹: {checkpoints_dir}")
    print(f"   - 评估文件夹: {evaluations_dir}")
    
    # 创建学习率调度器
    def lr_schedule(progress):
        """学习率调度函数"""
        initial_lr = training_config['learning_rate']
        if progress < 0.3:
            return initial_lr
        elif progress < 0.7:
            return initial_lr * 0.5
        else:
            return initial_lr * 0.1
    
    # 创建PPO智能体
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=training_config['learning_rate'],
        n_steps=training_config['n_steps'],
        batch_size=training_config['batch_size'],
        n_epochs=training_config['n_epochs'],
        gamma=training_config['gamma'],
        gae_lambda=training_config['gae_lambda'],
        clip_range=training_config['clip_range'],
        ent_coef=training_config['ent_coef'],
        vf_coef=training_config['vf_coef'],
        max_grad_norm=training_config['max_grad_norm'],
        verbose=1,
        device=device,
        tensorboard_log=logs_dir
    )
    
    # 创建回调函数
    training_callback = ImprovedTrainingCallback(
        eval_freq=callback_config['eval_freq'],
        save_freq=callback_config['save_freq'],
        demo_freq=callback_config['demo_freq'],
        results_dir=results_dir,
        models_dir=models_dir,
        checkpoints_dir=checkpoints_dir,
        evaluations_dir=evaluations_dir
    )
    
    # 创建学习率调度器回调
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    # 组合回调函数
    callback = CallbackList([training_callback, lr_scheduler])
    
    # 开始训练
    total_timesteps = training_config['total_timesteps']
    save_path = os.path.join(models_dir, "improved_screw_pushing_agent")
    print(f"🎯 开始训练，总时间步数: {total_timesteps}")
    print(f"💾 模型将保存至: {save_path}")
    start_time = time.time()
    
    try:
        print("🔄 开始训练...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"✅ 训练完成！用时: {training_time:.2f}秒")
        
        # 保存最终模型
        model.save(save_path)
        print(f"💾 模型已保存至: {save_path}.zip")
        
        # 保存为PyTorch格式
        torch.save(model.policy.state_dict(), f"{save_path}.pth")
        print(f"💾 PyTorch权重已保存至: {save_path}.pth")
        
        # 保存训练配置和结果
        training_info = {
            'training_time': training_time,
            'total_timesteps': total_timesteps,
            'config': config,
            'device': device,
            'model_path': save_path
        }
        
        with open(os.path.join(results_dir, 'improved_training_info.json'), 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        print(f"💾 训练信息已保存至: {os.path.join(results_dir, 'improved_training_info.json')}")
        
        return model, env
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return None, env
    
    finally:
        env.close()


def main():
    """主函数"""
    print("🔧 改进的螺丝推动强化学习系统")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 训练智能体
    model, env = train_improved_screw_pushing_agent(config)
    
    if model is not None:
        print("\n" + "=" * 50)
        print("🎉 训练完成！")
    else:
        print("❌ 训练失败")


if __name__ == "__main__":
    main() 