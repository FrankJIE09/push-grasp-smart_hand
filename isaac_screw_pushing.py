#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Isaac Gym的螺丝推动强化学习环境
实现GPU并行仿真，大幅提升训练速度
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Tuple

# Isaac Gym导入
try:
    import isaacgym
    from isaacgym import gymapi, gymtorch
    ISAAC_GYM_AVAILABLE = True
except ImportError:
    ISAAC_GYM_AVAILABLE = False
    print("⚠️  Isaac Gym未安装，请先安装Isaac Gym")


class ScrewPushingIsaacEnv:
    """基于Isaac Gym的螺丝推动环境"""
    
    def __init__(self, 
                 num_envs: int = 2048,
                 device: str = "cuda:0",
                 dt: float = 0.01,
                 substeps: int = 2):
        """
        初始化Isaac Gym环境
        
        Args:
            num_envs: 并行环境数量
            device: 计算设备
            dt: 仿真时间步长
            substeps: 物理仿真子步数
        """
        if not ISAAC_GYM_AVAILABLE:
            raise ImportError("Isaac Gym未安装")
            
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = dt
        self.substeps = substeps
        
        # 环境参数
        self.max_episode_steps = 500
        self.min_screw_distance = 0.05
        
        # 动作空间: 末端执行器位置和姿态增量
        self.action_dim = 6  # [dx, dy, dz, droll, dpitch, dyaw]
        self.action_scale = torch.tensor([0.01, 0.01, 0.01, 0.05, 0.05, 0.05], 
                                       device=self.device)
        
        # 观察空间: 末端位置(3) + 末端姿态(3) + 螺丝位置(9) + 螺丝间距(3) + 最小间距(1)
        self.obs_dim = 19
        
        # 创建Isaac Gym仿真
        self._create_sim()
        self._create_envs()
        self._setup_tensors()
        
        # 重置环境
        self.reset()
        
    def _create_sim(self):
        """创建Isaac Gym仿真"""
        self.gym = gymapi.acquire_gym()
        
        # 创建仿真
        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.substeps = self.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # 物理引擎参数
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        
        # 创建仿真
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        
    def _create_envs(self):
        """创建多个环境实例"""
        # 环境间距
        env_spacing = 1.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        
        # 创建地面
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)
        
        # 加载机械臂资产
        asset_root = "xacro-to-urdf-to-mjcf-converter/mjcf_models/elfin15/"
        arm_asset_file = "elfin15.xml"
        
        # 加载螺丝资产
        screw_asset_file = "stl/screw.STL"
        
        # 创建资产选项
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.angular_damping = 0.0
        
        # 加载机械臂资产
        self.arm_asset = self.gym.load_asset(self.sim, asset_root, arm_asset_file, asset_options)
        
        # 获取机械臂关节信息
        self.num_arm_dofs = self.gym.get_asset_dof_count(self.arm_asset)
        self.arm_dof_props = self.gym.get_asset_dof_properties(self.arm_asset)
        
        # 设置关节限制
        self.arm_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
        self.arm_dof_props['stiffness'].fill(100.0)
        self.arm_dof_props['damping'].fill(2.0)
        
        # 创建环境
        self.envs = []
        self.arm_handles = []
        self.screw_handles = []
        
        for i in range(self.num_envs):
            # 创建环境
            env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
            self.envs.append(env)
            
            # 添加机械臂
            initial_pose = gymapi.Transform()
            initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            arm_handle = self.gym.create_actor(env, self.arm_asset, initial_pose, 
                                             "arm", i, 1)
            self.arm_handles.append(arm_handle)
            
            # 设置机械臂属性
            self.gym.set_actor_dof_properties(env, arm_handle, self.arm_dof_props)
            
            # 添加螺丝（简化版本，使用立方体代替）
            self._add_screws(env, i)
            
    def _add_screws(self, env, env_id):
        """为环境添加螺丝"""
        # 创建螺丝资产（简化版本）
        screw_asset_options = gymapi.AssetOptions()
        screw_asset_options.density = 7850.0  # 钢的密度
        screw_asset_options.fix_base_link = False
        
        # 使用立方体作为螺丝（实际应用中应该使用真实的螺丝模型）
        screw_asset = self.gym.create_box(self.sim, 0.01, 0.01, 0.095, screw_asset_options)
        
        # 添加3个螺丝
        screw_handles = []
        for j in range(3):
            # 随机位置
            x = np.random.uniform(-0.76, -0.74)
            y = np.random.uniform(-0.03, 0.03)
            z = 0.25
            
            screw_pose = gymapi.Transform()
            screw_pose.p = gymapi.Vec3(x, y, z)
            
            screw_handle = self.gym.create_actor(env, screw_asset, screw_pose, 
                                               f"screw_{j}", env_id, 2)
            screw_handles.append(screw_handle)
            
        self.screw_handles.append(screw_handles)
        
    def _setup_tensors(self):
        """设置GPU张量"""
        # 获取状态张量
        self.gym.prepare_sim(self.sim)
        
        # 机械臂状态张量
        self.arm_dof_state = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim))
        self.arm_dof_pos = self.arm_dof_state.view(self.num_envs, self.num_arm_dofs, 2)[..., 0]
        self.arm_dof_vel = self.arm_dof_state.view(self.num_envs, self.num_arm_dofs, 2)[..., 1]
        
        # 刚体状态张量
        self.rb_states = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim))
        
        # 末端执行器状态
        self.ee_states = self.rb_states.view(self.num_envs, -1, 13)[:, -1, :7]  # 假设最后一个刚体是末端
        
        # 螺丝状态
        self.screw_states = []
        for i in range(self.num_envs):
            env_screw_states = []
            for j in range(3):
                screw_idx = self.gym.find_actor_index(self.envs[i], 
                                                    self.screw_handles[i][j], 
                                                    gymapi.DOMAIN_SIM)
                env_screw_states.append(self.rb_states[i, screw_idx, :7])
            self.screw_states.append(torch.stack(env_screw_states))
        
    def reset(self, env_ids=None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # 重置机械臂到初始位置
        initial_joints = torch.zeros(len(env_ids), self.num_arm_dofs, device=self.device)
        initial_joints[:, 0] = 0.0
        initial_joints[:, 1] = 0.0
        initial_joints[:, 2] = -1.57
        initial_joints[:, 3] = 0.0
        initial_joints[:, 4] = -1.57
        initial_joints[:, 5] = 0.0
        
        self.arm_dof_pos[env_ids] = initial_joints
        self.arm_dof_vel[env_ids] = 0.0
        
        # 重置螺丝位置
        self._reset_screws(env_ids)
        
        # 同步状态
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.arm_dof_state))
        
        # 获取初始观察
        obs = self._get_observations()
        return obs
        
    def _reset_screws(self, env_ids):
        """重置螺丝位置"""
        for env_id in env_ids:
            # 生成新的螺丝位置
            positions = self._generate_screw_positions()
            
            for i, pos in enumerate(positions):
                screw_handle = self.screw_handles[env_id][i]
                self.gym.set_actor_dof_position_targets(self.envs[env_id], screw_handle, pos)
                
    def _generate_screw_positions(self):
        """生成螺丝位置"""
        positions = []
        for i in range(3):
            x = np.random.uniform(-0.76, -0.74)
            y = np.random.uniform(-0.03, 0.03)
            z = 0.25
            positions.append([x, y, z])
        return positions
        
    def step(self, actions):
        """执行动作"""
        # 应用动作
        actions = actions.to(self.device)
        scaled_actions = actions * self.action_scale
        
        # 获取当前末端位置
        current_ee_pos = self.ee_states[:, :3]
        current_ee_rot = self.ee_states[:, 3:7]
        
        # 计算目标位置
        target_ee_pos = current_ee_pos + scaled_actions[:, :3]
        target_ee_rot = current_ee_rot + scaled_actions[:, 3:6]  # 简化旋转处理
        
        # 应用逆运动学（简化版本）
        target_joints = self._solve_ik_batch(target_ee_pos, target_ee_rot)
        
        # 设置关节目标
        self.arm_dof_pos = target_joints
        
        # 执行仿真
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.sync_frame_time(self.sim)
        
        # 更新张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # 计算奖励
        rewards = self._compute_rewards()
        
        # 检查终止条件
        dones = self._check_termination()
        
        # 获取新观察
        obs = self._get_observations()
        
        return obs, rewards, dones, {}
        
    def _solve_ik_batch(self, target_pos, target_rot):
        """批量求解逆运动学（简化版本）"""
        # 这里应该实现真正的逆运动学
        # 简化版本：直接设置关节角度
        current_joints = self.arm_dof_pos.clone()
        
        # 简单的关节增量控制
        joint_deltas = torch.zeros_like(current_joints)
        joint_deltas[:, 0] = (target_pos[:, 0] - self.ee_states[:, 0]) * 10.0
        joint_deltas[:, 1] = (target_pos[:, 1] - self.ee_states[:, 1]) * 10.0
        joint_deltas[:, 2] = (target_pos[:, 2] - self.ee_states[:, 2]) * 10.0
        
        return current_joints + joint_deltas
        
    def _compute_rewards(self):
        """计算奖励"""
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 计算螺丝间距
        for i in range(self.num_envs):
            screw_positions = self.screw_states[i][:, :3]  # 只取位置
            
            # 计算最小间距
            min_distance = float('inf')
            for j in range(3):
                for k in range(j+1, 3):
                    distance = torch.norm(screw_positions[j] - screw_positions[k])
                    min_distance = min(min_distance, distance.item())
            
            # 奖励计算
            if min_distance >= self.min_screw_distance:
                rewards[i] = 50.0 + (min_distance - self.min_screw_distance) * 20.0
            else:
                rewards[i] = -100.0 * (self.min_screw_distance - min_distance)
                
        return rewards
        
    def _check_termination(self):
        """检查终止条件"""
        # 简化版本：所有环境都继续
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
    def _get_observations(self):
        """获取观察"""
        obs = torch.zeros(self.num_envs, self.obs_dim, device=self.device)
        
        for i in range(self.num_envs):
            # 末端位置和姿态
            obs[i, :3] = self.ee_states[i, :3]  # 位置
            obs[i, 3:6] = self.ee_states[i, 3:6]  # 姿态（简化）
            
            # 螺丝位置
            screw_positions = self.screw_states[i][:, :3].flatten()
            obs[i, 6:15] = screw_positions
            
            # 螺丝间距
            distances = []
            for j in range(3):
                for k in range(j+1, 3):
                    distance = torch.norm(self.screw_states[i][j, :3] - self.screw_states[i][k, :3])
                    distances.append(distance)
            
            obs[i, 15:18] = torch.tensor(distances, device=self.device)
            
            # 最小间距
            obs[i, 18] = min(distances)
            
        return obs


def test_isaac_gym_performance():
    """测试Isaac Gym性能"""
    if not ISAAC_GYM_AVAILABLE:
        print("❌ Isaac Gym未安装，无法测试")
        return
        
    print("🧪 测试Isaac Gym性能...")
    
    # 创建环境
    env = ScrewPushingIsaacEnv(num_envs=1024)  # 使用1024个环境进行测试
    
    # 测试训练速度
    num_steps = 1000
    start_time = time.time()
    
    obs = env.reset()
    for step in range(num_steps):
        # 随机动作
        actions = torch.randn(env.num_envs, env.action_dim, device=env.device)
        obs, rewards, dones, info = env.step(actions)
        
        if step % 100 == 0:
            print(f"步骤 {step}: 平均奖励 = {rewards.mean().item():.2f}")
    
    end_time = time.time()
    steps_per_second = num_steps / (end_time - start_time)
    
    print(f"✅ Isaac Gym性能测试完成:")
    print(f"   环境数量: {env.num_envs}")
    print(f"   训练速度: {steps_per_second:.1f} 步/秒")
    print(f"   总时间: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    test_isaac_gym_performance() 