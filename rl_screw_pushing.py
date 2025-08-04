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
from stable_baselines3.common.callbacks import BaseCallback
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from tqdm import tqdm
import yaml


def load_config(config_file="config.yaml"):
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# 从原代码导入必要的函数
def get_transformation(body_element):
    """从MuJoCo的body XML元素中提取4x4变换矩阵"""
    pos = body_element.get('pos')
    pos = np.fromstring(pos, sep=' ') if pos else np.zeros(3)

    quat = body_element.get('quat')
    euler = body_element.get('euler')

    rot_matrix = np.eye(3)
    if quat is not None and quat.strip():
        q = np.fromstring(quat, sep=' ')
        rot_matrix = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    elif euler is not None and euler.strip():
        robot_e = np.fromstring(euler, sep=' ')
        rot_matrix = Rotation.from_euler('XYZ', robot_e).as_matrix()

    transformation = np.eye(4)
    transformation[:3, :3] = rot_matrix
    transformation[:3, 3] = pos
    return transformation


def create_chain_from_mjcf(xml_file):
    """从 MuJoCo XML 文件创建 ikpy 运动学链"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    links = [ikpy.link.OriginLink()]
    active_links_mask = [False]

    body_elements = []

    def find_elfin_bodies(element, bodies_list):
        for body in element.findall('body'):
            name = body.get('name', '')
            if name.startswith('elfin_link'):
                bodies_list.append(body)
            find_elfin_bodies(body, bodies_list)

    find_elfin_bodies(worldbody, body_elements)
    body_elements.sort(key=lambda x: x.get('name'))

    for body in body_elements:
        body_name = body.get('name')
        joint = body.find('joint')

        if joint is not None:
            joint_name = joint.get('name')
            axis_str = joint.get('axis', '0 0 1')
            axis = np.fromstring(axis_str, sep=' ')
            range_str = joint.get('range')
            bounds = tuple(map(float, range_str.split())) if range_str else (-3.14159, 3.14159)

            transform = get_transformation(body)
            translation = transform[:3, 3]
            orientation = Rotation.from_matrix(transform[:3, :3]).as_euler('xyz')

            link = ikpy.link.URDFLink(
                name=joint_name,
                origin_translation=translation,
                origin_orientation=orientation,
                rotation=axis,
                bounds=bounds
            )

            links.append(link)
            active_links_mask.append(True)

    links.append(ikpy.link.URDFLink(
        name="end_effector",
        origin_translation=[0, 0, 0.171],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    ))
    active_links_mask.append(False)

    chain = ikpy.chain.Chain(links, active_links_mask=active_links_mask)
    return chain


def check_screw_spacing(positions, min_distance=0.15):
    """检查螺丝之间的间距是否满足最小距离要求"""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pos1 = np.array(positions[i][:3])  # 只考虑xyz位置
            pos2 = np.array(positions[j][:3])
            distance = np.linalg.norm(pos1 - pos2)
            if distance < min_distance:
                return False
    return True


def generate_challenge_screw_positions(num_screws=3, target_distance=0.05, config=None):
    """生成用于强化学习训练的螺丝位置（故意设置较近的间距，需要智能体推开）"""
    positions = []

    # 设计初始螺丝布局：故意让间距小于要求，给智能体学习的机会
    if num_screws == 3:
        # 三个螺丝呈三角形紧密排列
        center_x, center_y = -0.75, 0.0

        positions = [
            (center_x - target_distance, center_y - target_distance / 2, 0.25, 0, 0, 0),
            (center_x + target_distance, center_y - target_distance / 2, 0.25, 0, 0, 0),
            (center_x, center_y + target_distance, 0.25, 0, 0, 0)
        ]
    else:
        # 对于其他螺丝数量，生成紧密的随机布局
        for i in range(num_screws):
            x = random.uniform(-0.76, -0.74)  # 小范围内随机
            y = random.uniform(-0.03, 0.03)  # 小范围内随机
            z = 0.25

            # 随机旋转
            roll = random.uniform(0, 2 * 3.141592653589793)
            pitch = random.uniform(0, 2 * 3.141592653589793)
            yaw = random.uniform(0, 2 * 3.141592653589793)

            positions.append((x, y, z, roll, pitch, yaw))

    return positions


def add_challenge_screws_to_xml(xml_file, num_screws=3, config=None):
    """添加用于强化学习训练的螺丝到XML文件（故意设置紧密间距）"""
    if config is None:
        config = load_config()

    screw_config = config['screw']
    mass = screw_config['mass']
    L = screw_config['length']
    r = screw_config['radius']

    Ixx = Iyy = (1 / 12) * mass * (3 * r ** 2 + L ** 2)
    Izz = (1 / 2) * mass * r ** 2
    com_offset = [0, 0, 0.01]

    with open(xml_file, 'r') as f:
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

    # 生成用于强化学习训练的螺丝位置（故意紧密排列）
    target_distance = screw_config['target_distance']
    screw_positions = generate_challenge_screw_positions(num_screws, target_distance, config)
    all_screw_bodies = ""

    for i, (x, y, z, roll, pitch, yaw) in enumerate(screw_positions):
        screw_body = f'''
    <!-- 训练螺丝 #{i + 1} -->
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
    temp_xml = "temp_rl_scene.xml"
    with open(temp_xml, 'w') as f:
        f.write(xml_content)
    return temp_xml, screw_positions


class ScrewPushingEnv(gym.Env):
    """螺丝推动强化学习环境"""

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
        # 只控制xy
        position_step = action_config['position_step']
        self.action_space = spaces.Box(
            low=np.array([-position_step, -position_step]),
            high=np.array([position_step, position_step]),
            dtype=np.float32
        )

        # 观察空间：末端位置(3) + 末端姿态(3) + 所有螺丝位置(3*3) + 螺丝间距(3) + 最小间距信息(1)
        # 总共：3+3+9+3+1 = 19维
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

    def _setup_environment(self):
        """设置MuJoCo环境"""
        # 创建包含训练用螺丝的临时XML（故意设置紧密间距）
        self.temp_xml, self.screw_positions = add_challenge_screws_to_xml(
            self.xml_file, self.num_screws, self.config
        )

        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(self.temp_xml)
        self.data = mujoco.MjData(self.model)

        # 设置重力
        self.model.opt.gravity[:] = [0, 0, -9.8]

        # 创建运动学链
        try:
            self.arm_chain = create_chain_from_mjcf(
                "xacro-to-urdf-to-mjcf-converter/mjcf_models/elfin15/elfin15.xml"
            )
        except Exception as e:
            print(f"❌ 创建运动学链失败: {e}")
            # 使用简单的逆运动学方法作为备选
            self.arm_chain = None

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

    def _apply_home_position(self):
        """应用home位置"""
        try:
            # 设置机械臂的home位置（基于机械臂的关节限制）
            home_joints = [-0.000, 0.67, -2.21, -0.000, -0.27, 0.000]  # 机械臂home位置

            # 设置灵巧手的home位置（基于灵巧手的执行器范围）
            home_hand_ctrl = [0.3, -1.27, -1.27, -1.27, -0.5, 1.2]  # 灵巧手home控制信号

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
            # 使用默认位置
            default_joints = [0, -0.5, 1.0, 0, -0.5, 0]
            for i, joint_id in enumerate(self.arm_joint_ids):
                if i < len(default_joints):
                    self.data.qpos[joint_id] = default_joints[i]
                    self.data.ctrl[self.arm_actuator_ids[i]] = default_joints[i]

    def _get_end_effector_pose(self):
        """获取末端执行器位置和姿态"""
        if self.arm_chain is not None:
            # 使用ikpy计算
            current_joints = self.data.qpos[self.arm_joint_ids]
            ikpy_joints = np.zeros(8)
            ikpy_joints[1:7] = current_joints

            ee_transform = self.arm_chain.forward_kinematics(ikpy_joints)
            ee_pos = ee_transform[:3, 3]
            ee_euler = Rotation.from_matrix(ee_transform[:3, :3]).as_euler('xyz')

            return ee_pos, ee_euler
        else:
            # 备选方法：直接从MuJoCo获取
            try:
                # 这里需要根据实际的末端执行器body名称调整
                ee_body_id = self.model.body('elfin_link6').id  # 假设link6是末端
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

        # 填充距离信息到固定长度（3个距离：1-2, 1-3, 2-3）
        while len(screw_distances) < 3:
            screw_distances.append(0.0)
        screw_distances = screw_distances[:3]

        # 最小间距不足时用实际值，否则用最小间距
        min_distance_normalized = min_distance if min_distance != float('inf') else 0.0

        # 组合观察
        obs = np.concatenate([
            ee_pos,  # 3 - 末端位置
            ee_euler,  # 3 - 末端姿态
            padded_screw_positions,  # 9 - 所有螺丝位置(3x3)
            screw_distances,  # 3 - 螺丝间距
            [min_distance_normalized]  # 1 - 最小间距
        ])

        return np.array(obs, dtype=np.float32)

    def _solve_inverse_kinematics(self, target_pos, target_euler, current_joints):
        """使用scipy.optimize.minimize求解逆运动学，z位置和姿态作为约束"""
        if self.arm_chain is None:
            return current_joints

        try:
            from scipy.optimize import minimize

            # 定义目标函数：最小化末端位置误差
            def objective_function(joint_angles):
                # 构建完整的关节角度向量（包含base和end effector）
                full_joints = np.zeros(8)
                full_joints[1:7] = joint_angles

                # 计算正运动学
                fk_transform = self.arm_chain.forward_kinematics(full_joints)
                current_pos = fk_transform[:3, 3]

                # 计算位置误差（只考虑xy平面）
                pos_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
                return pos_error

            # 定义约束函数：z位置和姿态约束
            def z_constraint(joint_angles):
                full_joints = np.zeros(8)
                full_joints[1:7] = joint_angles
                fk_transform = self.arm_chain.forward_kinematics(full_joints)
                current_z = fk_transform[2, 3]
                return current_z - self.fixed_z  # 约束：current_z = fixed_z

            def orientation_constraint(joint_angles):
                full_joints = np.zeros(8)
                full_joints[1:7] = joint_angles
                fk_transform = self.arm_chain.forward_kinematics(full_joints)
                current_rot_matrix = fk_transform[:3, :3]

                # 只约束rx和ry，不约束rz
                # 提取当前旋转矩阵的rx和ry分量
                current_rx = np.arctan2(current_rot_matrix[2, 1], current_rot_matrix[2, 2])
                current_ry = np.arctan2(-current_rot_matrix[2, 0],
                           np.sqrt(current_rot_matrix[2, 1]**2 + current_rot_matrix[2, 2]**2))

                # 提取固定旋转矩阵的rx和ry分量
                fixed_rx = np.arctan2(self.fixed_rot_matrix[2, 1], self.fixed_rot_matrix[2, 2])
                fixed_ry = np.arctan2(-self.fixed_rot_matrix[2, 0],
                          np.sqrt(self.fixed_rot_matrix[2, 1]**2 + self.fixed_rot_matrix[2, 2]**2))

                # 计算rx和ry的误差
                rx_error = current_rx - fixed_rx
                ry_error = current_ry - fixed_ry

                # 处理角度缠绕问题
                rx_error = np.arctan2(np.sin(rx_error), np.cos(rx_error))
                ry_error = np.arctan2(np.sin(ry_error), np.cos(ry_error))

                # 返回rx和ry误差的平方和
                return rx_error**2 + ry_error**2

            # 设置优化问题
            initial_guess = current_joints.copy()

            # 定义约束
            constraints = [
                {'type': 'eq', 'fun': z_constraint},  # z位置约束
                {'type': 'eq', 'fun': orientation_constraint}  # 姿态约束
            ]

            # 定义关节角度边界
            bounds = []
            for i in range(6):
                # 使用机械臂关节限制
                if i == 0:  # joint1
                    bounds.append((-3.141592653589793, 3.141592653589793))
                elif i == 1:  # joint2
                    bounds.append((-3.141592653589793 / 2, 3.141592653589793 / 2))
                elif i == 2:  # joint3
                    bounds.append((-3.141592653589793, 3.141592653589793))
                elif i == 3:  # joint4
                    bounds.append((-3.141592653589793, 3.141592653589793))
                elif i == 4:  # joint5
                    bounds.append((-3.141592653589793 / 2, 3.141592653589793 / 2))
                elif i == 5:  # joint6
                    bounds.append((-3.141592653589793, 3.141592653589793))

            # 执行优化
            result = minimize(
                objective_function,
                initial_guess,
                method='SLSQP',  # 序列最小二乘规划，适合带约束的优化
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-3}
            )

            if result.success:
                return result.x
            else:
                print(f"❌ 逆运动学优化失败: {result.message}")
                return current_joints

        except Exception as e:
            print(f"❌ 逆运动学求解失败: {e}")
            return current_joints

    def _get_index_finger_pos(self):
        """获取食指头的世界坐标"""
        try:
            # 假设食指头body名称为'index_finger_tip'，如有不同请修改
            idx_body_id = self.model.body('e_hhand_if_distal_link').id
            pos = self.data.xpos[idx_body_id].copy()
            return pos
        except Exception as e:
            print(f"❌ 获取食指头位置失败: {e}")
            return np.zeros(3)

    def _calculate_reward(self):
        """计算奖励"""
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
        # 新增：食指靠近奖励系数
        index_finger_coef = reward_config.get('index_finger_coef', 1.0)

        # 基础奖励：基于最小螺丝间距 - 改进奖励设计
        if min_distance >= self.min_screw_distance:
            # 当达到目标间距时，给予正奖励
            distance_reward = distance_base_reward + (min_distance - self.min_screw_distance) * distance_coefficient
        else:
            # 当未达到目标时，给予较小的负奖励，避免过度惩罚
            distance_reward = -distance_penalty_coefficient * (self.min_screw_distance - min_distance) * 0.1

        # 任务完成奖励
        success_reward = 0.0
        all_distances_satisfied = all(d >= self.min_screw_distance for d in screw_distances)
        if all_distances_satisfied:
            success_reward = success_reward_value

        # 改进的接近奖励：鼓励智能体接近需要分开的螺丝对
        closest_screw_pair_center = None
        closest_pair_distance = float('inf')

        for i, distance in enumerate(screw_distances):
            if distance < self.min_screw_distance:  # 只关注需要分开的螺丝
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

        # 接近奖励：使用更温和的奖励函数
        proximity_reward = 0.0
        if closest_screw_pair_center is not None:
            ee_to_pair = np.linalg.norm(ee_pos - closest_screw_pair_center)
            # 使用指数衰减，避免过度惩罚
            proximity_reward = -np.exp(-ee_to_pair) * proximity_coefficient

        # 改进的食指奖励：使用更合理的奖励函数
        index_finger_pos = self._get_index_finger_pos()
        if len(screw_positions) > 0:
            screws_center = np.mean(screw_positions, axis=0)
            finger_dist = np.linalg.norm(index_finger_pos[:2] - screws_center[:2])
            # 使用线性惩罚而不是平方惩罚
            index_finger_reward = -finger_dist * index_finger_coef
        else:
            index_finger_reward = 0.0

        # 时间惩罚 - 大幅降低
        time_penalty = -time_penalty_value
        # 碰撞检测（简化版本）
        collision_penalty = 0.0

        # 添加进度奖励：鼓励螺丝间距的改善
        progress_reward = 0.0
        if hasattr(self, 'last_min_distance'):
            if min_distance > self.last_min_distance:
                progress_reward = 5.0  # 奖励改善
            elif min_distance < self.last_min_distance:
                progress_reward = -1.0  # 轻微惩罚恶化
        self.last_min_distance = min_distance

        total_reward = distance_reward + success_reward + proximity_reward + index_finger_reward + time_penalty + collision_penalty + progress_reward

        return total_reward, all_distances_satisfied

    def step(self, action):
        """执行一步动作"""
        self.episode_steps += 1

        # 获取当前末端执行器位置
        ee_pos, ee_euler = self._get_end_effector_pose()

        # 处理action：确保绝对值不小于0.05
        processed_action = action

        # 只控制xy，z和姿态强制为home
        target_pos = ee_pos.copy()
        # print(f"原始action: {action}, 处理后action: {processed_action}")
        target_pos[0] += processed_action[0]
        target_pos[1] += processed_action[1]
        target_pos[2] = self.fixed_z  # 强制z为home高度
        target_euler = self.fixed_euler  # 姿态强制为home

        # 求解逆运动学
        current_joints = self.data.qpos[self.arm_joint_ids]
        target_joints = self._solve_inverse_kinematics(target_pos, target_euler, current_joints)

        # 应用控制信号
        for i, actuator_id in enumerate(self.arm_actuator_ids):
            if i < len(target_joints):
                self.data.ctrl[actuator_id] = target_joints[i]

        # 执行物理仿真步
        for _ in range(50):  # 每个RL步骤执行多个物理步骤
            mujoco.mj_step(self.model, self.data)

        # 计算奖励
        reward, success = self._calculate_reward()

        # 检查终止条件
        terminated = success  # 任务成功完成
        truncated = self.episode_steps >= self.max_episode_steps  # 达到最大步数

        # 获取新观察
        obs = self._get_observation()

        # 信息字典
        info = {
            'success': success,
            'episode_steps': self.episode_steps,
            'min_screw_distance': self._get_min_screw_distance()
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

        # 保存固定的旋转矩阵，避免欧拉角突变问题
        self.fixed_rot_matrix = Rotation.from_euler('xyz', self.fixed_euler).as_matrix()

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
        """渲染环境（暂时留空，MuJoCo viewer会单独处理）"""
        pass

    def close(self):
        """关闭环境"""
        if hasattr(self, 'temp_xml') and os.path.exists(self.temp_xml):
            os.remove(self.temp_xml)


class TrainingCallback(BaseCallback):
    """训练回调函数，用于记录训练过程"""

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

        # 按配置频率展示学习成果（减少频率以避免干扰训练）
        if self.n_calls - self.last_demo_step >= self.demo_freq and self.n_calls > 10:
            self.last_demo_step = self.n_calls
            self._demonstrate_learning()

        # 定期保存模型和检查点
        if self.n_calls % self.save_freq == 0:
            if self.checkpoints_dir:
                checkpoint_path = os.path.join(self.checkpoints_dir, f"screw_pushing_checkpoint_{self.n_calls}")
                self.model.save(checkpoint_path)
                print(f"💾 检查点已保存: {checkpoint_path}")

                # 保存训练统计信息
                if self.results_dir:
                    stats_path = os.path.join(self.results_dir, 'training_stats.json')
                    import json
                    with open(stats_path, 'w', encoding='utf-8') as f:
                        json.dump(self.training_stats, f, indent=2, ensure_ascii=False)

        return True

    def _demonstrate_learning(self):
        """展示当前学习成果"""
        print(f"🎬 第 {self.n_calls} 步演示")

        # 创建临时环境用于演示
        demo_env = ScrewPushingEnv()
        obs, info = demo_env.reset()

        # 从配置中读取演示参数
        demo_config = demo_env.config['demonstration']
        max_demo_steps = demo_config['max_demo_steps']
        demo_speed = demo_config['demo_speed']

        # 使用当前模型进行演示
        with mujoco.viewer.launch_passive(demo_env.model, demo_env.data) as viewer:
            episode_reward = 0
            episode_steps = 0

            while episode_steps < max_demo_steps:  # 限制演示步数
                # 使用当前模型预测动作
                action, _ = self.model.predict(obs, deterministic=True)

                # 执行动作
                obs, reward, terminated, truncated, info = demo_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1

                # 更新viewer
                viewer.sync()
                time.sleep(demo_speed)  # 控制播放速度

                if done:
                    break

                # 检查viewer是否关闭
                if not viewer.is_running():
                    break

            success = info.get('success', False)
            min_distance = info.get('min_screw_distance', 0)
            print(
                f"   奖励: {episode_reward:.1f}, 步数: {episode_steps}, 成功: {'✓' if success else '✗'}, 最小间距: {min_distance:.3f}m")

        demo_env.close()


def train_screw_pushing_agent(config=None):
    """训练螺丝推动智能体"""
    if config is None:
        config = load_config()

    print("🚀 开始训练螺丝推动强化学习智能体...")

    # 检测GPU可用性并选择设备
    import torch
    if torch.cuda.is_available():
        # 对于MLP策略，建议使用CPU以获得更好的性能
        # 但用户可以选择使用GPU
        device = 'cuda'  # 使用CPU以获得更好的性能
    else:
        device = 'cpu'
        print("⚠️  GPU不可用，使用CPU训练")

    # 创建环境
    env = ScrewPushingEnv(config)

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
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    evaluations_dir = os.path.join(results_dir, "evaluations")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(evaluations_dir, exist_ok=True)

    print(f"📁 创建子文件夹:")
    print(f"   - 模型文件夹: {models_dir}")
    print(f"   - 日志文件夹: {logs_dir}")
    print(f"   - 检查点文件夹: {checkpoints_dir}")
    print(f"   - 评估文件夹: {evaluations_dir}")

    # 创建PPO智能体（使用固定学习率，因为LearningRateScheduler已被移除）
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
        verbose=1,  # 关闭PPO的详细输出
        device=device,  # 使用检测到的设备
        tensorboard_log=logs_dir  # 设置tensorboard日志路径
    )

    # 创建回调函数，传入结果文件夹路径
    training_callback = TrainingCallback(
        eval_freq=callback_config['eval_freq'],
        save_freq=callback_config['save_freq'],
        demo_freq=callback_config['demo_freq'],
        results_dir=results_dir,
        models_dir=models_dir,
        checkpoints_dir=checkpoints_dir,
        evaluations_dir=evaluations_dir
    )
    
    # 使用单个回调函数（LearningRateScheduler已被移除）
    callback = training_callback

    # 开始训练
    total_timesteps = training_config['total_timesteps']
    save_path = os.path.join(models_dir, training_config['save_path'])
    print(f"🎯 开始训练，总时间步数: {total_timesteps}")
    print(f"💾 模型将保存至: {save_path}")
    start_time = time.time()

    try:
        # 直接使用PPO的learn方法，让stable-baselines3自己处理进度
        print("🔄 开始训练...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True  # 启用内置进度条
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

        import json
        with open(os.path.join(results_dir, 'training_info.json'), 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        print(f"💾 训练信息已保存至: {os.path.join(results_dir, 'training_info.json')}")

        return model, env

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return None, env

    finally:
        env.close()


def evaluate_and_demonstrate(model_path="screw_pushing_agent", num_episodes=5, results_dir=None):
    """评估和展示训练结果"""
    print("🎬 开始评估和展示训练结果...")

    # 检测GPU可用性
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️  GPU不可用，使用CPU")

    # 加载模型
    try:
        model = PPO.load(model_path)
        print(f"✅ 模型加载成功: {model_path}.zip")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return

    print("🎥 MuJoCo Viewer已启动，开始展示...")

    episode_rewards = []
    success_count = 0
    episode_results = []

    for episode in range(num_episodes):
        # 每个episode创建新环境
        env = ScrewPushingEnv()
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0

        print(f"\n📺 Episode {episode + 1}/{num_episodes}")
        print(f"🎯 任务目标: 确保所有螺丝间距≥{env.min_screw_distance:.2f}m")

        # 每个episode使用独立的viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while True:
                # 使用训练好的模型预测动作
                action, _ = model.predict(obs, deterministic=True)

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1

                # 更新viewer
                viewer.sync()
                time.sleep(0.02)  # 控制播放速度

                if done:
                    success = info.get('success', False)
                    min_distance = info.get('min_screw_distance', 0)

                    if success:
                        success_count += 1
                        print(f"🎉 Episode {episode + 1} 成功！奖励: {episode_reward:.2f}, 步数: {episode_steps}")
                    else:
                        print(f"😞 Episode {episode + 1} 失败。奖励: {episode_reward:.2f}, 步数: {episode_steps}")

                    episode_rewards.append(episode_reward)
                    episode_results.append({
                        'episode': episode + 1,
                        'reward': episode_reward,
                        'steps': episode_steps,
                        'success': success,
                        'min_screw_distance': min_distance
                    })
                    break

                # 检查viewer是否关闭
                if not viewer.is_running():
                    break

            if not viewer.is_running():
                break

        # 等待用户确认继续下一个episode
        if episode < num_episodes - 1:
            input("按回车键继续下一个episode...")

    # 统计结果
    if episode_rewards:
        avg_reward = np.mean(episode_rewards)
        success_rate = success_count / len(episode_rewards)
        print(f"\n📊 评估结果:")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   成功率: {success_rate:.2%}")
        print(f"   成功次数: {success_count}/{len(episode_rewards)}")

        # 保存评估结果
        if results_dir:
            evaluations_dir = os.path.join(results_dir, "evaluations")
            os.makedirs(evaluations_dir, exist_ok=True)

            evaluation_results = {
                'model_path': model_path,
                'num_episodes': num_episodes,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'success_count': success_count,
                'total_episodes': len(episode_rewards),
                'episode_results': episode_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            import json
            eval_path = os.path.join(evaluations_dir, f'evaluation_{int(time.time())}.json')
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            print(f"💾 评估结果已保存至: {eval_path}")

    print("🎉 演示完成！")


def main():
    """主函数"""
    print("🔧 螺丝推动强化学习系统")
    print("=" * 50)

    # 加载配置
    config = load_config()

    # 训练智能体
    model, env = train_screw_pushing_agent(config)

    if model is not None:
        print("\n" + "=" * 50)
        print("🎉 训练完成！开始评估和展示...")

        # 评估和展示
        demo_config = config['demonstration']
        results_dir = config['training']['results_dir']
        model_path = os.path.join(results_dir, "models", config['training']['save_path'])
        evaluate_and_demonstrate(model_path, num_episodes=demo_config['num_episodes'], results_dir=results_dir)
    else:
        print("❌ 训练失败，无法进行评估")


if __name__ == "__main__":
    main()
