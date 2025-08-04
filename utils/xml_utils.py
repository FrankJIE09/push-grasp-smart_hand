"""
XML处理相关工具函数
包含XML文件修改、螺丝添加、keyframe移除等功能
"""

import re
import random
import numpy as np


def remove_keyframe_section(xml_content):
    """移除XML中的keyframe段"""
    return re.sub(r'<keyframe>.*?</keyframe>', '', xml_content, flags=re.DOTALL)


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


def generate_challenge_screw_positions(num_screws=3, target_distance=0.05):
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
            roll = random.uniform(0, 2 * np.pi)
            pitch = random.uniform(0, 2 * np.pi)
            yaw = random.uniform(0, 2 * np.pi)

            positions.append((x, y, z, roll, pitch, yaw))

    return positions


def add_challenge_screws_to_xml(xml_file, num_screws=3):
    """添加用于强化学习训练的螺丝到XML文件（故意设置紧密间距）"""
    mass = 0.5
    L = 0.095
    r = 0.006

    Ixx = Iyy = (1 / 12) * mass * (3 * r ** 2 + L ** 2)
    Izz = (1 / 2) * mass * r ** 2
    com_offset = [0, 0, 0.01]

    with open(xml_file, 'r') as f:
        xml_content = f.read()

    # 移除keyframe定义
    xml_content = remove_keyframe_section(xml_content)

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
    screw_positions = generate_challenge_screw_positions(num_screws)
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