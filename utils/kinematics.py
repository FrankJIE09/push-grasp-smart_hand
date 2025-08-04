"""
运动学相关工具函数
包含正运动学、逆运动学、变换矩阵计算等功能
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import ikpy.chain
import ikpy.link


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


def solve_inverse_kinematics(chain, target_pos, target_euler, current_joints):
    """求解逆运动学"""
    if chain is None:
        return current_joints

    target_transform = np.eye(4)
    target_transform[:3, :3] = Rotation.from_euler('xyz', target_euler).as_matrix()
    target_transform[:3, 3] = target_pos

    try:
        # 使用当前关节角度作为初始猜测，确保在关节范围内
        initial_position = np.zeros(8)
        initial_position[1:7] = current_joints
        
        # 限制初始猜测在关节范围内
        for i in range(1, 7):
            if i < len(chain.links):
                bounds = chain.links[i].bounds
                if bounds is not None:
                    initial_position[i] = np.clip(initial_position[i], bounds[0], bounds[1])

        joint_angles = chain.inverse_kinematics(
            target_position=target_transform[:3, 3],
            target_orientation=target_transform[:3, :3],
            initial_position=initial_position,
            max_iter=100,  # 减少迭代次数
            orientation_mode="all"
        )

        # 限制结果在关节范围内
        result = joint_angles[1:7]
        for i in range(len(result)):
            if i < len(chain.links) - 1:  # -1因为第一个是基座
                bounds = chain.links[i + 1].bounds
                if bounds is not None:
                    result[i] = np.clip(result[i], bounds[0], bounds[1])

        return result

    except Exception as e:
        # 静默处理错误，返回当前关节角度
        return current_joints


def get_end_effector_pose(chain, joint_positions):
    """获取末端执行器位置和姿态"""
    if chain is not None:
        # 使用ikpy计算
        ikpy_joints = np.zeros(8)
        ikpy_joints[1:7] = joint_positions

        ee_transform = chain.forward_kinematics(ikpy_joints)
        ee_pos = ee_transform[:3, 3]
        ee_euler = Rotation.from_matrix(ee_transform[:3, :3]).as_euler('xyz')

        return ee_pos, ee_euler
    else:
        return np.zeros(3), np.zeros(3) 