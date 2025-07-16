"""
运动学链构建模块

这个模块负责从MuJoCo XML文件解析机器人模型并创建ikpy运动学链。
主要功能包括：
1. 解析MuJoCo XML文件结构
2. 提取关节和连杆信息
3. 创建ikpy运动学链对象
4. 处理双臂机器人的运动学模型

作者: frank
日期: 2024年6月19日
"""

import ikpy.chain
import ikpy.link
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from .transform_utils import get_transformation


def find_body_by_name(root, name):
    """递归查找任意层级body"""
    if root.tag == 'body' and root.get('name') == name:
        return root
    for child in root:
        if child.tag == 'body':
            result = find_body_by_name(child, name)
            if result is not None:
                return result
    return None

def create_chain_from_mjcf(xml_file, base_body_name, prefix=""):
    """
    解析MuJoCo XML文件，为指定的机械臂创建ikpy运动学链。
    支持prefix（如attach模型时的前缀）。
    """
    import ikpy.chain
    import ikpy.link
    import numpy as np
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from .transform_utils import get_transformation

    tree = ET.parse(xml_file)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    base_element = find_body_by_name(worldbody, base_body_name)
    if base_element is None:
        raise Exception(f"找不到基座body: {base_body_name}")
    base_transform = get_transformation(base_element)

    links = [ikpy.link.URDFLink(
        name="base",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    )]
    active_links_mask = [False]

    current_element = base_element
    for i in range(1, 7):
        link_name = f"{prefix}elfin_link{i}"
        current_element = find_body_by_name(base_element, link_name)
        if current_element is None:
            raise Exception(f"找不到机械臂连杆body: {link_name}，请检查prefix和模型结构")
        joint_element = current_element.find('joint')
        joint_name = joint_element.get('name')
        joint_axis = np.fromstring(joint_element.get('axis'), sep=' ') if joint_element.get('axis') else np.array([0,0,1])
        joint_range = joint_element.get('range')
        bounds = tuple(map(float, joint_range.split())) if joint_range else (None, None)
        link_transform = get_transformation(current_element)
        translation = link_transform[:3, 3]
        orientation_matrix = link_transform[:3, :3]
        orientation_rpy = Rotation.from_matrix(orientation_matrix).as_euler('xyz')
        link = ikpy.link.URDFLink(
            name=joint_name,
            origin_translation=translation,
            origin_orientation=orientation_rpy,
            rotation=joint_axis,
            bounds=bounds
        )
        links.append(link)
        active_links_mask.append(True)

    # 末端执行器body可选
    ee_body = None
    if current_element is not None:
        for body in current_element.iter('body'):
            if '_end_effector' in body.get('name', ''):
                ee_body = body
                break
    if ee_body is not None:
        ee_transform = get_transformation(ee_body)
        ee_orientation_matrix = ee_transform[:3, :3]
        ee_orientation_rpy = Rotation.from_matrix(ee_orientation_matrix).as_euler('xyz')
        ee_link = ikpy.link.URDFLink(
            name=ee_body.get('name'),
            origin_translation=ee_transform[:3, 3],
            origin_orientation=ee_orientation_rpy,
            rotation=[0, 0, 0]
        )
        links.append(ee_link)
        active_links_mask.append(False)
    chain = ikpy.chain.Chain(links, active_links_mask=active_links_mask)
    return chain, base_transform


def get_kinematics(xml_file_path):
    """
    为双臂机器人创建运动学模型。
    
    这个函数为左右两个机械臂分别创建运动学链，并返回它们的基座变换矩阵。
    
    Args:
        xml_file_path (str): MuJoCo XML文件的路径
    
    Returns:
        tuple: (左臂链, 右臂链, 左臂基座变换, 右臂基座变换)
    """
    left_chain, left_base_transform = create_chain_from_mjcf(xml_file_path, 'left_robot_base')
    right_chain, right_base_transform = create_chain_from_mjcf(xml_file_path, 'right_robot_base')
    return left_chain, right_chain, left_base_transform, right_base_transform 