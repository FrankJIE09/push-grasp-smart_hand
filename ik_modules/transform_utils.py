"""
变换工具函数模块

这个模块提供了机器人学中常用的坐标变换工具函数，包括：
1. 欧拉角与旋转矩阵的转换
2. 四元数与旋转矩阵的转换
3. 位姿向量与变换矩阵的转换
4. MuJoCo XML解析工具

作者: frank
日期: 2024年6月19日
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation


def rpy_to_matrix(rpy):
    """
    将RPY欧拉角转换为旋转矩阵。
    
    数学公式：
    R = Rz(ψ) @ Ry(θ) @ Rx(φ)
    其中：
    - φ (roll): 绕X轴旋转
    - θ (pitch): 绕Y轴旋转  
    - ψ (yaw): 绕Z轴旋转
    
    Args:
        rpy (list or np.ndarray): 包含roll, pitch, yaw的欧拉角 [rx, ry, rz]
    
    Returns:
        np.ndarray: 3x3旋转矩阵
    """
    return Rotation.from_euler('xyz', rpy, degrees=False).as_matrix()


def quat_to_matrix(quat):
    """
    将四元数转换为旋转矩阵。
    
    数学公式：
    q = [x, y, z, w] = [q1, q2, q3, q0]
    R = [1-2q2²-2q3²    2(q1q2-q0q3)    2(q1q3+q0q2)]
        [2(q1q2+q0q3)   1-2q1²-2q3²     2(q2q3-q0q1)]
        [2(q1q3-q0q2)   2(q2q3+q0q1)    1-2q1²-2q2²]
    
    Args:
        quat (list or np.ndarray): 四元数 [x, y, z, w]
    
    Returns:
        np.ndarray: 3x3旋转矩阵
    """
    return Rotation.from_quat(quat).as_matrix()


def pose_to_transformation_matrix(pose):
    """
    将位姿向量转换为4x4变换矩阵。
    
    数学公式：
    pose = [x, y, z, qx, qy, qz, qw]
    T = [R    p] 其中 R = quat_to_matrix([qx, qy, qz, qw])
        [0    1]      p = [x, y, z]ᵀ
    
    Args:
        pose (list or np.ndarray): 位姿向量 [x, y, z, qx, qy, qz, qw]
    
    Returns:
        np.ndarray: 4x4齐次变换矩阵
    """
    t = np.eye(4)
    t[:3, 3] = pose[:3]  # 设置平移部分 p = [x, y, z]ᵀ
    t[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()  # 设置旋转部分 R
    return t


def transformation_matrix_to_pose(t):
    """
    将4x4变换矩阵转换为位姿向量。
    
    数学公式：
    T = [R    p] → pose = [p_x, p_y, p_z, q_x, q_y, q_z, q_w]
        [0    1]
    其中 q = matrix_to_quat(R)
    
    Args:
        t (np.ndarray): 4x4齐次变换矩阵
    
    Returns:
        np.ndarray: 位姿向量 [x, y, z, qx, qy, qz, qw]
    """
    pose = np.zeros(7)
    pose[:3] = t[:3, 3]  # 提取平移部分 p
    pose[3:] = Rotation.from_matrix(t[:3, :3]).as_quat()  # 提取旋转部分（四元数）
    return pose


def get_transformation(body_element):
    """
    从MuJoCo的body XML元素中提取其相对于父坐标系的4x4变换矩阵。
    
    这个函数解析XML中的位置和姿态信息，支持以下属性：
    - pos: 3D位置向量
    - quat: 四元数姿态 (MuJoCo格式: w,x,y,z)
    - euler: 欧拉角姿态 (假设xyz顺序)
    
    数学公式：
    T = [R    p] 其中 p = pos, R = quat_to_matrix(quat) 或 euler_to_matrix(euler)
        [0    1]
    
    Args:
        body_element (xml.etree.ElementTree.Element): MuJoCo XML中的body元素
    
    Returns:
        np.ndarray: 4x4齐次变换矩阵
    """
    # 解析位置信息
    pos = body_element.get('pos')
    pos = np.fromstring(pos, sep=' ') if pos else np.zeros(3)

    # 解析姿态信息 (优先使用四元数)
    quat = body_element.get('quat')
    euler = body_element.get('euler')

    rot_matrix = np.eye(3)
    if quat:
        # MuJoCo的四元数顺序是 (w, x, y, z), 而Scipy需要 (x, y, z, w)
        # 转换公式: [w, x, y, z] → [x, y, z, w]
        q = np.fromstring(quat, sep=' ')
        rot_matrix = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    elif euler:
        # 假设欧拉角顺序为 'xyz'
        # 旋转矩阵: R = Rz(ψ) @ Ry(θ) @ Rx(φ)
        e = np.fromstring(euler, sep=' ')
        rot_matrix = Rotation.from_euler('XYZ', e).as_matrix()

    # 构建4x4变换矩阵
    # 数学公式: T = [R    p]
    #                [0    1]
    transformation = np.eye(4)
    transformation[:3, :3] = rot_matrix  # 旋转部分 R
    transformation[:3, 3] = pos  # 平移部分 p
    return transformation 