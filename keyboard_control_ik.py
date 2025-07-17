import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import os
from pynput import keyboard
import ikpy.chain
import ikpy.link
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET

# 机械臂和灵巧手执行器名称
ARM_ACTUATOR_NAMES = [
    'e_helfin_joint1_actuator', 'e_helfin_joint2_actuator', 'e_helfin_joint3_actuator',
    'e_helfin_joint4_actuator', 'e_helfin_joint5_actuator', 'e_helfin_joint6_actuator'
]
HAND_ACTUATOR_NAMES = [
    'e_hhand_index_finger', 'e_hhand_middle_finger', 'e_hhand_ring_finger',
    'e_hhand_little_finger', 'e_hhand_thumb_flex', 'e_hhand_thumb_rot'
]

# 控制步长
END_EFFECTOR_STEP = 0.01
END_EFFECTOR_ROT_STEP = 0.05
HAND_STEP = 0.1

# 控制标志和状态
exit_flag = [False]
key_state = {}
target_pos = None  # 将在应用 keyframe 后通过正运动学获取
target_euler = None  # 将在应用 keyframe 后通过正运动学获取
current_keyframe = []  # 当前使用的关键帧

def on_press(key):
    try:
        k = key.char.lower()
        key_state[k] = True
        if k == 'z':
            exit_flag[0] = True
        # Keyframe 快捷键
        elif k == '0':
            current_keyframe.clear()
            current_keyframe.append('home')
            print("切换到 home 关键帧")
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_flag[0] = True

def on_release(key):
    try:
        k = key.char.lower()
        key_state[k] = False
    except AttributeError:
        pass

def apply_keyframe(model, data, keyframe_name):
    """应用指定的关键帧"""
    # 查找关键帧ID
    keyframe_id = None
    for i in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        if name == keyframe_name:
            keyframe_id = i
            break
    
    if keyframe_id is not None:
        # 应用关键帧
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        print(f"已应用关键帧: {keyframe_name}")
        return True
    else:
        print(f"未找到关键帧: {keyframe_name}")
        return False

def add_random_screws_to_xml(xml_file, num_screws=3):
    """添加多个随机位置的螺丝到XML文件，保持现有参数设置"""
    
    # 保持您现有的所有参数设置
    # 螺丝参数
    mass = 0.5  # 500g = 0.5kg
    
    # 螺丝物理参数计算（假设为钢制螺丝，密度7850 kg/m³）
    # 假设螺丝长度L=80mm=0.08m，直径D=8mm=0.008m，半径r=0.004m
    L = 0.095  # 长度 (m)
    r = 0.006  # 半径 (m)
    
    # 对于实心圆柱体的惯性矩阵（近似螺丝为圆柱体）：
    # Ixx = Iyy = (1/12) * m * (3*r² + L²)  # 垂直于轴线的惯性
    # Izz = (1/2) * m * r²                   # 沿轴线的惯性
    Ixx = Iyy = (1/12) * mass * (3 * r**2 + L**2)  # ≈ 0.000267 kg⋅m²
    Izz = (1/2) * mass * r**2                       # ≈ 0.000004 kg⋅m²
    
    # 质心位置（螺丝头部可能较重，质心稍微偏移）
    com_offset = [0, 0, 0.01]  # 质心向螺丝头部偏移3mm
    
    with open(xml_file, 'r') as f:
        xml_content = f.read()
    
    # 移除keyframe定义，因为添加随机物体后DOF数量会改变
    import re
    # 移除整个keyframe section
    xml_content = re.sub(r'<keyframe>.*?</keyframe>', '', xml_content, flags=re.DOTALL)
    
    # 添加mesh资源到asset部分和编译器设置
    asset_section = '''
  <compiler coordinate="local" inertiafromgeom="auto"/>
  
  <option timestep="0.002" iterations="50" tolerance="1e-10" solver="Newton" jacobian="dense"/>
  
  <asset>
    <mesh name="screw_mesh" file="stl/screw.STL" scale="1 1 1"/>
  </asset>'''
    
    # 如果已经有asset部分，在其中添加mesh，否则创建新的asset部分
    if '<asset>' in xml_content:
        # 在现有asset部分中添加mesh
        xml_content = xml_content.replace('</asset>', '    <mesh name="screw_mesh" file="stl/screw.STL" scale="0.001 0.001 0.001"/>\n  </asset>')
    else:
        # 在mujoco标签后添加新的asset部分
        xml_content = xml_content.replace('<mujoco model="push_grasp_scene">', '<mujoco model="push_grasp_scene">\n' + asset_section)
    
    # 生成多个螺丝，位置随机范围相同
    screw_positions = []
    all_screw_bodies = ""
    
    for i in range(num_screws):
        # 为每个螺丝生成随机位置（使用相同的范围）
        x = random.uniform(-0.7, -0.8)
        y = random.uniform(-0.1, 0.1)
        z = 0.25  # 抬高到25cm，确保不与地面穿透
        
        # 为每个螺丝生成随机旋转角度（欧拉角，单位：弧度）
        roll = random.uniform(0, 2 * 3.14159)   # 绕X轴旋转（0-360度）
        pitch = random.uniform(0, 2 * 3.14159)  # 绕Y轴旋转（0-360度）
        yaw = random.uniform(0, 2 * 3.14159)    # 绕Z轴旋转（0-360度）
        
        screw_positions.append((x, y, z, roll, pitch, yaw))
        
        # 为每个螺丝创建唯一的名称
        screw_body = f'''
    <!-- 随机动态螺丝 #{i+1} -->
    <body name="random_screw_{i+1}" pos="{x} {y} {z}" euler="{roll} {pitch} {yaw}">
      <!-- 自由关节：允许物体在3D空间中自由移动和旋转，带阻尼和电枢惯量 -->
      <freejoint name="random_screw_{i+1}_freejoint"/>
      <!-- 惯性属性：质量和惯性矩阵，质心向螺丝头部偏移 -->
      <inertial pos="{com_offset[0]} {com_offset[1]} {com_offset[2]}" mass="{mass}" diaginertia="{Ixx:.6f} {Iyy:.6f} {Izz:.6f}"/>
      <!-- 几何形状和碰撞检测：钢制螺丝，高摩擦系数，优化接触参数 -->
      <!-- friction: [滑动摩擦, 滚动摩擦, 旋转摩擦] - 螺纹表面具有高摩擦 -->
      <!-- solref: [时间常数, 阻尼比] - 接触软度和稳定性 -->
      <!-- solimp: [dmin, dmax, width] - 接触阻抗参数，控制接触力响应 -->
      <!-- margin: 碰撞检测边距，提高稳定性 -->
      <!-- condim: 接触约束维度，6维允许完整的接触力和力矩 -->
      <geom name="random_screw_{i+1}_geom" type="mesh" mesh="screw_mesh" 
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
    
    # 将所有螺丝添加到XML中
    xml_content = xml_content.replace('</worldbody>', all_screw_bodies + '\n  </worldbody>')
    temp_xml = "temp_scene.xml"
    with open(temp_xml, 'w') as f:
        f.write(xml_content)
    return temp_xml, screw_positions


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
    if quat is not None and quat.strip():
        # MuJoCo的四元数顺序是 (w, x, y, z), 而Scipy需要 (x, y, z, w)
        # 转换公式: [w, x, y, z] → [x, y, z, w]
        q = np.fromstring(quat, sep=' ')
        rot_matrix = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    elif euler is not None and euler.strip():
        # 假设欧拉角顺序为 'xyz'
        # 旋转矩阵: R = Rz(ψ) @ Ry(θ) @ Rx(φ)
        robot_e = np.fromstring(euler, sep=' ')
        rot_matrix = Rotation.from_euler('XYZ', robot_e).as_matrix()
    # 如果既没有quat也没有euler，或者属性为空，rot_matrix保持为单位矩阵，表示无旋转

    # 构建4x4变换矩阵
    # 数学公式: T = [R    p]
    #                [0    1]
    transformation = np.eye(4)
    transformation[:3, :3] = rot_matrix  # 旋转部分 R
    transformation[:3, 3] = pos  # 平移部分 p
    return transformation


def create_chain_from_mjcf(xml_file):
    """
    从 MuJoCo XML 文件创建 ikpy 运动学链，专门针对 Elfin15 机械臂。
    
    Args:
        xml_file (str): MuJoCo XML 文件路径
        
    Returns:
        ikpy.chain.Chain: 创建的运动学链
    """
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    # 创建基座链接（固定）
    links = [ikpy.link.OriginLink()]
    active_links_mask = [False]
    
    # 按照嵌套结构遍历所有 body 元素
    body_elements = []
    
    # 从 worldbody 开始，递归找到所有 elfin_link 元素
    def find_elfin_bodies(element, bodies_list):
        for body in element.findall('body'):
            name = body.get('name', '')
            if name.startswith('elfin_link'):
                bodies_list.append(body)
            find_elfin_bodies(body, bodies_list)
    
    find_elfin_bodies(worldbody, body_elements)
    
    # 按名称排序确保顺序正确
    body_elements.sort(key=lambda x: x.get('name'))
    
    # 为每个 body 创建链接
    for body in body_elements:
        body_name = body.get('name')
        joint = body.find('joint')
        
        if joint is not None:
            # 获取关节信息
            joint_name = joint.get('name')
            axis_str = joint.get('axis', '0 0 1')
            axis = np.fromstring(axis_str, sep=' ')
            range_str = joint.get('range')
            bounds = tuple(map(float, range_str.split())) if range_str else (-3.14159, 3.14159)
            
            # 获取 body 的变换信息
            transform = get_transformation(body)
            translation = transform[:3, 3]
            orientation = Rotation.from_matrix(transform[:3, :3]).as_euler('xyz')
            
            # 创建链接
            link = ikpy.link.URDFLink(
                name=joint_name,
                origin_translation=translation,
                origin_orientation=orientation,
                rotation=axis,
                bounds=bounds
            )
            
            links.append(link)
            active_links_mask.append(True)
    
    # 添加末端执行器（固定）
    links.append(ikpy.link.URDFLink(
        name="end_effector",
        origin_translation=[0, 0, 0.171],  # 根据实际情况调整
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    ))
    active_links_mask.append(False)
    
    # 创建运动学链
    chain = ikpy.chain.Chain(links, active_links_mask=active_links_mask)
    return chain



def verify_ik_solution(chain, target_joints, target_pos, target_euler, pos_threshold=0.01, rot_threshold=0.1):
    """验证逆运动学解的准确性
    
    Args:
        chain: 运动学链
        target_joints: 逆运动学求解的关节角度
        target_pos: 目标位置
        target_euler: 目标欧拉角
        pos_threshold: 位置误差阈值 (m)
        rot_threshold: 旋转误差阈值 (rad)
        
    Returns:
        bool: True表示解合理，False表示解不合理
    """
    try:
        # 用求解出的关节角度进行正运动学计算
        ikpy_joints = np.zeros(8)  # 基座+6关节+末端
        ikpy_joints[1:7] = target_joints
        
        # 计算正运动学
        ee_transform = chain.forward_kinematics(ikpy_joints)
        calculated_pos = ee_transform[:3, 3]
        calculated_euler = Rotation.from_matrix(ee_transform[:3, :3]).as_euler('xyz')
        
        # 计算位置和姿态误差
        pos_error = np.linalg.norm(calculated_pos - target_pos)
        rot_error = np.linalg.norm(calculated_euler - target_euler)
        
        # 判断误差是否在合理范围内
        pos_ok = pos_error <= pos_threshold
        rot_ok = rot_error <= rot_threshold
        
        if not pos_ok or not rot_ok:
            print(f"⚠️  逆运动学解验证失败:")
            print(f"   位置误差: {pos_error:.4f}m (阈值: {pos_threshold}m) {'✓' if pos_ok else '✗'}")
            print(f"   姿态误差: {rot_error:.4f}rad (阈值: {rot_threshold}rad) {'✓' if rot_ok else '✗'}")
            print(f"   目标位置: {target_pos}")
            print(f"   计算位置: {calculated_pos}")
            print(f"   目标姿态: {target_euler}")
            print(f"   计算姿态: {calculated_euler}")
            return False
        
        return True
        
    except Exception as e:
        print(f"逆运动学解验证过程出错: {e}")
        return False

def solve_inverse_kinematics(chain, target_pos, target_euler, current_joints):
    """求解逆运动学"""
    # 构建目标变换矩阵
    target_transform = np.eye(4)
    target_transform[:3, :3] = Rotation.from_euler('xyz', target_euler).as_matrix()
    target_transform[:3, 3] = target_pos
    
    try:
        # ikpy需要完整的关节状态，包括固定关节
        initial_position = np.zeros(8)  # 8个链接：基座+6关节+末端
        initial_position[1:7] = current_joints  # 设置6个关节的当前角度
        
        joint_angles = chain.inverse_kinematics(
            target_position=target_transform[:3,3],
            target_orientation=target_transform[:3,:3],
            initial_position=initial_position,
            max_iter=1000,
            orientation_mode="all"
        )
        
        # 返回6个关节角度
        return joint_angles[1:7]
        
    except Exception as e:
        print(f"逆运动学求解失败: {e}")
        return current_joints

def main():
    global target_pos, target_euler
    
    xml_file = "push-grasp-scene.xml"
    if not os.path.exists(xml_file):
        print(f"错误：找不到文件 {xml_file}")
        return
    
    # 步骤1: 先加载原始XML并应用keyframe
    print("=== 步骤1: 加载原始XML并应用keyframe ===")
    try:
        original_model = mujoco.MjModel.from_xml_path(xml_file)
        original_data = mujoco.MjData(original_model)
        print(f"原始XML DOF数量: {original_model.nq}")
        print(f"原始XML keyframe数量: {original_model.nkey}")
        original_model.opt.gravity[:] = [0, 0, -9.8]

        # 应用home keyframe到原始模型
        if apply_keyframe(original_model, original_data, "home"):
            print("Home keyframe 应用成功")
            # 保存应用keyframe后的机械臂状态
            arm_joint_names = [name.replace('_actuator', '') for name in ARM_ACTUATOR_NAMES]
            arm_joint_ids = [original_model.joint(name).id for name in arm_joint_names]
            keyframe_arm_state = original_data.qpos[arm_joint_ids].copy()
            print(f"Keyframe机械臂状态: {keyframe_arm_state}")
            
            # 保存应用keyframe后的灵巧手状态
            try:
                hand_actuator_ids = [original_model.actuator(name).id for name in HAND_ACTUATOR_NAMES]
                keyframe_hand_ctrl = original_data.ctrl[hand_actuator_ids].copy()
                print(f"Keyframe灵巧手控制信号: {keyframe_hand_ctrl}")
            except Exception as e:
                print(f"灵巧手状态保存失败: {e}")
                keyframe_hand_ctrl = None
        else:
            print("Home keyframe 应用失败")
            keyframe_arm_state = None
            keyframe_hand_ctrl = None
    except Exception as e:
        print(f"原始XML处理失败: {e}")
        return
    
    # 步骤2: 添加随机螺丝并创建最终模型
    print("\n=== 步骤2: 添加3个随机螺丝 ===")
    temp_xml, screw_positions = add_random_screws_to_xml(xml_file, num_screws=3)
    for i, pos in enumerate(screw_positions):
        print(f"螺丝 #{i+1} 位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) 旋转: ({pos[3]:.2f}, {pos[4]:.2f}, {pos[5]:.2f}) rad")
    
    # 加载包含3个随机螺丝的最终模型
    model = mujoco.MjModel.from_xml_path(temp_xml)
    data = mujoco.MjData(model)
    print(f"最终模型DOF数量: {model.nq}")
    
    # 步骤3: 将keyframe状态应用到最终模型
    print("\n=== 步骤3: 应用keyframe状态到最终模型 ===")
    if keyframe_arm_state is not None:
        # 获取最终模型中的机械臂关节ID
        arm_joint_names = [name.replace('_actuator', '') for name in ARM_ACTUATOR_NAMES]
        final_arm_joint_ids = [model.joint(name).id for name in arm_joint_names]
        
        # 将keyframe的机械臂状态应用到最终模型
        for i, joint_id in enumerate(final_arm_joint_ids):
            data.qpos[joint_id] = keyframe_arm_state[i]
            data.ctrl[i] = keyframe_arm_state[i]  # 同时设置控制信号
        
        print("机械臂Keyframe状态已应用到最终模型")
    else:
        print("机械臂使用默认初始化")
    
    # 应用灵巧手keyframe状态
    if keyframe_hand_ctrl is not None:
        try:
            # 获取最终模型中的灵巧手执行器ID
            hand_actuator_ids = [model.actuator(name).id for name in HAND_ACTUATOR_NAMES]
            
            # 将keyframe的灵巧手控制信号应用到最终模型
            for i, actuator_id in enumerate(hand_actuator_ids):
                data.ctrl[actuator_id] = keyframe_hand_ctrl[i]  # 直接设置控制信号
            
            print("灵巧手Keyframe控制信号已应用到最终模型")
        except Exception as e:
            print(f"灵巧手状态应用失败: {e}")
    else:
        print("灵巧手使用默认初始化")

    # 创建运动学链
    try:
        arm_chain = create_chain_from_mjcf("xacro-to-urdf-to-mjcf-converter/mjcf_models/elfin15/elfin15.xml")
        print("运动学链创建成功")
        print(f"链接数量: {len(arm_chain.links)}")
        print(f"活动关节: {sum(arm_chain.active_links_mask)}")
        
        # 获取当前关节角度并使用 ikpy 计算正运动学
        arm_joint_names = [name.replace('_actuator', '') for name in ARM_ACTUATOR_NAMES]
        arm_joint_ids = [model.joint(name).id for name in arm_joint_names]
        current_joints = data.qpos[arm_joint_ids]
        
        # ikpy 需要包含固定关节的完整关节状态
        ikpy_joints = np.zeros(8)  # 基座+6关节+末端
        ikpy_joints[1:7] = current_joints
        
        # 使用 ikpy 正运动学计算末端执行器变换矩阵
        ee_transform = arm_chain.forward_kinematics(ikpy_joints)
        
        # 提取位置和姿态
        target_pos = ee_transform[:3, 3]
        target_euler = Rotation.from_matrix(ee_transform[:3, :3]).as_euler('xyz')
        
        print(f"使用 ikpy 正运动学计算的初始位置: {target_pos}")
        print(f"使用 ikpy 正运动学计算的初始姿态: {target_euler}")
        
    except Exception as e:
        print(f"运动学链创建失败: {e}")
        return
    
    # 获取执行器ID
    try:
        arm_actuator_ids = [model.actuator(name).id for name in ARM_ACTUATOR_NAMES]
        print(f"机械臂执行器ID: {arm_actuator_ids}")
    except Exception as e:
        print(f"机械臂执行器获取失败: {e}")
        return
    
    try:
        hand_actuator_ids = [model.actuator(name).id for name in HAND_ACTUATOR_NAMES]
        print(f"灵巧手执行器ID: {hand_actuator_ids}")
    except Exception as e:
        print(f"灵巧手执行器获取失败: {e}")
        hand_actuator_ids = []
    

    
    # 启动键盘监听
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    print("\n控制说明：")
    print("末端执行器位置控制：W/S(X轴) A/D(Y轴) Q/E(Z轴)")
    print("末端执行器旋转控制：R/F(绕X) T/G(绕Y) Y/H(绕Z)")
    print("灵巧手控制：1-6(正向) Shift+1-6(反向)")
    print("Keyframe控制：0(回到home位置)")
    print("ESC 或 Z 退出")
    print("\n安全验证：")
    print("- 逆运动学解会自动验证准确性")
    print("- 位置误差阈值: 0.01m, 姿态误差阈值: 0.1rad")
    print("- 不合理的解会被拒绝执行")
    print(f"\n初始目标位置: {target_pos}")
    
    # 启动viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo viewer 已启动，开始控制循环...")
        
        while viewer.is_running() and not exit_flag[0]:
            
            # 处理 keyframe 切换
            if current_keyframe:
                if apply_keyframe(model, data, current_keyframe[0]):
                    # keyframe 应用成功后，更新目标位置和姿态
                    current_joints = data.qpos[arm_joint_ids]
                    ikpy_joints = np.zeros(8)
                    ikpy_joints[1:7] = current_joints
                    ee_transform = arm_chain.forward_kinematics(ikpy_joints)
                    target_pos = ee_transform[:3, 3]
                    target_euler = Rotation.from_matrix(ee_transform[:3, :3]).as_euler('xyz')
                    print(f"切换后目标位置: {target_pos}")
                    print(f"切换后目标姿态: {target_euler}")
                current_keyframe.clear()  # 重置
            
            # 末端执行器位置控制
            pos_changed = False
            # 备份当前目标值，以便验证失败时恢复
            backup_target_pos = target_pos.copy()
            backup_target_euler = target_euler.copy()
            
            if key_state.get('w'):
                target_pos[0] += END_EFFECTOR_STEP
                pos_changed = True
            if key_state.get('s'):
                target_pos[0] -= END_EFFECTOR_STEP
                pos_changed = True
            if key_state.get('a'):
                target_pos[1] += END_EFFECTOR_STEP
                pos_changed = True
            if key_state.get('d'):
                target_pos[1] -= END_EFFECTOR_STEP
                pos_changed = True
            if key_state.get('q'):
                target_pos[2] += END_EFFECTOR_STEP
                pos_changed = True
            if key_state.get('e'):
                target_pos[2] -= END_EFFECTOR_STEP
                pos_changed = True
            
            # 末端执行器旋转控制
            rot_changed = False
            if key_state.get('r'):
                target_euler[0] += END_EFFECTOR_ROT_STEP
                rot_changed = True
            if key_state.get('f'):
                target_euler[0] -= END_EFFECTOR_ROT_STEP
                rot_changed = True
            if key_state.get('t'):
                target_euler[1] += END_EFFECTOR_ROT_STEP
                rot_changed = True
            if key_state.get('g'):
                target_euler[1] -= END_EFFECTOR_ROT_STEP
                rot_changed = True
            if key_state.get('y'):
                target_euler[2] += END_EFFECTOR_ROT_STEP
                rot_changed = True
            if key_state.get('h'):
                target_euler[2] -= END_EFFECTOR_ROT_STEP
                rot_changed = True
            
            # 如果目标发生变化，求解逆运动学
            if pos_changed or rot_changed:
                current_joints = data.qpos[arm_joint_ids]
                target_joints = solve_inverse_kinematics(
                    arm_chain, target_pos, target_euler, current_joints
                )
                
                # 验证逆运动学解的准确性
                if target_joints is not None and verify_ik_solution(
                    arm_chain, target_joints, target_pos, target_euler
                ):
                    # 验证通过，应用控制信号
                    for i, actuator_id in enumerate(arm_actuator_ids):
                        if i < len(target_joints):
                            data.ctrl[actuator_id] = target_joints[i]
                    
                    if pos_changed:
                        print(f"✓ 新目标位置: {target_pos}")
                    if rot_changed:
                        print(f"✓ 新目标旋转: {target_euler}")
                else:
                    # 验证失败，恢复目标值并不执行控制命令
                    target_pos[:] = backup_target_pos  # 恢复位置
                    target_euler[:] = backup_target_euler  # 恢复姿态
                    print("❌ 逆运动学解不合理，已恢复目标值，跳过此次控制命令")
                    if pos_changed:
                        print(f"❌ 位置目标被拒绝，已恢复到: {target_pos}")
                    if rot_changed:
                        print(f"❌ 姿态目标被拒绝，已恢复到: {target_euler}")
            
            # 灵巧手控制
            for idx, k in enumerate(['1','2','3','4','5','6']):
                if idx < len(hand_actuator_ids):
                    if key_state.get(k):
                        data.ctrl[hand_actuator_ids[idx]] += HAND_STEP
                    if key_state.get(k.upper()):
                        data.ctrl[hand_actuator_ids[idx]] -= HAND_STEP
                    
                    # 限制控制范围
                    ctrl_range = model.actuator_ctrlrange[hand_actuator_ids[idx]]
                    data.ctrl[hand_actuator_ids[idx]] = np.clip(
                        data.ctrl[hand_actuator_ids[idx]], 
                        ctrl_range[0], ctrl_range[1]
                    )
            
            mujoco.mj_step(model, data)
            viewer.sync()
            # time.sleep(0.01)
    
    listener.stop()
    listener.join()
    print("程序已退出。")

if __name__ == "__main__":
    main() 