import numpy as np
from scipy.spatial.transform import Rotation
from keyboard_control_ik import create_chain_from_mjcf, solve_inverse_kinematics
import mujoco
import mujoco.viewer

# 1. 构建运动学链
mjcf_path = "xacro-to-urdf-to-mjcf-converter/mjcf_models/elfin15/elfin15.xml"
chain = create_chain_from_mjcf(mjcf_path)

# 2. 设定目标末端位置和姿态
target_pos = np.array([-0.6, 0.0, 0.115])  # 末端xyz
target_euler = np.array([0, np.pi, 0])    # 朝下

# 3. 当前关节角度（初始值全0）
current_joints = np.array([0, 0, -1.57, 0, -1.57, 0])  # 机械臂home位置])

# 4. 求解逆运动学
home_joints = solve_inverse_kinematics(chain, target_pos, target_euler, current_joints)
print("home_joints (radians):", ", ".join([f"{x:.3f}" for x in home_joints]))
print("home_joints (degrees):", ", ".join([f"{x:.3f}" for x in np.degrees(home_joints)]))

# 5. 用 mujoco 加载模型并设置关节
xml_path = "push-grasp-scene.xml"  # 你的主场景
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 机械臂关节名称
arm_joint_names = [
    'e_helfin_joint1', 'e_helfin_joint2', 'e_helfin_joint3',
    'e_helfin_joint4', 'e_helfin_joint5', 'e_helfin_joint6'
]
arm_joint_ids = [model.joint(name).id for name in arm_joint_names]

# 设置关节角度
for i, joint_id in enumerate(arm_joint_ids):
    data.qpos[joint_id] = home_joints[i]
    data.ctrl[joint_id] = home_joints[i]
mujoco.mj_forward(model, data)

# 6. 展示
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started. Press ESC to exit.")
    while viewer.is_running():
        viewer.sync()