# 机械臂和灵巧手键盘控制系统

这是一个使用MuJoCo仿真环境的机械臂和灵巧手键盘控制系统。

## 功能特性

- 使用键盘控制机械臂的6自由度运动
- 使用键盘控制灵巧手的6自由度运动
- 在场景中自动生成随机位置的方块
- 使用ik_modules进行正逆运动学计算

## 环境要求

- Python 3.9+
- Conda环境管理
- MuJoCo许可证（免费获取）

## 安装步骤

1. 克隆或下载项目文件
2. 运行环境设置脚本：
   ```bash
   chmod +x setup_environment.sh
   ./setup_environment.sh
   ```

3. 激活conda环境：
   ```bash
   conda activate mujoco
   ```

## 使用方法

1. 确保在mujoco环境中：
   ```bash
   conda activate mujoco
   ```

2. 运行键盘控制程序：
   ```bash
   python keyboard_control.py
   ```

## 控制说明

### 机械臂控制
- **W/S**: 前后移动
- **A/D**: 左右移动  
- **Q/E**: 上下移动
- **R/F**: Yaw旋转

### 灵巧手控制
- **1-6**: 手指关节1-6（正向）
- **7-9**: 手腕3自由度（正向）
- **0/-/=**: 手掌控制（正向）
- **1-6**: 手指关节1-6（反向）
- **7-9**: 手腕3自由度（反向）
- **0/-/=**: 手掌控制（反向）

### 其他控制
- **ESC**: 退出程序

## 文件结构

```
push-grasp-smart_hand/
├── keyboard_control.py      # 主控制程序
├── push-grasp-scene.xml    # MuJoCo场景文件
├── ik_modules/             # 运动学模块
│   ├── kinematic_chain.py  # 运动学链构建
│   └── transform_utils.py  # 变换工具函数
├── requirements.txt         # Python依赖
├── setup_environment.sh    # 环境设置脚本
└── README.md              # 说明文档
```

## 技术细节

- 使用MuJoCo进行物理仿真
- 使用ikpy进行逆运动学计算
- 支持实时键盘交互
- 自动生成随机方块作为目标物体

## 故障排除

1. **MuJoCo许可证问题**：
   - 访问 https://www.mujoco.org/ 获取免费许可证
   - 将许可证文件放在正确位置

2. **依赖安装问题**：
   - 确保使用conda环境
   - 检查Python版本是否为3.9+

3. **键盘控制无响应**：
   - 确保窗口处于焦点状态
   - 检查键盘布局设置

## 许可证

本项目遵循MIT许可证。 