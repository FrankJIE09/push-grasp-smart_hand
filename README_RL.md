# 螺丝推动强化学习系统

基于MuJoCo物理仿真和Stable-Baselines3的螺丝推动强化学习训练系统。

## 🎯 系统目标

训练一个机械臂智能体学会推动螺丝到指定目标区域，同时确保螺丝之间保持安全距离，避免与灵巧手碰撞。

## 🔧 核心功能

- **安全螺丝布置**: 自动确保螺丝间距满足最小距离要求（默认15cm）
- **末端执行器控制**: 直接控制机械臂末端的xyz位置和rpy姿态
- **强化学习训练**: 使用PPO算法进行1000个episode的训练
- **实时可视化**: 通过MuJoCo viewer展示训练结果
- **模型保存**: 自动保存为.zip和.pth格式

## 📦 依赖安装

```bash
pip install -r requirements_rl.txt
```

主要依赖：
- `mujoco>=2.3.0`
- `stable-baselines3>=1.7.0`
- `torch>=1.12.0`
- `ikpy>=3.3.0`
- `numpy>=1.21.0`
- `scipy>=1.7.0`

## 🚀 快速开始

### 1. 测试环境

首先验证环境是否正常工作：

```bash
python test_environment.py
```

### 2. 开始训练

运行完整的训练流程：

```bash
python rl_screw_pushing.py
```

### 3. 仅测试训练好的模型

如果已有训练好的模型：

```python
from rl_screw_pushing import evaluate_and_demonstrate
evaluate_and_demonstrate("screw_pushing_agent", num_episodes=5)
```

## 🎮 环境参数

### 动作空间
- **维度**: 6维连续动作
- **范围**: 
  - 位置增量: ±0.01m (xyz)
  - 旋转增量: ±0.05rad (rpy)

### 观察空间
- **维度**: 18维
- **内容**:
  - 末端执行器位置(3) + 姿态(3)
  - 目标螺丝位置(3)
  - 其他螺丝位置(6)
  - 距离信息(3): 末端到螺丝、螺丝到目标、末端到目标

### 奖励函数
- **距离奖励**: -10.0 × 螺丝到目标距离
- **成功奖励**: +100.0 (螺丝进入目标区域)
- **接近奖励**: -5.0 × 末端到螺丝距离
- **时间惩罚**: -0.1 每步

## ⚙️ 配置参数

在`ScrewPushingEnv`初始化时可调整的参数：

```python
env = ScrewPushingEnv(
    xml_file="push-grasp-scene.xml",      # MuJoCo场景文件
    num_screws=3,                         # 螺丝数量
    min_screw_distance=0.15,              # 螺丝最小间距(m)
    target_area_center=[-0.6, 0.0, 0.25], # 目标区域中心
    target_area_radius=0.05,              # 目标区域半径(m)
    max_episode_steps=500                 # 最大步数
)
```

## 🏋️ 训练配置

PPO算法参数：

```python
model = PPO(
    "MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    total_timesteps=500000  # 1000 episodes × 500 steps
)
```

## 📁 文件结构

```
项目根目录/
├── rl_screw_pushing.py      # 主要强化学习代码
├── test_environment.py      # 环境测试脚本
├── requirements_rl.txt      # 依赖项列表
├── README_RL.md            # 说明文档
├── push-grasp-scene.xml    # 原始MuJoCo场景
├── temp_rl_scene.xml       # 生成的训练场景(自动创建)
└── 训练输出/
    ├── screw_pushing_agent.zip    # Stable-Baselines3模型
    ├── screw_pushing_agent.pth    # PyTorch权重
    ├── tensorboard_logs/          # TensorBoard日志
    └── screw_pushing_checkpoint_* # 训练检查点
```

## 🎯 自定义任务

### 修改螺丝数量和布局

```python
# 修改螺丝数量
env = ScrewPushingEnv(num_screws=5)

# 修改安全距离
env = ScrewPushingEnv(min_screw_distance=0.20)
```

### 修改目标区域

```python
# 修改目标位置和大小
env = ScrewPushingEnv(
    target_area_center=np.array([-0.5, 0.1, 0.25]),
    target_area_radius=0.08
)
```

### 自定义奖励函数

在`ScrewPushingEnv._calculate_reward()`方法中修改奖励计算逻辑。

## 📊 监控训练

### TensorBoard可视化

```bash
tensorboard --logdir ./tensorboard_logs/
```

### 训练回调

系统自动记录：
- Episode奖励
- Episode长度  
- 成功率（最近100个episode）
- 定期保存检查点

## 🔍 故障排除

### 常见问题

1. **运动学链创建失败**
   - 检查`xacro-to-urdf-to-mjcf-converter/mjcf_models/elfin15/elfin15.xml`是否存在
   - 系统会自动使用备选方法

2. **执行器ID获取失败**
   - 检查XML文件中的执行器名称是否正确
   - 确认机械臂模型加载正常

3. **逆运动学求解失败**
   - 目标位置可能超出机械臂工作空间
   - 系统会保持当前关节角度

4. **螺丝生成失败**
   - 增加`max_attempts`参数
   - 减小`min_screw_distance`或减少螺丝数量

### 调试模式

在环境创建时启用详细输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

env = ScrewPushingEnv()
```

## 🎯 性能优化

### 训练加速
- 减少物理仿真步数（修改`step()`中的仿真循环）
- 使用GPU训练（确保PyTorch支持CUDA）
- 并行环境训练

### 仿真优化
- 降低MuJoCo求解精度
- 减少contact检测精度
- 优化mesh分辨率

## 📈 结果分析

训练完成后，系统会自动：
1. 保存最终模型
2. 展示5个测试episode
3. 输出平均奖励和成功率
4. 生成TensorBoard日志

## 🤝 扩展开发

### 添加新的观察信息
在`_get_observation()`方法中添加新的传感器数据。

### 修改动作空间
调整`action_space`定义以支持不同的控制方式。

### 集成其他算法
替换PPO为其他强化学习算法（SAC、TD3等）。

## 📄 许可证

基于原项目许可证。

## 🆘 支持

如遇问题，请检查：
1. 所有依赖项是否正确安装
2. MuJoCo许可证是否有效
3. 场景XML文件是否完整

---

*最后更新：2024年* 