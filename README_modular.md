# 螺丝推动强化学习系统 - 模块化版本

## 📁 项目结构

```
push-grasp-smart_hand/
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── kinematics.py        # 运动学相关工具
│   └── xml_utils.py         # XML处理工具
├── environment/              # 环境模块
│   ├── __init__.py
│   └── screw_pushing_env.py # 螺丝推动强化学习环境
├── training/                 # 训练模块
│   ├── __init__.py
│   └── trainer.py           # 训练器
├── evaluation/               # 评估模块
│   ├── __init__.py
│   └── evaluator.py         # 评估器
├── main.py                   # 主程序入口
├── test_modular_environment.py # 模块化环境测试
└── README_modular.md         # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_rl.txt
```

### 2. 测试模块化环境

```bash
python test_modular_environment.py
```

### 3. 训练智能体

```bash
# 使用默认参数训练
python main.py --mode train

# 自定义训练参数
python main.py --mode train --total_timesteps 50000 --model_path my_agent
```

### 4. 评估智能体

```bash
# 评估训练好的智能体
python main.py --mode evaluate --model_path screw_pushing_agent --num_episodes 10
```

### 5. 演示智能体

```bash
# 使用MuJoCo viewer演示
python main.py --mode demonstrate --model_path screw_pushing_agent --num_episodes 5
```

## 📋 模块说明

### utils/kinematics.py
- **功能**: 运动学计算工具
- **主要函数**:
  - `create_chain_from_mjcf()`: 从MuJoCo XML创建ikpy运动学链
  - `solve_inverse_kinematics()`: 求解逆运动学
  - `get_end_effector_pose()`: 获取末端执行器位姿

### utils/xml_utils.py
- **功能**: XML文件处理工具
- **主要函数**:
  - `add_challenge_screws_to_xml()`: 添加训练用螺丝
  - `remove_keyframe_section()`: 移除keyframe段
  - `check_screw_spacing()`: 检查螺丝间距

### environment/screw_pushing_env.py
- **功能**: 螺丝推动强化学习环境
- **特性**:
  - 基于Gymnasium标准
  - 支持末端执行器位置和姿态控制
  - 自动螺丝间距检测和奖励计算
  - 集成MuJoCo物理仿真

### training/trainer.py
- **功能**: 强化学习训练器
- **特性**:
  - 基于PPO算法
  - 支持模型检查点保存
  - 自动评估和最佳模型保存
  - TensorBoard日志记录

### evaluation/evaluator.py
- **功能**: 智能体评估和演示
- **特性**:
  - 多episode评估
  - 成功率统计
  - MuJoCo viewer实时演示
  - 详细性能指标

## 🎯 使用示例

### 训练自定义智能体

```python
from training.trainer import ScrewPushingTrainer

# 创建训练器
trainer = ScrewPushingTrainer(
    total_timesteps=50000,
    save_freq=2000,
    eval_freq=1000,
    model_save_path="my_custom_agent"
)

# 开始训练
model = trainer.train()
```

### 评估智能体性能

```python
from evaluation.evaluator import ScrewPushingEvaluator

# 创建评估器
evaluator = ScrewPushingEvaluator(model_path="screw_pushing_agent")

# 评估多个episode
results = evaluator.evaluate_multiple_episodes(num_episodes=10)

# 查看结果
print(f"成功率: {results['success_rate']:.2%}")
print(f"平均奖励: {results['avg_reward']:.2f}")
```

### 自定义环境参数

```python
from environment.screw_pushing_env import ScrewPushingEnv

# 创建自定义环境
env = ScrewPushingEnv(
    num_screws=5,              # 5个螺丝
    min_screw_distance=0.20,   # 最小间距20cm
    max_episode_steps=1000     # 最大1000步
)
```

## 🔧 配置参数

### 环境参数
- `num_screws`: 螺丝数量 (默认: 3)
- `min_screw_distance`: 最小安全间距 (默认: 0.15m)
- `max_episode_steps`: 最大episode步数 (默认: 500)

### 训练参数
- `total_timesteps`: 总训练步数 (默认: 10000)
- `save_freq`: 模型保存频率 (默认: 1000)
- `eval_freq`: 评估频率 (默认: 500)
- `learning_rate`: 学习率 (默认: 3e-4)

### 动作空间
- 位置控制: ±0.01m (x, y, z)
- 姿态控制: ±0.05rad (roll, pitch, yaw)

### 观察空间 (19维)
- 末端位置: 3维
- 末端姿态: 3维
- 螺丝位置: 9维 (3个螺丝 × 3维)
- 螺丝间距: 3维
- 最小间距: 1维

## 📊 奖励函数

### 奖励组成
1. **距离奖励**: 基于螺丝间最小距离
   - 达到要求: +50 + 超额距离×20
   - 未达到: -100 × 距离差

2. **成功奖励**: 所有螺丝间距满足要求时 +200

3. **接近奖励**: 鼓励智能体接近需要分开的螺丝对

4. **时间惩罚**: 每步 -0.1

## 🐛 故障排除

### 常见问题

1. **运动学链创建失败**
   - 检查XML文件路径是否正确
   - 确保ikpy库已正确安装

2. **模型加载失败**
   - 检查模型文件是否存在
   - 确保模型文件完整

3. **MuJoCo viewer无法启动**
   - 检查MuJoCo安装
   - 确保图形驱动正常

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试单个模块
python test_modular_environment.py
```

## 📈 性能优化建议

1. **增加训练步数**: 建议至少50000步以获得良好性能
2. **调整奖励函数**: 根据任务需求优化奖励权重
3. **使用更复杂的网络**: 考虑使用CNN或Transformer架构
4. **多进程训练**: 使用多个环境并行训练
5. **超参数调优**: 使用Optuna等工具优化超参数

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证。 