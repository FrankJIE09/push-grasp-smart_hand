# 训练结果文件夹结构说明

## 📁 文件夹结构

```
training_results/
├── models/                    # 模型文件夹
│   ├── screw_pushing_agent.zip    # 训练好的模型文件
│   └── README.md                  # 模型文件夹说明
├── logs/                      # 日志文件夹
│   ├── PPO_1/                     # TensorBoard日志
│   │   └── events.out.tfevents.*  # 训练日志文件
│   └── README.md                  # 日志文件夹说明
├── checkpoints/               # 检查点文件夹
│   ├── screw_pushing_checkpoint_*.zip  # 训练过程中的检查点
│   └── README.md                  # 检查点文件夹说明
├── evaluations/              # 评估文件夹
│   ├── evaluation_*.json          # 模型评估结果
│   └── README.md                  # 评估文件夹说明
├── training_info.json        # 训练信息文件
└── README.md                 # 主文件夹说明
```

## 📋 文件说明

### 🤖 模型文件夹 (`models/`)
- **用途**: 存放训练好的模型文件
- **文件类型**: `.zip` (Stable-Baselines3格式), `.pth` (PyTorch格式)
- **主要文件**: `screw_pushing_agent.zip` - 最终训练好的模型

### 📈 日志文件夹 (`logs/`)
- **用途**: 存放训练日志和TensorBoard文件
- **文件类型**: TensorBoard事件文件
- **功能**: 可以通过TensorBoard查看训练曲线和指标

### 📊 检查点文件夹 (`checkpoints/`)
- **用途**: 存放训练过程中的检查点文件
- **文件命名**: `screw_pushing_checkpoint_{步数}.zip`
- **功能**: 可以从检查点恢复训练或加载中间模型

### 📋 评估文件夹 (`evaluations/`)
- **用途**: 存放模型评估结果
- **文件类型**: JSON格式
- **内容**: 包含评估指标、成功率、奖励等统计信息

### 📄 训练信息文件 (`training_info.json`)
- **用途**: 记录训练的基本信息
- **内容**: 
  - 训练时间
  - 总时间步数
  - 配置文件
  - 模型路径
  - 时间戳

## 🚀 使用方法

### 1. 开始训练
```bash
python rl_screw_pushing.py
```

### 2. 查看训练进度
```bash
# 使用TensorBoard查看训练曲线
tensorboard --logdir training_results/logs

# 查看训练信息
cat training_results/training_info.json
```

### 3. 加载模型进行推理
```python
from stable_baselines3 import PPO

# 加载训练好的模型
model = PPO.load("training_results/models/screw_pushing_agent")
```

### 4. 从检查点恢复训练
```python
# 加载检查点继续训练
model = PPO.load("training_results/checkpoints/screw_pushing_checkpoint_10000")
```

### 5. 查看评估结果
```bash
# 查看最新的评估结果
ls training_results/evaluations/
cat training_results/evaluations/evaluation_*.json
```

## ⚙️ 配置说明

在 `config.yaml` 中可以修改以下参数：

```yaml
training:
  results_dir: "training_results"    # 训练结果文件夹名称
  save_path: "screw_pushing_agent"  # 模型保存名称
  total_timesteps: 500000000        # 总训练步数
  # ... 其他训练参数

callback:
  save_freq: 10000                  # 检查点保存频率
  eval_freq: 100                    # 评估频率
  demo_freq: 10000                  # 演示频率
```

## 📊 监控训练

### TensorBoard监控
```bash
# 启动TensorBoard
tensorboard --logdir training_results/logs --port 6006

# 在浏览器中访问
# http://localhost:6006
```

### 实时查看训练统计
```bash
# 查看训练统计信息
cat training_results/training_stats.json

# 查看最新的检查点
ls -la training_results/checkpoints/
```

## 🔧 自定义配置

### 修改保存路径
在 `config.yaml` 中修改：
```yaml
training:
  results_dir: "my_training_results"  # 自定义文件夹名
  save_path: "my_model"               # 自定义模型名
```

### 添加自定义评估
在评估函数中添加自定义指标：
```python
# 在 evaluate_and_demonstrate 函数中添加
evaluation_results['custom_metric'] = your_custom_metric
```

## 📝 注意事项

1. **文件夹权限**: 确保有写入权限
2. **磁盘空间**: 训练文件可能较大，注意磁盘空间
3. **备份重要**: 定期备份重要的模型和检查点
4. **版本控制**: 建议将配置文件加入版本控制，但排除大型模型文件

## 🎯 最佳实践

1. **命名规范**: 使用有意义的文件夹和文件名称
2. **定期清理**: 删除不需要的检查点文件以节省空间
3. **实验记录**: 为每次实验创建独立的文件夹
4. **结果对比**: 使用TensorBoard对比不同实验的结果 