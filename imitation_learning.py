#!/usr/bin/env python3
"""
模仿学习系统
使用强化学习结果作为样板，过滤无价值的动作
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from typing import List, Tuple, Dict, Any
import pickle
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

# 导入原有的环境
from rl_screw_pushing import ScrewPushingEnv, load_config

def get_obs_array(obs):
    # 如果是tuple，取第一个
    if isinstance(obs, tuple):
        return np.asarray(obs[0], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)

class ValueFilter:
    """价值过滤器：过滤掉没有产生价值的动作"""
    
    def __init__(self, min_reward_threshold=0.0, min_improvement_threshold=0.01):
        self.min_reward_threshold = min_reward_threshold
        self.min_improvement_threshold = min_improvement_threshold
        self.episode_rewards = deque(maxlen=100)  # 记录最近100个episode的奖励
        
    def is_valuable_action(self, action: np.ndarray, reward: float, 
                          episode_reward: float, step_count: int) -> bool:
        """
        判断动作是否有价值
        
        Args:
            action: 动作向量
            reward: 当前步骤的奖励
            episode_reward: 当前episode的总奖励
            step_count: 当前步骤数
            
        Returns:
            bool: 动作是否有价值
        """
        # 1. 奖励阈值过滤
        if reward < self.min_reward_threshold:
            return False
            
        # 2. 奖励改进过滤
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards)
            if episode_reward < avg_reward + self.min_improvement_threshold:
                return False
                
        # 3. 动作有效性过滤（避免重复或无效动作）
        if np.linalg.norm(action) < 0.01:  # 动作太小
            return False
            
        return True
    
    def update_episode_reward(self, episode_reward: float):
        """更新episode奖励记录"""
        self.episode_rewards.append(episode_reward)

class DemonstrationCollector:
    """演示数据收集器"""
    
    def __init__(self, model_path: str, config: Dict[str, Any], 
                 value_filter: ValueFilter):
        self.model_path = model_path
        self.config = config
        self.value_filter = value_filter
        self.demonstrations = []
        
    def collect_demonstrations(self, num_episodes: int = 100, 
                             min_episode_length: int = 50) -> List[Dict]:
        """
        收集有价值的演示数据
        
        Args:
            num_episodes: 收集的episode数量
            min_episode_length: 最小episode长度
            
        Returns:
            List[Dict]: 演示数据列表
        """
        print(f"🔄 开始收集演示数据，目标episode数: {num_episodes}")
        
        # 加载训练好的模型
        try:
            model = PPO.load(self.model_path)
            print(f"✅ 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return []
        
        # 创建环境
        env = ScrewPushingEnv(config=self.config)
        env = DummyVecEnv([lambda: env])
        
        valuable_episodes = 0
        total_episodes = 0
        
        while valuable_episodes < num_episodes and total_episodes < num_episodes * 3:
            obs = env.reset()
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'episode_reward': 0.0,
                'episode_length': 0
            }
            
            done = False
            step_count = 0
            
            while not done:
                # 使用模型预测动作
                action, _ = model.predict(obs, deterministic=True)
                
                # 执行动作
                next_obs, reward, done, info = env.step(action)
                
                # 记录数据
                episode_data['observations'].append(obs[0].copy())
                episode_data['actions'].append(action[0].copy())
                episode_data['rewards'].append(reward[0])
                episode_data['episode_reward'] += reward[0]
                episode_data['episode_length'] += 1
                
                obs = next_obs
                step_count += 1
                
                # 检查episode是否过长
                if step_count > 500:  # 最大步数限制
                    break
            
            total_episodes += 1
            
            # 使用价值过滤器判断episode是否有价值
            if (episode_data['episode_length'] >= min_episode_length and
                self.value_filter.is_valuable_action(
                    np.array(episode_data['actions']), 
                    episode_data['episode_reward'],
                    episode_data['episode_reward'],
                    episode_data['episode_length']
                )):
                
                self.demonstrations.append(episode_data)
                valuable_episodes += 1
                self.value_filter.update_episode_reward(episode_data['episode_reward'])
                
                print(f"✅ 收集到有价值的episode {valuable_episodes}/{num_episodes}, "
                      f"奖励: {episode_data['episode_reward']:.3f}, "
                      f"长度: {episode_data['episode_length']}")
            else:
                print(f"❌ 过滤掉无价值的episode {total_episodes}, "
                      f"奖励: {episode_data['episode_reward']:.3f}, "
                      f"长度: {episode_data['episode_length']}")
        
        print(f"🎯 演示数据收集完成！")
        print(f"   总episode数: {total_episodes}")
        print(f"   有价值episode数: {len(self.demonstrations)}")
        print(f"   过滤率: {(total_episodes - len(self.demonstrations)) / total_episodes * 100:.1f}%")
        
        return self.demonstrations

class DemonstrationDataset(Dataset):
    """演示数据集"""
    
    def __init__(self, demonstrations: List[Dict], sequence_length: int = 10):
        self.demonstrations = demonstrations
        self.sequence_length = sequence_length
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self) -> List[Tuple]:
        """准备训练样本"""
        samples = []
        
        for demo in self.demonstrations:
            obs = np.array(demo['observations'])
            actions = np.array(demo['actions'])
            
            # 创建滑动窗口样本
            for i in range(len(obs) - self.sequence_length + 1):
                obs_seq = obs[i:i + self.sequence_length]
                action_seq = actions[i:i + self.sequence_length]
                
                samples.append((obs_seq, action_seq))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        obs_seq, action_seq = self.samples[idx]
        return (torch.FloatTensor(obs_seq), torch.FloatTensor(action_seq))

class ImitationNetwork(nn.Module):
    """模仿学习网络"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ImitationNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 编码器：将观察序列编码为隐藏状态
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 动作预测器：从隐藏状态预测动作
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
        # LSTM用于处理序列信息
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obs_seq: 观察序列 [batch_size, sequence_length, obs_dim]
            
        Returns:
            torch.Tensor: 预测的动作序列 [batch_size, sequence_length, action_dim]
        """
        batch_size, seq_len, _ = obs_seq.shape
        
        # 编码观察序列
        encoded = self.encoder(obs_seq)  # [batch_size, sequence_length, hidden_dim]
        
        # LSTM处理序列
        lstm_out, _ = self.lstm(encoded)  # [batch_size, sequence_length, hidden_dim]
        
        # 预测动作
        actions = self.action_predictor(lstm_out)  # [batch_size, sequence_length, action_dim]
        
        return actions

class ImitationTrainer:
    """模仿学习训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
    def train(self, demonstrations: List[Dict], 
              batch_size: int = 32, 
              learning_rate: float = 1e-4,
              num_epochs: int = 100,
              sequence_length: int = 10) -> ImitationNetwork:
        """
        训练模仿学习网络
        
        Args:
            demonstrations: 演示数据
            batch_size: 批次大小
            learning_rate: 学习率
            num_epochs: 训练轮数
            sequence_length: 序列长度
            
        Returns:
            ImitationNetwork: 训练好的网络
        """
        print(f"🎯 开始训练模仿学习网络")
        print(f"   演示数据数量: {len(demonstrations)}")
        print(f"   序列长度: {sequence_length}")
        print(f"   批次大小: {batch_size}")
        print(f"   学习率: {learning_rate}")
        print(f"   训练轮数: {num_epochs}")
        
        # 准备数据集
        dataset = DemonstrationDataset(demonstrations, sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 获取输入维度
        sample_obs, sample_action = dataset[0]
        obs_dim = sample_obs.shape[-1]
        action_dim = sample_action.shape[-1]
        
        print(f"   观察维度: {obs_dim}")
        print(f"   动作维度: {action_dim}")
        
        # 创建网络
        network = ImitationNetwork(obs_dim, action_dim).to(self.device)
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 训练记录
        train_losses = []
        
        # 开始训练
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_obs, batch_actions in dataloader:
                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                
                # 前向传播
                predicted_actions = network(batch_obs)
                
                # 计算损失
                loss = criterion(predicted_actions, batch_actions)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}/{num_epochs}, 损失: {avg_loss:.6f}")
        
        print(f"✅ 模仿学习训练完成！")
        
        # 绘制训练曲线
        self._plot_training_curve(train_losses)
        
        return network
    
    def _plot_training_curve(self, losses: List[float]):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Imitation Learning Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"imitation_learning_loss_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"📊 训练曲线已保存: {plot_path}")

class ImitationAgent:
    """模仿学习智能体"""
    
    def __init__(self, network: ImitationNetwork, sequence_length: int = 10):
        self.network = network
        self.sequence_length = sequence_length
        self.device = next(network.parameters()).device
        self.observation_buffer = deque(maxlen=sequence_length)
        
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        预测动作
        
        Args:
            observation: 当前观察
            
        Returns:
            np.ndarray: 预测的动作
        """
        obs = get_obs_array(observation)
        self.observation_buffer.append(obs)
        
        # 如果缓冲区未满，返回零动作
        if len(self.observation_buffer) < self.sequence_length:
            return np.zeros(self.network.action_dim)
        
        # 准备输入序列
        obs_seq = np.array(list(self.observation_buffer), dtype=np.float32)
        obs_tensor = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)
        
        # 预测动作
        with torch.no_grad():
            action_seq = self.network(obs_tensor)
            predicted_action = action_seq[0, -1].cpu().numpy()  # 取最后一个时间步的动作
        
        return predicted_action
    
    def reset(self):
        """重置智能体状态"""
        self.observation_buffer.clear()

def evaluate_imitation_agent(agent: ImitationAgent, config: Dict[str, Any], 
                           num_episodes: int = 10) -> Dict[str, float]:
    """
    评估模仿学习智能体
    
    Args:
        agent: 模仿学习智能体
        config: 环境配置
        num_episodes: 评估episode数
        
    Returns:
        Dict[str, float]: 评估结果
    """
    print(f"🔍 开始评估模仿学习智能体")
    
    env = ScrewPushingEnv(config=config)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        obs = get_obs_array(obs)
        agent.reset()
        
        episode_reward = 0.0
        step_count = 0
        
        done = False
        while not done and step_count < 500:
            action = agent.predict(obs)
            result = env.step(action)
            
            # 处理不同版本的环境返回值
            if len(result) == 4:
                obs, reward, done, info = result
            elif len(result) == 5:
                obs, reward, done, truncated, info = result
                done = done or truncated
            else:
                print(f"❌ 未知的环境step返回值格式: {result}")
                break
            
            obs = get_obs_array(obs)
            episode_reward += reward
            step_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # 判断是否成功（根据奖励阈值）
        if episode_reward > -15000:  # 根据实际情况调整阈值
            success_count += 1
        
        print(f"   Episode {episode + 1}: 奖励={episode_reward:.3f}, 步数={step_count}")
    
    # 计算统计结果
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    
    results = {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'success_rate': success_rate,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'std_reward': np.std(episode_rewards)
    }
    
    print(f"📊 评估结果:")
    print(f"   平均奖励: {avg_reward:.3f}")
    print(f"   平均步数: {avg_length:.1f}")
    print(f"   成功率: {success_rate:.1%}")
    print(f"   奖励范围: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")
    
    return results

def demo_imitation_agent_with_viewer(agent: ImitationAgent, config: dict, max_steps: int = 500):
    """
    用MuJoCo viewer展示模仿学习智能体的推理过程
    """
    import time
    env = ScrewPushingEnv(config=config, render_mode='human')  # 确保render_mode='human'
    obs = env.reset()
    obs = get_obs_array(obs)
    agent.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    while not done and step_count < max_steps:
        action = agent.predict(obs)
        result = env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
        elif len(result) == 5:
            obs, reward, done, truncated, info = result
            done = done or truncated
        else:
            print(f"❌ 未知的环境step返回值格式: {result}")
            break
        obs = get_obs_array(obs)
        total_reward += reward
        step_count += 1
        time.sleep(0.03)  # 控制仿真速度，30~40ms一帧更流畅

    print(f"演示结束，总步数: {step_count}, 总奖励: {total_reward:.2f}")
    env.close()

def main():
    """主函数"""
    print("🎯 模仿学习系统")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 创建价值过滤器 - 调整参数以适应当前情况
    value_filter = ValueFilter(
        min_reward_threshold=-20000.0,  # 大幅降低奖励阈值
        min_improvement_threshold=-1000.0  # 允许负改进
    )
    
    # 收集演示数据
    collector = DemonstrationCollector(
        model_path="training_results/models/screw_pushing_agent",
        config=config,
        value_filter=value_filter
    )
    
    demonstrations = collector.collect_demonstrations(
        num_episodes=10,  # 减少目标episode数
        min_episode_length=20  # 减少最小episode长度
    )
    
    if not demonstrations:
        print("❌ 没有收集到有价值的演示数据")
        return
    
    print(f"✅ 成功收集到 {len(demonstrations)} 个有价值的演示数据")
    
    # 训练模仿学习网络
    trainer = ImitationTrainer(config)
    network = trainer.train(
        demonstrations=demonstrations,
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=30,  # 减少训练轮数
        sequence_length=10
    )
    
    # 创建模仿学习智能体
    agent = ImitationAgent(network, sequence_length=10)
    
    # 评估智能体
    results = evaluate_imitation_agent(agent, config, num_episodes=5)
    
    # 保存模型和结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存网络
    torch.save(network.state_dict(), f"imitation_network_{timestamp}.pth")
    print(f"💾 网络已保存: imitation_network_{timestamp}.pth")
    
    # 保存演示数据
    with open(f"demonstrations_{timestamp}.pkl", 'wb') as f:
        pickle.dump(demonstrations, f)
    print(f"💾 演示数据已保存: demonstrations_{timestamp}.pkl")
    
    # 保存评估结果
    with open(f"imitation_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 评估结果已保存: imitation_results_{timestamp}.json")
    
    # 演示智能体
    demo_imitation_agent_with_viewer(agent, config, max_steps=500)

if __name__ == "__main__":
    main() 