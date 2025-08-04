"""
强化学习评估器
"""

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from environment.screw_pushing_env import ScrewPushingEnv


class ScrewPushingEvaluator:
    """螺丝推动强化学习评估器"""

    def __init__(self, model_path="screw_pushing_agent"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """加载训练好的模型"""
        try:
            self.model = PPO.load(self.model_path)
            print(f"✅ 模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def evaluate_episode(self, env, max_steps=500):
        """评估单个episode"""
        obs, info = env.reset()
        total_reward = 0
        episode_steps = 0
        success = False

        for step in range(max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            episode_steps += 1
            
            if terminated:
                success = info.get('success', False)
                break

        return {
            'total_reward': total_reward,
            'episode_steps': episode_steps,
            'success': success,
            'min_screw_distance': info.get('min_screw_distance', 0.0)
        }

    def evaluate_multiple_episodes(self, num_episodes=5, max_steps=500, with_viewer=False):
        """评估多个episode"""
        if self.model is None:
            if not self.load_model():
                return None

        print(f"🔍 开始评估 {num_episodes} 个episode...")
        
        results = []
        success_count = 0
        
        for episode in range(num_episodes):
            env = ScrewPushingEnv()
            obs, info = env.reset()
            
            if with_viewer:
                # 带viewer的评估
                with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                    result = self._evaluate_single_episode_with_viewer(env, max_steps, episode)
            else:
                # 静默评估
                result = self.evaluate_episode(env, max_steps)
            
            results.append(result)
            
            if result['success']:
                success_count += 1
            
            print(f"Episode {episode + 1}: 奖励={result['total_reward']:.2f}, "
                  f"步数={result['episode_steps']}, 成功={result['success']}, "
                  f"最小间距={result['min_screw_distance']:.3f}m")

        # 计算统计信息
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_steps = np.mean([r['episode_steps'] for r in results])
        success_rate = success_count / num_episodes
        avg_min_distance = np.mean([r['min_screw_distance'] for r in results])

        print(f"\n📊 评估结果:")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   平均步数: {avg_steps:.1f}")
        print(f"   成功率: {success_rate:.2%}")
        print(f"   平均最小间距: {avg_min_distance:.3f}m")

        return {
            'results': results,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_rate': success_rate,
            'avg_min_distance': avg_min_distance
        }

    def _evaluate_single_episode_with_viewer(self, env, max_steps, episode_num):
        """使用viewer评估单个episode"""
        obs, info = env.reset()
        total_reward = 0
        episode_steps = 0
        success = False

        print(f"\n🎯 Episode {episode_num + 1} (带viewer)")
        print(f"🎯 任务目标: 确保所有螺丝间距≥{env.min_screw_distance:.2f}m")

        while episode_steps < max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            episode_steps += 1
            
            if terminated:
                success = info.get('success', False)
                break

        print(f"✅ Episode {episode_num + 1} 完成: 总奖励={total_reward:.2f}, "
              f"步数={episode_steps}, 成功={success}")

        return {
            'total_reward': total_reward,
            'episode_steps': episode_steps,
            'success': success,
            'min_screw_distance': info.get('min_screw_distance', 0.0)
        }

    def demonstrate_with_viewer(self, num_episodes=5, max_steps=500):
        """使用MuJoCo viewer演示训练好的智能体"""
        if self.model is None:
            if not self.load_model():
                return

        print(f"🎮 开始演示 {num_episodes} 个episode...")
        
        episode_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
            
            # 每个episode创建新环境
            env = ScrewPushingEnv()
            obs, info = env.reset()
            
            print(f"🎯 任务目标: 确保所有螺丝间距≥{env.min_screw_distance:.2f}m")
            print("🎮 使用训练好的智能体进行演示...")
            
            # 每个episode使用独立的viewer
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                episode_step = 0
                total_reward = 0
                
                while viewer.is_running() and episode_step < max_steps:
                    # 使用训练好的模型预测动作
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # 执行动作
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    episode_step += 1
                    
                    # 显示进度
                    if episode_step % 50 == 0:
                        print(f"   步骤 {episode_step}: 奖励={reward:.3f}, 总奖励={total_reward:.3f}, "
                              f"成功={info.get('success', False)}, 最小间距={info.get('min_screw_distance', 0.0):.3f}m")
                    
                    if terminated or truncated:
                        break
                
                print(f"✅ Episode {episode + 1} 完成: 总奖励={total_reward:.2f}, "
                      f"步数={episode_step}, 成功={info.get('success', False)}")
                
                episode_rewards.append(total_reward)
                if info.get('success', False):
                    success_count += 1
            
            # 等待用户确认继续下一个episode
            if episode < num_episodes - 1:
                input("按回车键继续下一个episode...")

        # 统计结果
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            success_rate = success_count / len(episode_rewards)
            print(f"\n📊 演示结果:")
            print(f"   平均奖励: {avg_reward:.2f}")
            print(f"   成功率: {success_rate:.2%}")
            print(f"   成功次数: {success_count}/{len(episode_rewards)}")

        print("🎉 演示完成！") 