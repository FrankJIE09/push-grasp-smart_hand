"""
强化学习训练器
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment.screw_pushing_env import ScrewPushingEnv


class ScrewPushingTrainer:
    """螺丝推动强化学习训练器"""

    def __init__(self, 
                 total_timesteps=10000,
                 save_freq=1000,
                 eval_freq=500,
                 model_save_path="screw_pushing_agent"):
        self.total_timesteps = total_timesteps
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.model_save_path = model_save_path

    def create_env(self):
        """创建训练环境"""
        def make_env():
            env = ScrewPushingEnv()
            env = Monitor(env)
            return env

        return DummyVecEnv([make_env])

    def create_eval_env(self):
        """创建评估环境"""
        def make_eval_env():
            env = ScrewPushingEnv()
            env = Monitor(env)
            return env

        return DummyVecEnv([make_eval_env])

    def train(self):
        """训练智能体"""
        print("🚀 开始训练螺丝推动智能体...")
        
        # 创建环境
        env = self.create_env()
        eval_env = self.create_eval_env()

        # 创建模型
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./logs/"
        )

        # 设置回调函数
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path="./checkpoints/",
            name_prefix="screw_pushing_model"
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./best_model/",
            log_path="./logs/",
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False
        )

        # 开始训练
        print(f"📊 训练参数:")
        print(f"   总步数: {self.total_timesteps}")
        print(f"   保存频率: {self.save_freq}")
        print(f"   评估频率: {self.eval_freq}")
        print(f"   模型保存路径: {self.model_save_path}")

        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )

        # 保存最终模型
        model.save(self.model_save_path)
        print(f"✅ 训练完成！模型已保存到: {self.model_save_path}")

        return model

    def load_model(self, model_path):
        """加载训练好的模型"""
        if os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path)
            print(f"✅ 模型加载成功: {model_path}")
            return model
        else:
            print(f"❌ 模型文件不存在: {model_path}")
            return None 