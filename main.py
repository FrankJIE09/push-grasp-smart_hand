"""
螺丝推动强化学习主程序
"""

import os
import argparse
from training.trainer import ScrewPushingTrainer
from evaluation.evaluator import ScrewPushingEvaluator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='螺丝推动强化学习训练和评估')
    parser.add_argument('--mode', type=str, default='demonstrate',
                       choices=['train', 'evaluate', 'demonstrate'],
                       help='运行模式: train(训练), evaluate(评估), demonstrate(演示)')
    parser.add_argument('--model_path', type=str, default='screw_pushing_agent',
                       help='模型文件路径')
    parser.add_argument('--total_timesteps', type=int, default=1,
                       help='训练总步数')
    parser.add_argument('--num_episodes', type=int, default=1,
                       help='评估/演示的episode数量')
    parser.add_argument('--with_viewer', action='store_true', 
                       help='评估时是否弹出MuJoCo viewer')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 开始训练模式...")
        trainer = ScrewPushingTrainer(
            total_timesteps=args.total_timesteps,
            model_save_path=args.model_path
        )
        trainer.train()
        
    elif args.mode == 'evaluate':
        print("🔍 开始评估模式...")
        evaluator = ScrewPushingEvaluator(model_path=args.model_path)
        evaluator.evaluate_multiple_episodes(num_episodes=args.num_episodes, with_viewer=args.with_viewer)
        
    elif args.mode == 'demonstrate':
        print("🎮 开始演示模式...")
        evaluator = ScrewPushingEvaluator(model_path=args.model_path)
        evaluator.demonstrate_with_viewer(num_episodes=args.num_episodes)
    
    else:
        print("❌ 无效的运行模式")


if __name__ == "__main__":
    main() 