import argparse
import os
import sys
import pprint
import torch
import numpy as np
from os import path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from env import make_env
from policy import RandomPolicy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='AaaS')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--episode-per-collect', type=int, default=1)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # create environment
    env, train_envs, test_envs = make_env(args.task, args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # policy
    policy = RandomPolicy(
        torch.distributions.Categorical,
        action_space=env.action_space,
        action_scaling=False,
        action_bound_method="",
    )

    # collector
    train_collector = Collector(policy, train_envs)
    test_collector = Collector(policy, test_envs)

    # log
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    root = path.dirname(path.dirname(path.abspath(__file__)))
    log_path = os.path.join(
        root, args.logdir, args.log_prefix, args.task, 'random', time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # trainer
    if not args.watch:
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            episode_per_collect=args.episode_per_collect,
            logger=logger
        )
        pprint.pprint(result)

    # Watch the performance
    if __name__ == '__main__':
        np.random.seed(args.seed)
        env, _, _ = make_env(args.task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())
