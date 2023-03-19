import argparse
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer

from env import make_env
from policy import DiffusionSAC
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic


def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--task', type=str, default='AaaS')
    parser.add_argument('--algorithm', type=str, default='diffusion_sac')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('-e', '--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--episode-per-collect', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, default=256)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action='store_true', default=False)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--note', type=str, default='')

    # for diffusion discrete sac
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.05)  # for action entropy
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    parser.add_argument('-t', '--n-timesteps', type=int, default=5)  # for diffusion chain
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])
    parser.add_argument('--pg-coef', type=float, default=1.)

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.6)
    parser.add_argument('--prior-beta', type=float, default=0.4)

    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # create environments
    env, train_envs, test_envs = make_env(args.task, args.training_num, args.test_num)
    args.state_shape = int(np.prod(env.observation_space.shape))
    args.action_shape = env.action_space.n
    args.max_action = 1.
    print(f'Environment Name: {args.task}')
    print(f'Algorithm Name: DiffusionSAC')
    print(f'Shape of Observation Space: {args.state_shape}')
    print(f'Shape of Action Space: {args.action_shape}')

    # seed
    if args.task == 'AaaS':
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_envs.seed(args.seed)
        test_envs.seed(args.seed)

    # create actor
    actor_net = MLP(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        hidden_dim=args.hidden_sizes
    )
    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=actor_net,
        max_action=args.max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps
    ).to(args.device)
    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )

    # create critic
    critic = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        hidden_dim=args.hidden_sizes
    ).to(args.device)
    critic_optim = torch.optim.Adam(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    # log
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(
        args.logdir, args.log_prefix, args.task, args.algorithm, time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    # visualize model graphs
    # dummy_input = torch.randn(1, args.state_shape, device=args.device)
    # writer.add_graph(actor, dummy_input)
    # writer.add_graph(critic, dummy_input)
    logger = TensorboardLogger(writer)

    # policy
    policy = DiffusionSAC(
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        torch.distributions.Categorical,
        args.device,
        alpha=args.alpha,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        pg_coef=args.pg_coef,
        action_space=env.action_space
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # buffer
    if args.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
    else:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )

    # collector
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # trainer
    if not args.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False
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
