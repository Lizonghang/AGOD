import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR


class DiffusionSAC(BasePolicy):
    """
    Implementation of diffusion-based discrete soft actor-critic policy.
    """

    def __init__(
            self,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            dist_fn: Type[torch.distributions.Distribution],
            device: torch.device,
            alpha: float = 0.05,
            tau: float = 0.005,
            gamma: float = 0.95,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            pg_coef: float = 1.,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= alpha <= 1.0, "alpha should be in [0, 1]"
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor
            self._target_actor = deepcopy(actor)
            self._target_actor.eval()
            self._actor_optim: torch.optim.Optimizer = actor_optim
            self._action_dim = action_dim

        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic
            self._target_critic = deepcopy(critic)
            self._target_critic.eval()
            self._critic_optim: torch.optim.Optimizer = critic_optim

        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(
                self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(
                self._critic_optim, T_max=lr_maxt, eta_min=0.)

        self._dist_fn = dist_fn
        self._alpha = alpha
        self._tau = tau
        self._gamma = gamma
        self._rew_norm = reward_normalization
        self._n_step = estimation_step
        self._lr_decay = lr_decay
        self._pg_coef = pg_coef
        self._device = device

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        obs_next_ = torch.FloatTensor(batch.obs_next).to(self._device)
        dist = self.forward(
            batch, input="obs_next", model="target_actor").dist
        target_q = dist.probs * self._target_critic.q_min(obs_next_)
        return target_q.sum(dim=-1)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        return self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm
        )

    def update(
            self,
            sample_size: int,
            buffer: Optional[ReplayBuffer],
            **kwargs: Any
    ) -> Dict[str, Any]:
        if buffer is None: return {}
        self.updating = True
        # sample from replay buffer
        batch, indices = buffer.sample(sample_size)
        # calculate n_step returns
        batch = self.process_fn(batch, buffer, indices)
        # update network parameters
        result = self.learn(batch, **kwargs)
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        self.updating = False
        return result

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        model_ = self._actor if model == "actor" else self._target_actor
        logits, hidden = model_(obs_), None
        dist = self._dist_fn(logits)
        acts = dist.sample() if self.training else logits.argmax(axis=-1)
        return Batch(logits=logits, act=acts, state=hidden, dist=dist)

    def _to_one_hot(
            self,
            data: np.ndarray,
            one_hot_dim: int
    ) -> np.ndarray:
        batch_size = data.shape[0]
        one_hot_codes = np.eye(one_hot_dim)
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim))
                       for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)

    def _update_critic(self, batch: Batch) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(batch.act[:, np.newaxis], device=self._device, dtype=torch.long)
        target_q = batch.returns
        current_q1, current_q2 = self._critic(obs_)
        critic_loss = F.mse_loss(current_q1.gather(1, acts_), target_q) \
                      + F.mse_loss(current_q2.gather(1, acts_), target_q)
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()
        return critic_loss

    def _update_bc(self, batch: Batch, update: bool = False) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = self._to_one_hot(batch.act, self._action_dim)
        acts_ = to_torch(acts_, device=self._device, dtype=torch.float32)
        bc_loss = self._actor.loss(acts_, obs_).mean()
        if update:
            self._actor_optim.zero_grad()
            bc_loss.backward()
            self._actor_optim.step()
        return bc_loss

    def _update_policy(self, batch: Batch, update: bool = False) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        dist = self.forward(batch).dist
        entropy = dist.entropy()
        with torch.no_grad():
            q = self._critic.q_min(obs_)
        pg_loss = -(self._alpha * entropy + (dist.probs * q).sum(dim=-1)).mean()
        if update:
            self._actor_optim.zero_grad()
            pg_loss.backward()
            self._actor_optim.step()
        return pg_loss

    def _update_targets(self):
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    def learn(
            self,
            batch: Batch,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        # update critic network
        critic_loss = self._update_critic(batch)
        # update actor network
        pg_loss = self._update_policy(batch, update=False)
        bc_loss = self._update_bc(batch, update=False) if self._pg_coef < 1. else 0.
        overall_loss = self._pg_coef * pg_loss + (1 - self._pg_coef) * bc_loss
        self._actor_optim.zero_grad()
        overall_loss.backward()
        self._actor_optim.step()
        # update target networks
        self._update_targets()
        return {
            'loss/critic': critic_loss.item(),
            'overall_loss': overall_loss.item()
        }
