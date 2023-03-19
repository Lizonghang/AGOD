import torch
import numpy as np
from torch.nn.functional import one_hot
from tianshou.policy import BasePolicy
from tianshou.data import Batch
from typing import Any, Dict, List, Type, Optional, Union

from env.config import NUM_SERVICE_PROVIDERS


class RoundRobinPolicy(BasePolicy):
    """Implementation of round robin policy. This policy assign user tasks to service
    providers in turn.

    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    """

    def __init__(
            self,
            dist_fn: Type[torch.distributions.Distribution],
            action_scaling: bool = True,
            action_bound_method: str = "clip",
            **kwargs: Any
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs)
        self.dist_fn = dist_fn
        self.round_act = 0

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any
    ) -> Batch:
        """Compute action at random."""
        act_ = (self.round_act % NUM_SERVICE_PROVIDERS)
        act_ = torch.Tensor([act_] * batch.obs.shape[0]).to(torch.int64)
        logits, hidden = one_hot(act_, num_classes=NUM_SERVICE_PROVIDERS), None
        self.round_act += 1

        # convert to probability distribution
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)

        # use deterministic policy
        if self.action_type == "discrete":
            act = logits.argmax(-1)
        elif self.action_type == "continuous":
            act = logits[0]

        assert act.equal(act_), f"Action mismatch: {act_} != {act}"

        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def learn(
            self,
            batch: Batch,
            batch_size: int,
            repeat: int,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        return {"loss": [0.]}
