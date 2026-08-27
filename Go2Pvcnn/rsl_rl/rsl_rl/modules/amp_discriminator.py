"""AMP discriminator and masked running observation normalization."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AMPObservationNormalizer(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1.0e-5):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(int(input_dim)))
        self.register_buffer("var", torch.ones(int(input_dim)))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, values: Tensor, active: Tensor | None = None) -> None:
        values = torch.as_tensor(values, device=self.mean.device, dtype=torch.float32).reshape(-1, self.mean.numel())
        if active is not None:
            mask = torch.as_tensor(active, device=values.device, dtype=torch.bool).reshape(-1)
            if mask.numel() != values.shape[0]:
                raise ValueError("normalizer mask must match the flattened batch")
            values = values[mask]
        if values.numel() == 0:
            return
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        old_count = self.count.to(dtype=values.dtype)
        total = old_count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total)
        old_m2 = self.var * old_count
        batch_m2 = batch_var * batch_count
        new_m2 = old_m2 + batch_m2 + delta.square() * old_count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(new_m2 / total.clamp_min(1.0))
        self.count.copy_(total.to(dtype=torch.long))

    def normalize(self, values: Tensor) -> Tensor:
        values = torch.as_tensor(values, device=self.mean.device, dtype=torch.float32)
        return (values - self.mean) / torch.sqrt(self.var + self.eps)


class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int = 936,
        hidden_dims: tuple[int, ...] = (1024, 512, 256),
        learning_rate: float = 1.0e-4,
        grad_penalty: float = 5.0,
        logit_reg: float = 0.01,
        reward_scale: float = 2.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current = int(input_dim)
        for hidden in hidden_dims:
            layers.extend((nn.Linear(current, int(hidden)), nn.LeakyReLU(0.2)))
            current = int(hidden)
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)
        self.normalizer = AMPObservationNormalizer(int(input_dim))
        self.grad_penalty = float(grad_penalty)
        self.logit_reg = float(logit_reg)
        self.reward_scale = float(reward_scale)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))

    def forward(self, windows: Tensor) -> Tensor:
        values = torch.as_tensor(windows, device=self.mean_device, dtype=torch.float32).reshape(windows.shape[0], -1)
        return self.network(values).squeeze(-1)

    @property
    def mean_device(self):
        return self.normalizer.mean.device

    def reward(self, agent_windows: Tensor, active: Tensor | None = None) -> Tensor:
        values = torch.as_tensor(agent_windows, device=self.mean_device, dtype=torch.float32)
        mask = torch.ones(values.shape[0], dtype=torch.bool, device=values.device) if active is None else torch.as_tensor(active, device=values.device, dtype=torch.bool).reshape(-1)
        with torch.no_grad():
            flat = values.reshape(values.shape[0], -1)
            logits = self(self.normalizer.normalize(flat))
            rewards = -F.logsigmoid(-logits) * self.reward_scale
        return torch.where(mask, rewards, torch.zeros_like(rewards))

    def update(self, expert_windows: Tensor, agent_windows: Tensor, active: Tensor) -> dict[str, float]:
        expert = torch.as_tensor(expert_windows, device=self.mean_device, dtype=torch.float32).reshape(expert_windows.shape[0], -1)
        agent = torch.as_tensor(agent_windows, device=self.mean_device, dtype=torch.float32).reshape(agent_windows.shape[0], -1)
        mask = torch.as_tensor(active, device=expert.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != expert.shape[0] or mask.numel() != agent.shape[0]:
            raise ValueError("AMP active mask must match expert and agent batches")
        if not bool(mask.any()):
            return {"loss": 0.0, "gradient_penalty": 0.0, "expert_accuracy": 0.0, "agent_accuracy": 0.0}
        self.normalizer.update(torch.cat((expert, agent), dim=0), torch.cat((mask, mask), dim=0))
        expert = self.normalizer.normalize(expert[mask]).detach().requires_grad_(True)
        agent = self.normalizer.normalize(agent[mask]).detach().requires_grad_(True)
        expert_logits = self.network(expert).squeeze(-1)
        agent_logits = self.network(agent).squeeze(-1)
        labels_expert = torch.ones_like(expert_logits)
        labels_agent = torch.zeros_like(agent_logits)
        bce = F.binary_cross_entropy_with_logits(expert_logits, labels_expert) + F.binary_cross_entropy_with_logits(agent_logits, labels_agent)
        grad_expert = torch.autograd.grad(expert_logits.sum(), expert, create_graph=True)[0]
        grad_agent = torch.autograd.grad(agent_logits.sum(), agent, create_graph=True)[0]
        gradient_penalty = 0.5 * (grad_expert.square().sum(dim=-1).mean() + grad_agent.square().sum(dim=-1).mean())
        logit_reg = 0.5 * (expert_logits.square().mean() + agent_logits.square().mean())
        loss = bce + self.grad_penalty * gradient_penalty + self.logit_reg * logit_reg
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()
        with torch.no_grad():
            expert_accuracy = (expert_logits > 0).float().mean()
            agent_accuracy = (agent_logits < 0).float().mean()
        return {
            "loss": float(loss.detach()),
            "gradient_penalty": float(gradient_penalty.detach()),
            "expert_accuracy": float(expert_accuracy),
            "agent_accuracy": float(agent_accuracy),
        }
