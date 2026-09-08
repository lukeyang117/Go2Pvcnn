"""Isolated runner adapter and checkpoint validation for AME."""

from __future__ import annotations

import torch

from ame_baseline.actor_critic_ame import ActorCriticAME
from rsl_rl.runners import on_policy_runner


AME_ARCHITECTURE_SIGNATURE = "ame_xyz_semantic_v1_c6_s16_mha64_h16"


class AmeOnPolicyRunner(on_policy_runner.OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        on_policy_runner.ActorCriticAME = ActorCriticAME
        super().__init__(*args, **kwargs)

    def save(self, path, infos=None):
        super().save(path, infos=infos)
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint["ame_architecture_signature"] = AME_ARCHITECTURE_SIGNATURE
        checkpoint["ame_num_actor_obs"] = self.alg.actor_critic.map_dim + self.alg.actor_critic.actor_state_dim
        checkpoint["ame_num_critic_obs"] = self.alg.actor_critic.map_dim + self.alg.actor_critic.critic_state_dim
        checkpoint["ame_num_actions"] = int(self.alg.actor_critic.std.numel())
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True, keep_std=True):
        checkpoint = torch.load(path, map_location=self.device)
        actual = checkpoint.get("ame_architecture_signature")
        if actual != AME_ARCHITECTURE_SIGNATURE:
            raise ValueError(
                "Checkpoint is not compatible with this AME baseline: "
                f"expected signature {AME_ARCHITECTURE_SIGNATURE!r}, got {actual!r}"
            )
        expected_dims = (
            self.alg.actor_critic.map_dim + self.alg.actor_critic.actor_state_dim,
            self.alg.actor_critic.map_dim + self.alg.actor_critic.critic_state_dim,
            int(self.alg.actor_critic.std.numel()),
        )
        saved_dims = (
            checkpoint.get("ame_num_actor_obs"),
            checkpoint.get("ame_num_critic_obs"),
            checkpoint.get("ame_num_actions"),
        )
        if saved_dims != expected_dims:
            raise ValueError(f"AME checkpoint dimensions {saved_dims} do not match runtime {expected_dims}")
        state_dict = dict(checkpoint["model_state_dict"])
        if not keep_std:
            state_dict.pop("std", None)
        incompatible = self.alg.actor_critic.load_state_dict(state_dict, strict=keep_std)
        if not keep_std and (incompatible.unexpected_keys or incompatible.missing_keys != ["std"]):
            raise RuntimeError(f"Unexpected AME checkpoint incompatibility: {incompatible}")
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_learning_iteration = int(checkpoint["iter"])
        return checkpoint.get("infos")


__all__ = ["AME_ARCHITECTURE_SIGNATURE", "AmeOnPolicyRunner"]
