"""Isolated runner adapter and checkpoint validation for AME."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from m1_ame.m1_actor_critic_ame import M1ActorCriticAME
from m1_ame import M1_AME_ACTION_DIM, M1_AME_CRITIC_OBS_DIM, M1_AME_POLICY_OBS_DIM
from rsl_rl.runners import on_policy_runner


M1_AME_ARCHITECTURE_SIGNATURE = "m1_ame_xyz_semantic_v1_c6_s16_mha64_h16"


class M1AmeOnPolicyRunner(on_policy_runner.OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        on_policy_runner.ActorCriticAME = M1ActorCriticAME
        super().__init__(*args, **kwargs)
        self.source_metadata = None

    def save(self, path, infos=None):
        super().save(path, infos=infos)
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint["ame_architecture_signature"] = M1_AME_ARCHITECTURE_SIGNATURE
        checkpoint["ame_num_actor_obs"] = self.alg.actor_critic.map_dim + self.alg.actor_critic.actor_state_dim
        checkpoint["ame_num_critic_obs"] = self.alg.actor_critic.map_dim + self.alg.actor_critic.critic_state_dim
        checkpoint["ame_num_actions"] = int(self.alg.actor_critic.std.numel())
        from extension.semantic_curriculum import capture_terrain_curriculum_state

        curriculum_state = capture_terrain_curriculum_state(getattr(self, "env", None))
        if curriculum_state is not None:
            checkpoint["ame_curriculum_state"] = curriculum_state
        if getattr(self, "source_metadata", None) is not None:
            checkpoint["ame_continuation_source"] = self.source_metadata
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True, keep_std=True):
        checkpoint = torch.load(path, map_location=self.device)
        actual = checkpoint.get("ame_architecture_signature")
        if actual != M1_AME_ARCHITECTURE_SIGNATURE:
            raise ValueError(
                "Checkpoint is not compatible with this AME baseline: "
                f"expected signature {M1_AME_ARCHITECTURE_SIGNATURE!r}, got {actual!r}"
            )
        expected_dims = (M1_AME_POLICY_OBS_DIM, M1_AME_CRITIC_OBS_DIM, M1_AME_ACTION_DIM)
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
            self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]["lr"])
        # Saved `iter` is the zero-based index of the update already completed.
        self.current_learning_iteration = int(checkpoint["iter"]) + 1
        checkpoint_path = Path(path).expanduser().resolve()
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.source_metadata = {
            "checkpoint": str(checkpoint_path),
            "sha256": digest.hexdigest(),
            "source_iter": int(checkpoint["iter"]),
            "first_new_iteration": self.current_learning_iteration,
            "restored_learning_rate": float(self.alg.learning_rate),
        }
        if getattr(self, "env", None) is not None:
            from extension.semantic_curriculum import restore_terrain_curriculum_state

            mode = restore_terrain_curriculum_state(self.env, checkpoint.get("ame_curriculum_state"))
            self.source_metadata["curriculum_restore"] = mode
        return checkpoint.get("infos")

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        if self.writer is None:
            return
        root = getattr(self.env, "unwrapped", self.env)
        manager = getattr(root, "curriculum_manager", None)
        states = getattr(manager, "_curriculum_state", {}) if manager is not None else {}
        for term_state in states.values():
            if not isinstance(term_state, dict):
                continue
            for key, value in term_state.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.writer.add_scalar(f"Curriculum/terrain_levels/{key}", float(value.item()), locs["it"])


__all__ = ["M1_AME_ARCHITECTURE_SIGNATURE", "M1AmeOnPolicyRunner"]
