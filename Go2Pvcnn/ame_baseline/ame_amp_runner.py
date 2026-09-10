"""Runner adapter for AME legacy warm-start and complete AMP resume."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from rsl_rl.modules import AMPDiscriminator
from rsl_rl.runners import on_policy_runner

from .amp_actor_critic_ame import AmpActorCriticAME
from .ame_runner import AME_ARCHITECTURE_SIGNATURE


AME_AMP_ARCHITECTURE_SIGNATURE = "ame_xyz_semantic_v1_c6_s16_mha64_h16_amp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AmeAmpOnPolicyRunner(on_policy_runner.OnPolicyRunner):
    """Use the existing AMP PPO loop with an AME-compatible policy class."""

    def __init__(self, *args, **kwargs):
        on_policy_runner.AmpActorCriticAME = AmpActorCriticAME
        super().__init__(*args, **kwargs)
        self.logger_type = "tensorboard"
        self.source_metadata: dict[str, Any] = {}

    @staticmethod
    def _validate_state_finite(state_dict: dict[str, torch.Tensor]) -> None:
        invalid = [key for key, value in state_dict.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
        if invalid:
            raise ValueError("Checkpoint contains non-finite tensors: " + ",".join(invalid))

    def _validate_ame_metadata(self, checkpoint: dict[str, Any]) -> None:
        signature = checkpoint.get("ame_architecture_signature")
        if signature not in (None, AME_ARCHITECTURE_SIGNATURE, AME_AMP_ARCHITECTURE_SIGNATURE):
            raise ValueError(f"Unsupported AME architecture signature: {signature!r}")
        expected = (
            self.alg.actor_critic.map_dim + self.alg.actor_critic.actor_state_dim,
            self.alg.actor_critic.map_dim + self.alg.actor_critic.critic_state_dim,
            int(self.alg.actor_critic.std.numel()),
        )
        saved = (
            checkpoint.get("ame_num_actor_obs", expected[0]),
            checkpoint.get("ame_num_critic_obs", expected[1]),
            checkpoint.get("ame_num_actions", expected[2]),
        )
        if tuple(saved) != expected:
            raise ValueError(f"AME checkpoint dimensions {saved} do not match runtime {expected}")

    def load_amp_checkpoint(self, path: str | Path, *, keep_std: bool = True) -> str:
        """Load either a legacy AME checkpoint or a complete AME-AMP checkpoint."""

        checkpoint_path = Path(path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"AME-AMP checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint must be a dictionary containing model_state_dict")
        state_dict = dict(checkpoint["model_state_dict"])
        has_amp_head = any(key.startswith("amp_value_head.") for key in state_dict)
        has_discriminator = "amp_discriminator_state_dict" in checkpoint
        if has_amp_head != has_discriminator:
            raise RuntimeError("Incomplete AMP checkpoint: amp_value_head and discriminator must both be present")
        self._validate_ame_metadata(checkpoint)
        self._validate_state_finite(state_dict)

        if not has_amp_head:
            if not keep_std:
                state_dict.pop("std", None)
                state_dict["std"] = self.alg.actor_critic.state_dict()["std"]
            self.alg.actor_critic.load_common_state_dict(state_dict)
            self.current_learning_iteration = 0
            self.source_metadata = {
                "ame_source_checkpoint": str(checkpoint_path),
                "ame_source_sha256": _sha256(checkpoint_path),
                "ame_source_iteration": int(checkpoint.get("iter", 0)),
            }
            print("[Checkpoint] checkpoint_mode=legacy_ame_warm_start", flush=True)
            print("[Checkpoint] amp_value_head=initialized, discriminator=initialized", flush=True)
            return "legacy_policy_warm_start"

        self.alg.actor_critic.load_state_dict(state_dict, strict=True)
        self.alg.amp_discriminator.load_state_dict(checkpoint["amp_discriminator_state_dict"], strict=True)
        if "optimizer_state_dict" in checkpoint:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "amp_optimizer_state_dict" in checkpoint:
            self.alg.amp_discriminator.optimizer.load_state_dict(checkpoint["amp_optimizer_state_dict"])
        self.current_learning_iteration = int(checkpoint.get("iter", 0))
        self.source_metadata = dict(checkpoint.get("ame_source_metadata", {}))
        if not self.source_metadata:
            self.source_metadata = {
                key: checkpoint[key]
                for key in ("ame_source_checkpoint", "ame_source_sha256", "ame_source_iteration")
                if key in checkpoint
            }
        print(f"[Checkpoint] checkpoint_mode=full_ame_amp_resume iter={self.current_learning_iteration}", flush=True)
        return "full_amp_resume"

    def save(self, path, infos=None):
        super().save(path, infos=infos)
        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["ame_amp_signature"] = AME_AMP_ARCHITECTURE_SIGNATURE
        checkpoint["ame_architecture_signature"] = AME_AMP_ARCHITECTURE_SIGNATURE
        checkpoint["ame_num_actor_obs"] = self.alg.actor_critic.map_dim + self.alg.actor_critic.actor_state_dim
        checkpoint["ame_num_critic_obs"] = self.alg.actor_critic.map_dim + self.alg.actor_critic.critic_state_dim
        checkpoint["ame_num_actions"] = int(self.alg.actor_critic.std.numel())
        checkpoint["ame_source_metadata"] = dict(self.source_metadata)
        for key, value in self.source_metadata.items():
            checkpoint[key] = value
        torch.save(checkpoint, checkpoint_path)


__all__ = ["AME_AMP_ARCHITECTURE_SIGNATURE", "AmeAmpOnPolicyRunner"]
