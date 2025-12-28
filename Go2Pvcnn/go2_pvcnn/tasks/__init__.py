"""Tasks for Go2 robot with PVCNN."""

from .go2_pvcnn_env_cfg import Go2PvcnnEnvCfg, Go2PvcnnEnvCfg_PLAY

# Register environments
import go2_pvcnn.tasks.register_envs  # noqa: F401

__all__ = ["Go2PvcnnEnvCfg", "Go2PvcnnEnvCfg_PLAY"]
