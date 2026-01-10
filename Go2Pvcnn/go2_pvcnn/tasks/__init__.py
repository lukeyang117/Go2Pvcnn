"""Tasks for Go2 robot with PVCNN."""

from .go2_pvcnn_env_cfg import Go2PvcnnEnvCfg, Go2PvcnnEnvCfg_PLAY
from .go2_pvcnn_test_env_cfg import Go2PvcnnTestEnvCfg
from .go2_pvcnn_furniture_test_env_cfg import Go2PvcnnFurnitureTestEnvCfg

# Register environments
import go2_pvcnn.tasks.register_envs  # noqa: F401

__all__ = ["Go2PvcnnEnvCfg", "Go2PvcnnEnvCfg_PLAY", "Go2PvcnnTestEnvCfg", "Go2PvcnnFurnitureTestEnvCfg"]
