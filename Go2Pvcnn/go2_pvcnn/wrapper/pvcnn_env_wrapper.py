"""Wrapper for Isaac Lab environments to work with RSL-RL PPO algorithm with PVCNN integration."""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from ..pvcnn_wrapper import PVCNNWrapper


class RslRlPvcnnEnvWrapper(VecEnv):
    """Wraps Isaac Lab environment for RSL-RL with PVCNN integration.
    
    This wrapper extends the base VecEnv interface and integrates PVCNN for LiDAR-based observation.
    Based on Isaac Lab's RslRlVecEnvWrapper but using local rsl_rl package.
    
    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        pvcnn_wrapper: PVCNNWrapper,
        clip_actions: float | None = None,
    ):
        """Initialize the wrapper.
        
        Note:
            The wrapper calls reset at the start since the RSL-RL runner does not call reset.
        
        Args:
            env: The Isaac Lab environment to wrap.
            pvcnn_wrapper: Pre-initialized PVCNN wrapper for feature extraction.
            clip_actions: The clipping value for actions. If None, no clipping is done.
        
        Raises:
            ValueError: If environment is not an instance of ManagerBasedRLEnv or DirectRLEnv.
        """
        # Check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        # Store environment and settings
        self.env = env
        self.clip_actions = clip_actions
        self.pvcnn_wrapper = pvcnn_wrapper

        # Store information required by VecEnv interface
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # Obtain action dimension
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # Modify the action space to the clip range
        self._modify_action_space()

        # Inject PVCNN wrapper into environment BEFORE reset
        # This is critical because observation functions need the wrapper during reset
        self._inject_pvcnn_wrapper()

        # Reset environment (required by RSL-RL runner)
        # This will call observation functions which need pvcnn_wrapper
        print(f"[RslRlPvcnnEnvWrapper] Resetting environment (this will compute observations)...")
        self.env.reset()
        print(f"[RslRlPvcnnEnvWrapper] Environment reset complete")

    def _inject_pvcnn_wrapper(self):
        """Inject PVCNN wrapper into the unwrapped environment."""
        # Store in unwrapped environment so observation functions can access it
        print(f"\n[RslRlPvcnnEnvWrapper] Attempting to inject PVCNN wrapper...")
        print(f"  - env type: {type(self.env)}")
        print(f"  - unwrapped type: {type(self.unwrapped)}")
        print(f"  - has observation_manager: {hasattr(self.unwrapped, 'observation_manager')}")
        
        if hasattr(self.unwrapped, "observation_manager"):
            self.unwrapped.pvcnn_wrapper = self.pvcnn_wrapper
            print(f"  ✓ Injected PVCNN wrapper into env.unwrapped.pvcnn_wrapper")
            print(f"  - Wrapper device: {self.pvcnn_wrapper.device}")
            print(f"  - Wrapper feature_dim: {self.pvcnn_wrapper.get_feature_dim()}")
            
            # Verify injection
            if hasattr(self.unwrapped, 'pvcnn_wrapper'):
                print(f"  ✓ Verification: env.unwrapped.pvcnn_wrapper exists = {self.unwrapped.pvcnn_wrapper is not None}")
            else:
                print(f"  ✗ Verification FAILED: attribute not found after injection!")
        else:
            print(f"  ✗ ERROR: Environment does not have observation_manager!")
            raise RuntimeError("Environment must have observation_manager to use PVCNN wrapper")

    def __str__(self):
        """Returns the wrapper name and environment representation."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the render mode."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the observation space."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the action space."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.
        
        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:
        """Set the seed for the environment."""
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment.
        
        Returns:
            A tuple of (obs, extras) compatible with local rsl_rl.
        """
        obs_dict, extras = self.env.reset()
        # Return policy observations and extras with all observation groups
        extras["observations"] = obs_dict
        return obs_dict["policy"], extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment.
        
        Returns:
            A tuple of (obs, extras) where:
            - obs: policy observations as tensor (num_envs, obs_dim)
            - extras: dict with "observations" key containing all observation groups
        """
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        
        # Return format compatible with local rsl_rl (2.0.x)
        # obs is the policy observations, extras["observations"] contains all groups
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment.
        
        Returns:
            A tuple of (obs, rew, dones, extras) compatible with local rsl_rl.
        """
        # Clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # Record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # Compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # Move time out information to the extras dict (for infinite horizon tasks)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # Add all observation groups to extras
        extras["observations"] = obs_dict
        # Return policy observations and other info
        return obs_dict["policy"], rew, dones, extras

    def close(self):
        """Close the environment."""
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # Modify the action space to the clip range
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

