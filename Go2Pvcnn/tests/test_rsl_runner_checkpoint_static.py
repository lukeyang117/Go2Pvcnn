from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py"


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_runner_load_defaults_to_resetting_checkpoint_std() -> None:
    source = _source()

    assert "def load(self, path, load_optimizer=True, keep_std=False):" in source
    assert "if not keep_std:" in source
    assert 'state_dict.pop("std", None)' in source


def test_runner_load_can_keep_checkpoint_std_when_requested() -> None:
    source = _source()

    assert source.index("if not keep_std:") < source.index('state_dict.pop("std", None)')
    assert source.index('state_dict.pop("std", None)') < source.index(
        "self.alg.actor_critic.load_state_dict(state_dict, strict=False)"
    )
