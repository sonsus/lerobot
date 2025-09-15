"""
Replays the actions of an episode from a dataset on a simulation environment.

Usage:

python examples/replay_in_sim.py \
    --env.type=pusht \
    --dataset.repo_id=lerobot/pusht \
    --dataset.episode=0
"""

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from pprint import pformat

import draccus
import gymnasium as gym
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.factory import make_env
from lerobot.envs.configs import EnvConfig
from lerobot.utils.utils import init_logging, log_say


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None


@dataclass
class ReplayInSimConfig:
    env: EnvConfig
    dataset: DatasetReplayConfig
    # Frames per second to replay.
    fps: int = 30
    # Use vocal synthesis to read events.
    play_sounds: bool = False


@draccus.wrap()
def replay_in_sim(cfg: ReplayInSimConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Make the environment
    cfg.env.render_mode = "human"
    env = make_env(cfg.env)

    # Load the dataset
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    
    actions = dataset.hf_dataset.select_columns("action")[:]["action"] # 에피소드 단위로 보통 가져오는거긴해서 익숙해보이진 않음

    df = dataset.hf_dataset.to_pandas()
    print()

    log_say(f"Replaying episode {cfg.dataset.episode} in {cfg.env.type} environment.", cfg.play_sounds, blocking=True)

    obs, info = env.reset()
    # The HumanRendering wrapper renders on reset.

    for action in actions:
        start_step_t = time.perf_counter()

        # The environment expects a batched action
        action_batch = np.expand_dims(np.array(action), axis=0)

        obs, reward, terminated, truncated, info = env.step(action_batch)
        # The HumanRendering wrapper renders on step.

        if terminated or truncated:
            log_say("Episode finished.", cfg.play_sounds)
            break

        dt_s = time.perf_counter() - start_step_t
        sleep_duration = 1 / cfg.fps - dt_s
        if sleep_duration > 0:
            time.sleep(sleep_duration)

    env.close()
    log_say("Replay finished.", cfg.play_sounds)


if __name__ == "__main__":
    replay_in_sim()
