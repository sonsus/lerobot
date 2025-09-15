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
    
@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
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

    # Load the full dataset by not specifying episodes
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)

    # this part for inspection
    import torch 
    import pandas as pd
    # ends here

    try:
        all_episode_indices = sorted(dataset.hf_dataset.unique("episode_index"))
        if not all_episode_indices:
            if len(dataset.hf_dataset) > 0:
                all_episode_indices = [0]
            else:
                log_say("Dataset is empty. Nothing to replay.", cfg.play_sounds)
                env.close()
                return
    except Exception:
        log_say("Could not find 'episode_index' column. Assuming a single episode dataset.", cfg.play_sounds)
        all_episode_indices = [0]

    log_say(f"Found {len(all_episode_indices)} episodes. Starting replay.", cfg.play_sounds, blocking=True)

    for episode_idx in all_episode_indices:
        log_say(f"Replaying episode {episode_idx} in {cfg.env.type} environment.", cfg.play_sounds, blocking=True)

        if "episode_index" in dataset.hf_dataset.column_names:
            episode_actions = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)["action"]
        else:
            # Fallback for datasets without an episode_index column
            episode_actions = dataset.hf_dataset["action"]
        

        def list_round(lst:list, decimals:int):
            return [round(x, decimals) for x in lst]

        # this part for inspection
        episode_actions = torch.stack(episode_actions)
        max_actions = episode_actions.max(dim=0).values
        mean_actions = episode_actions.mean(dim=0)
        med_actions = episode_actions.median(dim=0).values
        min_actions = episode_actions.min(dim=0).values
        episode_states = torch.stack(dataset.hf_dataset["observation.state"])
        max_states = episode_states.max(dim=0).values
        mean_states = episode_states.mean(dim=0)
        med_states = episode_states.median(dim=0).values
        min_states = episode_states.min(dim=0).values
        print()
        print("-"*50)
        print(episode_idx)
        print(f"episode_actions shape: {episode_actions.shape}\n\tmax: {list_round(max_actions.tolist(), 2)}\n\tmin: {list_round(min_actions.tolist(), 2)}")
        print(f"\tmean: {list_round(mean_actions.tolist(), 2)}\n\tmed: {list_round(med_actions.tolist(),2)}")
        print(f"episode_states shape: {episode_states.shape}\n\tmax: {list_round(max_states.tolist(),2)}\n\tmin: {list_round(min_states.tolist(),2)}")
        print(f"\tmean: {list_round(mean_states.tolist(),2)}\n\tmed: {list_round(med_states.tolist(),2)}")
        print("-"*50)
        print()
        

        if episode_idx == max(all_episode_indices):
            return
        continue
        # ends here

        obs, info = env.reset()
        # The HumanRendering wrapper renders on reset.

        for action in episode_actions:
            start_step_t = time.perf_counter()

            # The environment expects a batched action
            action_batch = np.expand_dims(np.array(action), axis=0)

            obs, reward, terminated, truncated, info = env.step(action_batch)
            # The HumanRendering wrapper renders on step.

            if terminated or truncated:
                log_say(f"Episode {episode_idx} finished.", cfg.play_sounds)
                break

            dt_s = time.perf_counter() - start_step_t
            sleep_duration = 1 / cfg.fps - dt_s
            if sleep_duration > 0:
                time.sleep(sleep_duration)
        
        if user_kbd.lower() == 'e':
            return  # Exit the function to end all replays

        # Wait a moment before starting the next episode
        user_kbd = input("to next episode (return), quit (q), skip (s), replay (r), to specific episode (number), run to the end (e):")
        if user_kbd.lower() == 'q':
            log_say("Replay interrupted by user.", cfg.play_sounds)
            break   
        elif user_kbd.lower() == 's':
            log_say("Skipping to the next episode.", cfg.play_sounds)
            continue
        elif user_kbd.lower() == 'r':
            log_say("Replaying the current episode again.", cfg.play_sounds)
            continue
        elif user_kbd.lower() == 'e':
            log_say("Running to the end of all episodes.", cfg.play_sounds)
            all_episode_indices = all_episode_indices[all_episode_indices.index(episode_idx):]
            continue
        elif user_kbd.isdigit():
            episode_to_jump = int(user_kbd) # check if it is number of an episode
            if episode_to_jump in all_episode_indices:
                log_say(f"Jumping to episode {episode_to_jump}.", cfg.play_sounds)
                episode_idx = episode_to_jump - 1  # -1 because it will be incremented in the next loop iteration
                continue
            else:
                log_say(f"Episode {episode_to_jump} does not exist. Continuing with the next episode.", cfg.play_sounds)

    env.close()
    log_say("All replays finished.", cfg.play_sounds)

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
