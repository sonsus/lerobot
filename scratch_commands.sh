

# pro pencil inspection
# doesn't work (env-data compatibility differs from below)
uv run python examples/replay_in_sim.py --env.type=aloha --dataset=lerobot/aloha_static_pro_pencil  > inspect_pro_pencil_log.txt  


# aloha insertion radian validate (this works)
uv run python examples/replay_in_sim.py --env.type=aloha --dataset.repo_id=lerobot/aloha_insertion_20250604_radian > inspection_re.txt




# viz_dataset.sh
bash viz_dataset.sh aloha_static_pro_pencil 0
