import subprocess as sb
i = 0
while True:
    cmd = f"uv run python examples/replay_in_sim.py --env.type=aloha --dataset.repo_id=lerobot/aloha_insertion_20250604_radian --dataset.episode={i}"
    sb.run(cmd, shell=True)
    i+=1
    