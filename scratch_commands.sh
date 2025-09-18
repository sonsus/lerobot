


# aloha insertion radian validate (this works)
uv run python examples/check_dataset_radian.py --env.type=aloha --dataset.repo_id=lerobot/aloha_insertion_20250918 > inspection_re.txt


# viz_dataset.sh # local dataset은 이렇게 이름만 적어주자
bash viz_dataset.sh aloha_insertion_20250918 41 
# 41번 에피소드에 문제가 있는걸 확인함. 