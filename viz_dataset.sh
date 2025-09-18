#!/bin/bash
repo_id=$1
ep_idx=$2

BASE_DIR="$HOME/.cache/huggingface/lerobot/lerobot"

# Function to check if a directory is a valid LeRobot dataset
is_valid_dataset() {
  local dataset_path=$1
  if [ -d "$dataset_path" ]; then
    if [ -d "$dataset_path/data" ] && \
       [ -d "$dataset_path/meta" ] && \
       [ -d "$dataset_path/videos" ]; then
      return 0 # True, it's a valid dataset
    fi
  fi
  return 1 # False, not a valid dataset
}

# Try the given repo_id as is
if is_valid_dataset "$repo_id"; then
  actual_repo_id="$repo_id"
elif is_valid_dataset "$BASE_DIR/$repo_id"; then
  # If not found, try prepending the base directory
  actual_repo_id="$BASE_DIR/$repo_id"
else
  echo "Error: Dataset '$repo_id' not found in current path or in $BASE_DIR."
  exit 1
fi


# ffmpeg path for mac (trying to manually fix the linking makes security issues, adding env variable is simpler)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib uv run python -m lerobot.scripts.visualize_dataset --repo-id "$actual_repo_id" --episode-index "$ep_idx"


