#!/bin/bash

BASE_DIR="$HOME/.cache/huggingface/lerobot/lerobot"

if [ ! -d "$BASE_DIR" ]; then
  echo "Base directory $BASE_DIR does not exist."
  exit 1
fi

echo "Checking for available LeRobot datasets in $BASE_DIR..."
echo "-----------------------------------------------------"

found_datasets=0

for dataset_dir in "$BASE_DIR"/*; do
  # Check if the entry is a directory or a symbolic link to a directory.
  # The -d operator follows symbolic links.
  if [ -d "$dataset_dir" ]; then
    DATA_PATH="$dataset_dir/data"
    META_PATH="$dataset_dir/meta"
    # README_PATH="$dataset_dir/README.md"
    VIDEOS_PATH="$dataset_dir/videos"

    # Check for existence and type of required components.
    # The -d and -f operators also follow symbolic links.
    if [ -d "$DATA_PATH" ] && \
       [ -d "$META_PATH" ] && \
       [ -d "$VIDEOS_PATH" ]; then
      #  [ -f "$README_PATH" ] && \
      echo "$(basename "$dataset_dir")"
      found_datasets=$((found_datasets+1))
    fi
  fi
done

if [ "$found_datasets" -eq 0 ]; then
  echo "No valid LeRobot datasets found."

fi
echo "-----------------------------------------------------"
