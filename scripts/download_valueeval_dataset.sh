#!/bin/bash
set -euo pipefail

TARGET_DIR=${1:-$HOME/data/valueeval}
BASE_URL="https://zenodo.org/records/10564870/files"

FILES=(
  arguments-training.tsv
  arguments-validation.tsv
  arguments-validation-zhihu.tsv
  arguments-test.tsv
  arguments-test-nahjalbalagha.tsv
  labels-training.tsv
  labels-validation.tsv
  labels-validation-zhihu.tsv
  labels-test.tsv
  labels-test-nyt.tsv
  labels-test-nahjalbalagha.tsv
  value-categories.json
  meta-arguments-c.tsv
  meta-arguments-e.tsv
  meta-arguments-f.tsv
  meta-arguments-g.tsv
  level1-labels-training.tsv
  level1-labels-validation.tsv
  level1-labels-test.tsv
  level1-labels-test-nyt.tsv
  level1-labels-test-nahjalbalagha.tsv
)

mkdir -p "$TARGET_DIR"

for file in "${FILES[@]}"; do
  dest="$TARGET_DIR/$file"
  if [[ -f "$dest" ]]; then
    echo "Skipping $file (already exists)."
    continue
  fi
  echo "Downloading $file"
  curl -L "$BASE_URL/$file" -o "$dest"
done

echo "Touché23-ValueEval files saved to $TARGET_DIR"
