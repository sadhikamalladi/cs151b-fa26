#!/usr/bin/env bash
# Run the autograder on one submission folder in a Gradescope-like layout:
#   <tmp>/autograder/{source,submission,results}
#
#   bash autograder/tests/run_local.sh path/to/submission [autograder.zip]
#
# Without a zip, the source is assembled from this checkout (data/ is generated on first use
# into autograder/tests/work/source_cache). Uses $AG_PYTHON (default: python3 on PATH), which
# needs the packages in autograder/requirements.txt plus torch. Extra grade.py flags can be
# passed through AG_FLAGS (e.g. AG_FLAGS=--skip-code). Set AUTOGRADER_DRIVE_MOCK to use the
# mock Drive from make_examples.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AG="$(dirname "$HERE")"
REPO="$(dirname "$AG")"
SUB="$(realpath "$1")"
ZIP="${2:-}"
export AG_PYTHON="${AG_PYTHON:-$(command -v python3)}"

ROOT="$(mktemp -d)/autograder"
mkdir -p "$ROOT/source" "$ROOT/results"
cp -r "$SUB" "$ROOT/submission"
if [ -n "$ZIP" ]; then
  unzip -q "$ZIP" -d "$ROOT/source"
else
  CACHE="$HERE/work/source_cache"
  if [ ! -f "$CACHE/data/instruct_train.json.gz" ]; then
    mkdir -p "$CACHE"
    "$AG_PYTHON" "$AG/grader/prepare_data.py" "$CACHE"
  fi
  mkdir -p "$ROOT/source/grader"
  cp "$AG/run_autograder" "$ROOT/source/"
  cp "$AG"/grader/*.py "$REPO/common.py" "$REPO/check_submission.py" "$ROOT/source/grader/"
  cp -r "$CACHE/data" "$CACHE/reference.json" "$ROOT/source/grader/"
fi
export AUTOGRADER_ROOT="$ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -n "${AG_FLAGS:-}" ]; then
  # same as run_autograder, plus debugging flags
  cd "$ROOT/source/grader"
  AUTOGRADER_DATA="$ROOT/source/grader/data" "$AG_PYTHON" grade.py --submission "$ROOT/submission" \
    --results "$ROOT/results/results.json" $AG_FLAGS
else
  bash "$ROOT/source/run_autograder"
fi
echo "results.json: $ROOT/results/results.json"
