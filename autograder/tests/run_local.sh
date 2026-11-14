#!/usr/bin/env bash
# Run the autograder on one submission folder in a Gradescope-like layout:
#   <tmp>/autograder/{source,submission,results}
#
#   bash autograder/tests/run_local.sh path/to/submission [autograder.zip]
#
# Without a zip, the zip is built from this checkout first. Uses $AG_PYTHON (default python3),
# which needs `requests`. Set AUTOGRADER_DRIVE_MOCK to use the mock Drive from make_examples.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB="$(realpath "$1")"
ZIP="${2:-}"
ROOT="$(mktemp -d)/autograder"
mkdir -p "$ROOT/source" "$ROOT/results"
if [ -z "$ZIP" ]; then
  ZIP="$(dirname "$ROOT")/autograder.zip"
  bash "$HERE/../build_zip.sh" "$ZIP" >/dev/null
fi
unzip -q "$ZIP" -d "$ROOT/source"
cp -r "$SUB" "$ROOT/submission"
AUTOGRADER_ROOT="$ROOT" bash "$ROOT/source/run_autograder"
echo "results.json: $ROOT/results/results.json"
