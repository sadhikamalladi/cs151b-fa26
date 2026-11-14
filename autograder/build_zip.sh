#!/usr/bin/env bash
# Build the Gradescope autograder zip:  bash autograder/build_zip.sh [out.zip]
# Optional: MISSION=M1|M2|M3 makes the autograder reject other missions' code.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
OUT="$(realpath -m "${1:-$REPO/autograder.zip}")"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

mkdir -p "$BUILD/grader"
cp "$HERE/setup.sh" "$HERE/run_autograder" "$HERE/requirements.txt" "$BUILD/"
cp "$HERE/grader/grade.py" "$HERE/grader/drive.py" "$HERE/grader/model_meta.py" \
   "$HERE/grader/reference.json" "$REPO/check_submission.py" "$BUILD/grader/"
if [ -n "${MISSION:-}" ]; then echo "$MISSION" > "$BUILD/grader/expected_mission.txt"; fi
chmod +x "$BUILD/setup.sh" "$BUILD/run_autograder"
rm -f "$OUT"
(cd "$BUILD" && zip -qr "$OUT" setup.sh run_autograder requirements.txt grader)
echo "Built $OUT ($(ls -lh "$OUT" | awk '{print $5}'))"
