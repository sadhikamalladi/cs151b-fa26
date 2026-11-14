#!/usr/bin/env bash
# Gradescope runs this once, as root, when building the autograder image.
# The zip's contents are in /autograder/source.
set -euo pipefail
cd /autograder/source
apt-get update -y
apt-get install -y --no-install-recommends python3 python3-pip
python3 -m pip install --no-cache-dir -r requirements.txt
