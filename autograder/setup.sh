#!/usr/bin/env bash
# Gradescope runs this once, as root, when building the autograder image.
# The zip's contents are in /autograder/source.
set -euo pipefail
cd /autograder/source

apt-get update -y
apt-get install -y --no-install-recommends curl ca-certificates

# Python 3.11 venv through uv (independent of the base image's Python)
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
uv venv --python 3.11 /opt/agvenv
export VIRTUAL_ENV=/opt/agvenv
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt

# Unprivileged user that runs student code
id -u student >/dev/null 2>&1 || useradd --create-home --shell /bin/bash student

# Cache the base model so checks run offline
export HF_HOME=/opt/hf
cd grader
/opt/agvenv/bin/python -c "import common; common.load_model(device='cpu'); print('Base model cached.')"
chmod -R a+rX /opt/hf /opt/agvenv /opt/uv-python /autograder/source
