#!/usr/bin/env bash
#
# One-command launcher: installs dependencies (if needed) and starts the app.
#
#   ./run.sh
#
set -euo pipefail
cd "$(dirname "$0")"

# Install/refresh dependencies into .venv from pyproject.toml + uv.lock
uv sync

# Launch the control panel
uv run python gui/main.py
