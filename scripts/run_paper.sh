#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src"
python -m tradebot.app run --paper
