#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf build/ ./*.egg-info
pip uninstall -y bsi_ops
python setup.py clean
pip install . -v --no-build-isolation > install.log 2>&1
