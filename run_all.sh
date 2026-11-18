#!/usr/bin/env bash
# Full pipeline (or --report-only after experiments). Runtime depends on hardware.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 tools/run_pipeline.py "$@"
