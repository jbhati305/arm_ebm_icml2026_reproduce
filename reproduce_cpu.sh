#!/usr/bin/env bash
set -euo pipefail

uv sync --cache-dir /tmp/arm_ebm_uv_cache
uv run --cache-dir /tmp/arm_ebm_uv_cache python -m pytest
uv run --cache-dir /tmp/arm_ebm_uv_cache python -m repro_arm_ebm.run \
  --steps 3000 \
  --output-dir outputs/cpu_exact
