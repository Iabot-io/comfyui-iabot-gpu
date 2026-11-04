#!/usr/bin/env bash
set -euo pipefail
custom=()
if [ "${IABOT_PROFILE:-video}" = "voice" ]; then custom=(--whitelist-custom-nodes iabot_voice); fi
python -u /comfyui/main.py --disable-auto-launch --disable-metadata --disable-api-nodes --listen 127.0.0.1 --port 8188 --disable-all-custom-nodes "${custom[@]}" &
comfy_pid=$!
trap 'kill "$comfy_pid" 2>/dev/null || true' EXIT
python - <<'PY'
import time, requests
for attempt in range(180):
    try:
        if requests.get('http://127.0.0.1:8188/system_stats', timeout=2).status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError('ComfyUI did not start')
PY
exec python -u /iabot/worker.py
