#!/bin/bash
set -e

# Start Xvfb for headless rendering (SAPIEN/ManiSkill needs a display)
Xvfb :99 -screen 0 1024x768x24 +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to be ready
for i in $(seq 1 10); do
    if xdpyinfo -display :99 >/dev/null 2>&1; then
        echo "Xvfb is ready on :99"
        break
    fi
    sleep 0.5
done

export DISPLAY=:99

exec "$@"
