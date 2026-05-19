#!/usr/bin/env bash
# Keeps a reverse SSH tunnel from this workstation to proxy.g5d.io alive.
# With GatewayPorts yes on the VM, port 2222 is publicly reachable.
# Single-hop access: ssh -p 2222 goodlad@workstation.g5d.io
#
# Run via systemd: systemctl --user start workstation-tunnel
# Logs: journalctl --user -u workstation-tunnel -f

set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/gcp_proxy}"
PROXY_HOST="${PROXY_HOST:-proxy.g5d.io}"
REMOTE_PORT="${REMOTE_PORT:-2222}"
RETRY_SECS="${RETRY_SECS:-10}"

while true; do
    echo "[workstation-tunnel] opening reverse tunnel → ${PROXY_HOST}:${REMOTE_PORT}"
    ssh \
        -i "$SSH_KEY" \
        -R "${REMOTE_PORT}:localhost:22" \
        -N \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=no \
        "goodlad@${PROXY_HOST}" || true
    echo "[workstation-tunnel] disconnected, retrying in ${RETRY_SECS}s..."
    sleep "$RETRY_SECS"
done
