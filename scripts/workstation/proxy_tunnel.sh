#!/usr/bin/env bash
# Keeps the SOCKS5 SSH tunnel to proxy.g5d.io alive.
# Run via systemd: systemctl --user start proxy-tunnel
# Logs viewable with: journalctl --user -u proxy-tunnel -f

set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/gcp_proxy}"
PROXY_HOST="${PROXY_HOST:-proxy.g5d.io}"
LOCAL_PORT="${LOCAL_PORT:-1080}"
RETRY_SECS="${RETRY_SECS:-10}"

while true; do
    echo "[proxy-tunnel] connecting ${PROXY_HOST} → localhost:${LOCAL_PORT}"
    ssh \
        -i "$SSH_KEY" \
        -D "$LOCAL_PORT" \
        -N \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=no \
        "goodlad@${PROXY_HOST}" || true
    echo "[proxy-tunnel] disconnected, retrying in ${RETRY_SECS}s..."
    sleep "$RETRY_SECS"
done
