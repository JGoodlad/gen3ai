#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-us-west1}"

IP=$(gcloud compute addresses describe proxy-ip \
  --region "$REGION" --format "value(address)")

echo "proxy-ip  = $IP"
echo "SSH cmd   : ssh -i ~/.ssh/gcp_proxy -D 1080 -N -q goodlad@${IP}"
echo "Proxy URL : socks5h://127.0.0.1:1080"
