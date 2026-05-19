# Proxy Tunnel

Routes bot and replay-collector traffic through a GCP VM (`proxy.g5d.io`) so Pokémon
Showdown only sees the GCP IP, not your home IP. Uses an SSH SOCKS5 tunnel — no ports
need to be open on the VM.

---

## How it works

An SSH connection to `proxy.g5d.io` creates a local SOCKS5 proxy on `localhost:1080`.
The bot and collector pass all WebSocket and HTTP traffic through that port. From
Showdown's perspective, every connection comes from `136.109.158.194` (the GCP VM in
us-west1).

---

## The tunnel daemon

The tunnel runs as a systemd user service that starts automatically when you log in and
restarts within 10 seconds if the connection drops.

```bash
# Check if the tunnel is running
systemctl --user status proxy-tunnel

# View live logs (Ctrl+C to exit)
journalctl --user -u proxy-tunnel -f

# View recent logs
journalctl --user -u proxy-tunnel -n 50

# Manually stop the tunnel
systemctl --user stop proxy-tunnel

# Manually start it again
systemctl --user start proxy-tunnel

# Restart it (e.g. after changing the script)
systemctl --user restart proxy-tunnel
```

The tunnel is **enabled** — it starts automatically on login. To disable auto-start:
```bash
systemctl --user disable proxy-tunnel
```

Service file: `~/.config/systemd/user/proxy-tunnel.service`
Script: `scripts/proxy_tunnel.sh`

---

## Running the replay collector through the proxy

The tunnel must be running (it auto-starts, so it usually is). Pass `--proxy`:

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 src/main/collect_replays.py \
  --format gen3ou \
  --save-dir replays/gen3ou \
  --max-concurrent 20 \
  --proxy socks5h://127.0.0.1:1080
```

The dashboard shows **PROXIED socks5h://127.0.0.1:1080** in the stats row when the
proxy is active, so you can confirm it at a glance.

---

## Running the bot through the proxy

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 src/main/play.py --proxy socks5h://127.0.0.1:1080
```

To play without the proxy (e.g. against the local Showdown server during development):
```bash
python3 src/main/play.py
```

---

## Check your current proxy IP

```bash
./scripts/get_proxy_ip.sh
```

---

## GCP VM details

| Field | Value |
|---|---|
| Project | g5d-dev |
| VM name | proxy |
| Zone | us-west1-a |
| External IP | 136.109.158.194 (static, reserved as `proxy-ip`) |
| DNS | proxy.g5d.io |
| SSH key | ~/.ssh/gcp_proxy |
| SSH user | goodlad |

The e2-micro VM in us-west1 is free-tier eligible — no charges as long as it stays
running and attached to the static IP.

---

## Verify no IP leak

After starting the collector, confirm its only outbound connection goes through the
tunnel (not directly to Showdown):

```bash
# Find the collector PID
pgrep -f collect_replays.py

# Check its TCP connections — should only show 127.0.0.1:1080, nothing to Showdown
ss -tnp | grep "pid=<PID>,"

# Confirm the SSH process itself exits to the GCP IP
ss -tnp | grep ssh
# Should show: 192.168.x.x:<port>  136.109.158.194:22
```

---

## Troubleshooting

**Tunnel shows `failed` in status:**
```bash
journalctl --user -u proxy-tunnel -n 20
```
Usually means the VM is unreachable. Check GCP Console → Compute Engine → VM instances
and confirm the `proxy` VM is running.

**Collector can't connect even though tunnel is active:**
Check the tunnel is actually listening:
```bash
ss -tlnp | grep 1080
```
Should show something like `127.0.0.1:1080`. If empty, restart the tunnel:
```bash
systemctl --user restart proxy-tunnel
```

**SSH host key changed warning:**
If the VM was recreated, remove the old key:
```bash
ssh-keygen -R proxy.g5d.io
```
