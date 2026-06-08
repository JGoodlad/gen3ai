# GCP Infrastructure

One GCP e2-micro VM (`proxy.g5d.io`) serves two purposes:
1. **SOCKS5 proxy** — routes bot/collector traffic so Showdown sees the GCP IP, not your home IP
2. **Workstation gateway** — lets you SSH into your desktop from anywhere in one command

Both tunnels are desktop-initiated (no inbound ports needed on your home router).
Both ends are key-only — no password auth possible from outside.

---

## GCP VM details

| Field | Value |
|---|---|
| Project | g5d-dev |
| VM name | proxy |
| Zone | us-west1-a |
| External IP | 136.109.158.194 (static, reserved as `proxy-ip`) |
| DNS — proxy | proxy.g5d.io |
| DNS — workstation | workstation.g5d.io |
| SSH key | ~/.ssh/gcp_proxy |
| SSH user | goodlad |

The e2-micro VM in us-west1 is free-tier eligible — no charges as long as it stays
running and attached to the static IP.

---

## How the tunnels work

```
Desktop ──SSH──► GCP VM :22        (proxy-tunnel)     SOCKS5 on localhost:1080
Desktop ──SSH──► GCP VM :22        (workstation-tunnel) reverse tunnel → desktop:22

External client ──SSH──► GCP VM :2222 ──► desktop:22   (single hop, transparent)
```

Both tunnels are outbound connections from the desktop — your home router needs no
open ports. The GCP VM acts as a pure switchboard; it never terminates your SSH session.

---

## SSH access to the GCP VM

```bash
# Using the gcp_proxy key directly
ssh -i ~/.ssh/gcp_proxy goodlad@proxy.g5d.io

# Using gcloud (no key needed, just Google auth)
gcloud compute ssh proxy --zone us-west1-a --project g5d-dev
```

Add to `~/.ssh/config` for a shortcut (`ssh proxy`):
```
Host proxy proxy.g5d.io
    HostName proxy.g5d.io
    User goodlad
    IdentityFile ~/.ssh/gcp_proxy
```

**Emergency access:** `gcloud compute ssh` injects a temporary key via GCP project
metadata — works even if `authorized_keys` on the VM is broken or missing.

---

## SOCKS5 proxy tunnel

Routes bot and collector outbound traffic through the GCP VM.

**Daemon** (`proxy-tunnel.service` on the desktop):
```bash
systemctl --user status proxy-tunnel
systemctl --user restart proxy-tunnel
journalctl --user -u proxy-tunnel -f
```
Service: `~/.config/systemd/user/proxy-tunnel.service`  
Script: `scripts/workstation/proxy_tunnel.sh`

**Using the proxy:**
```bash
# Replay collector
python3 src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou \
  --max-concurrent 20 --proxy socks5h://127.0.0.1:1080

# Bot
python3 src/main/play.py --proxy socks5h://127.0.0.1:1080
```

The collector dashboard shows **PROXIED** in green when active.

**Verify no IP leak:**
```bash
pgrep -f collect_replays.py          # get PID
ss -tnp | grep "pid=<PID>,"          # should only show 127.0.0.1:1080
ss -tnp | grep ssh                   # should show 192.168.x.x → 136.109.158.194:22
```

---

## Workstation reverse tunnel

Lets you SSH into your desktop from anywhere:
```bash
ssh -p 2222 goodlad@workstation.g5d.io
```

Add to `~/.ssh/config` for a shortcut (`ssh workstation`):
```
Host workstation workstation.g5d.io
    HostName workstation.g5d.io
    Port 2222
    User goodlad
    IdentityFile ~/.ssh/gcp_proxy
```

**Daemon** (`workstation-tunnel.service` on the desktop):
```bash
systemctl --user status workstation-tunnel
systemctl --user restart workstation-tunnel
journalctl --user -u workstation-tunnel -f
```
Service: `~/.config/systemd/user/workstation-tunnel.service`  
Script: `scripts/workstation/reverse_tunnel.sh`

---

## Security model

### GCP VM (`/etc/ssh/sshd_config.d/99-hardened.conf`)
```
GatewayPorts yes          # makes reverse tunnel port publicly reachable
PasswordAuthentication no  # key-only — no brute-force possible
PubkeyAuthentication yes
```
OS Login is **disabled** — the `goodlad` user and `~/.ssh/authorized_keys` are used
directly. `gcloud compute ssh` still works via temporary project-metadata key injection.

### Desktop (`/etc/ssh/sshd_config.d/99-hardened.conf`)
```
# Default: key only
PasswordAuthentication no
PubkeyAuthentication yes

# Local network: also allow password (for adding new devices)
Match Address 192.168.0.0/16,10.0.0.0/8,172.16.0.0/12
    PasswordAuthentication yes
```

External connections (via the reverse tunnel) arrive as `127.0.0.1` — key-only.
Local network connections allow passwords so you can add a new device without
needing an existing key.

To apply after any sshd config change:
```bash
sudo sshd -t && sudo systemctl daemon-reload && sudo systemctl reload ssh
```

### Authorized keys

| Machine | File | Contains |
|---|---|---|
| GCP VM | `~/.ssh/authorized_keys` | `gcp_proxy` public key |
| Desktop | `~/.ssh/authorized_keys` | `gcp_proxy` public key, MacBook Air key |

SSH keys don't expire. The `gcp_proxy` key is permanent until explicitly removed.

### Firewall (GCP)
| Rule | Port | Purpose |
|---|---|---|
| default-allow-ssh | 22 | VM management, tunnel establishment |
| allow-workstation-ssh | 2222 | Reverse tunnel endpoint |

Port 2222 is publicly reachable but requires a key matching the desktop's
`authorized_keys` — unauthenticated connections are rejected immediately.

---

## Adding a new device

**From the local network** (easiest — password works locally):
1. Generate a key on the new device:
   ```bash
   ssh-keygen -t ed25519 -C "device-name"
   cat ~/.ssh/id_ed25519.pub
   ```
2. SSH to the desktop using your password:
   ```bash
   ssh goodlad@goodlad-desktop.local
   ```
3. Add the key:
   ```bash
   echo "ssh-ed25519 AAAA...newkey..." >> ~/.ssh/authorized_keys
   ```

The device can now connect externally via `ssh -p 2222 goodlad@workstation.g5d.io`.

**From outside** (already have an authorized device):  
SSH in from the authorized device and add the key to `~/.ssh/authorized_keys` as above.

---

## Restart / retry behaviour

Both tunnel scripts use a `while true` retry loop (10s delay on disconnect) and are
managed by systemd with `Restart=always`. SSH keepalives (`ServerAliveInterval=30`,
`ServerAliveCountMax=3`) detect dead connections within ~90 seconds.

Worst-case recovery: ~100 seconds from network drop to tunnel restored.

---

## TensorBoard remote access

TensorBoard is exposed publicly via Cloudflare Tunnel at **https://tensorboard.g5d.io**.

Make sure TensorBoard is running on the desktop first:
```bash
tensorboard --logdir models/   # recursively finds every models/*/tb/ (runs + _goldens)
```

The tunnel daemon runs as a systemd user service and starts automatically on login:
```bash
systemctl --user status cloudflared-tensorboard
systemctl --user restart cloudflared-tensorboard
journalctl --user -u cloudflared-tensorboard -f
```

Config: `~/.cloudflared/config.yml`  
Credentials: `~/.cloudflared/9ebabecb-fbdb-476a-925b-7329596cb38f.json`

### SSH port-forward (alternative, no cloudflared required)

If the tunnel is down, forward over the workstation SSH tunnel instead:
```bash
ssh -p 2222 -L 6006:localhost:6006 goodlad@workstation.g5d.io
# then open http://localhost:6006
```

---

## Troubleshooting

**SOCKS5 tunnel not working:**
```bash
ss -tlnp | grep 1080   # should show 127.0.0.1:1080
systemctl --user restart proxy-tunnel
```

**Can't reach workstation on port 2222:**
```bash
systemctl --user status workstation-tunnel
journalctl --user -u workstation-tunnel -n 20
# Check firewall rule exists:
gcloud compute firewall-rules list --project g5d-dev | grep 2222
```

**Permission denied (publickey) on workstation:**
Your key isn't in the desktop's `~/.ssh/authorized_keys`. Connect locally with
password and add it (see Adding a new device above).

**SSH host key changed warning (VM recreated):**
```bash
ssh-keygen -R proxy.g5d.io
ssh-keygen -R "[workstation.g5d.io]:2222"
```

**Check current GCP IP:**
```bash
./scripts/workstation/get_proxy_ip.sh
```
