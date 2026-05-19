# Workstation Tunnel

Lets you SSH directly into your local workstation from anywhere with a single command:

```bash
ssh -p 2222 goodlad@workstation.g5d.io
```

Traffic flows: `you → workstation.g5d.io:2222 → GCP VM → reverse tunnel → local sshd`

No intermediate hop visible. Both ends are key-only — no password auth possible anywhere.

---

## How it works

The local machine runs a persistent reverse SSH tunnel to `proxy.g5d.io`. With
`GatewayPorts yes` on the VM, port 2222 is publicly reachable. Connecting to port 2222
transparently forwards through to the local machine's port 22.

---

## The tunnel daemon

Runs as a systemd user service on the local machine.

```bash
# Check status
systemctl --user status workstation-tunnel

# View live logs
journalctl --user -u workstation-tunnel -f

# Restart
systemctl --user restart workstation-tunnel
```

Service file: `~/.config/systemd/user/workstation-tunnel.service`
Script: `scripts/reverse_tunnel.sh`

---

## Connecting

```bash
# Direct (requires specifying port)
ssh -p 2222 -i ~/.ssh/gcp_proxy goodlad@workstation.g5d.io

# Or via DNS alias (same IP as proxy.g5d.io)
ssh -p 2222 -i ~/.ssh/gcp_proxy goodlad@proxy.g5d.io
```

### Optional SSH config shortcut

Add to `~/.ssh/config` on any machine you connect from — then just `ssh workstation`:

```
Host workstation workstation.g5d.io
    HostName workstation.g5d.io
    Port 2222
    User goodlad
    IdentityFile ~/.ssh/gcp_proxy
```

---

## DNS

| Record | Type | Value |
|--------|------|-------|
| `workstation.g5d.io` | A | `136.109.158.194` |

Same IP as `proxy.g5d.io` — the GCP VM handles both.

---

## Security model

| Connection source | Auth allowed |
|---|---|
| Local network (`192.168.x`, `10.x`, `172.16.x`) | Password **or** key |
| External via tunnel (appears as `127.0.0.1`) | Key only |

This means you can always add a new device from your home network using just a password,
but external access always requires a key — no brute-force possible remotely.

### Local sshd config (`/etc/ssh/sshd_config.d/99-hardened.conf`)

```
# Default: key only
PasswordAuthentication no
PubkeyAuthentication yes

# Local network: also allow password (for adding new devices)
Match Address 192.168.0.0/16,10.0.0.0/8,172.16.0.0/12
    PasswordAuthentication yes
```

To apply changes after editing:
```bash
sudo sshd -t && sudo systemctl daemon-reload && sudo systemctl reload ssh
```

### GCP VM sshd (`/etc/ssh/sshd_config.d/99-hardened.conf` on proxy.g5d.io)

```
GatewayPorts yes
PasswordAuthentication no
PubkeyAuthentication yes
```

### Authorized keys

`~/.ssh/authorized_keys` on the desktop currently contains:
- `gcp_proxy` key (for the reverse tunnel and remote connections)
- MacBook Air key

---

## Adding a new device

**From the local network** (easiest — no existing key needed):

1. On the new device, generate a key if you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "device-name"
   cat ~/.ssh/id_ed25519.pub
   ```
2. SSH to the desktop with your password (works on local network):
   ```bash
   ssh goodlad@goodlad-desktop.local
   ```
3. Add the new key:
   ```bash
   echo "ssh-ed25519 AAAA...newkey..." >> ~/.ssh/authorized_keys
   ```

The new device can now connect externally via `ssh -p 2222 goodlad@workstation.g5d.io` key-only.

**From outside** (if you already have another authorized device):

SSH in from an authorized device and add the new key to `~/.ssh/authorized_keys` as above.

---

## Troubleshooting

**Connection refused on port 2222:**
Tunnel is down. Check status:
```bash
systemctl --user status workstation-tunnel
journalctl --user -u workstation-tunnel -n 20
```

**Permission denied (publickey):**
Your key isn't in the local machine's `~/.ssh/authorized_keys`.
Add it: `cat ~/.ssh/your_key.pub >> ~/.ssh/authorized_keys`

**Tunnel won't start (ExitOnForwardFailure):**
Port 2222 is already in use on the VM or the GCP firewall rule is missing:
```bash
gcloud compute firewall-rules list --project g5d-dev | grep 2222
```
