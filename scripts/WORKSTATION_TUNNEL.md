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

## Security

- **Port 2222** is publicly reachable but the local sshd is **key-only** —
  no password brute-force is possible
- **GCP VM sshd** (`/etc/ssh/sshd_config.d/99-hardened.conf`):
  `PasswordAuthentication no`, `GatewayPorts yes`
- **Local sshd** (`/etc/ssh/sshd_config.d/99-hardened.conf`):
  `PasswordAuthentication no` — run once to apply:
  ```bash
  sudo tee /etc/ssh/sshd_config.d/99-hardened.conf <<'EOF'
  PasswordAuthentication no
  PubkeyAuthentication yes
  EOF
  sudo sshd -t && sudo systemctl reload ssh
  ```
- **Authorized keys**: only keys in `~/.ssh/authorized_keys` on the local machine
  can connect. Currently includes `gcp_proxy` key.

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
