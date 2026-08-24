# GCP Infrastructure

One GCP e2-micro VM (`proxy.g5d.io`) serves two purposes:
1. **SOCKS5 proxy** — routes bot/collector traffic so Showdown sees the GCP IP, not your home IP
2. **Workstation gateway** — lets you SSH into your desktop from anywhere in one command

Both tunnels are desktop-initiated (no inbound ports needed on your home router).
Both ends are key-only — no password auth possible from outside.

## Web endpoints at a glance

| Port | What | Exposed as | How it runs |
|---|---|---|---|
| 6006 | TensorBoard | `tensorboard.g5d.io` | `tensorboard.service` + `cloudflared-tensorboard` |
| 6007 | Model architecture viewer | `model.g5d.io` | `gen3ai-model-viewer.service` + tunnel ingress |
| 6008 | Prober web views | `prober.g5d.io` | `gen3ai-prober-web.service` + tunnel ingress |
| 1080 | SOCKS5 proxy | — | `proxy-tunnel` |

Every origin binds `127.0.0.1`, so a Cloudflare tunnel entry is the only way anything becomes
publicly reachable. Ports **8000** (dev Showdown) and **8001** (LIVE TRAINING Showdown — never
touch it) are not web endpoints and are not tunnelled; an agent needing its own Showdown server
binds a `9XXX` port.

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

# Bot (public LADDER — needs a registered account; see designs/research_state/ladder_readiness.md)
PS_PASSWORD=... python3 src/main/play.py --mode ladder --server official \
  --model models/<run>/final_model.zip --username <acct> --n-battles 20 \
  --proxy socks5h://127.0.0.1:1080
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

**Both halves run as systemd user services** — the origin (TensorBoard on `:6006`) and the
tunnel (`cloudflared`). Neither should ever be started by hand; see the outage note below
for why.

```bash
# the origin — serves models/ on localhost:6006
systemctl --user status  tensorboard
systemctl --user restart tensorboard
journalctl --user -u tensorboard -f

# the tunnel — tensorboard.g5d.io -> localhost:6006
systemctl --user status  cloudflared-tensorboard
systemctl --user restart cloudflared-tensorboard
journalctl --user -u cloudflared-tensorboard -f
```

Units: `~/.config/systemd/user/{tensorboard,cloudflared-tensorboard}.service`
(a reference copy of the TensorBoard unit is versioned at
`scripts/workstation/tensorboard.service`)  
Config: `~/.cloudflared/config.yml`  
Credentials: `~/.cloudflared/9ebabecb-fbdb-476a-925b-7329596cb38f.json`

`--logdir models/` recursively finds every `models/*/tb/` (runs + `_goldens`).

### Why the origin is a service (Jul 29 2026 outage)

A machine-wide OOM (`global_oom`, triggered by a `python3` process) swept the user session
at ~01:01 and killed both halves. `cloudflared` is an enabled unit, so systemd restarted it
10s later — but TensorBoard had been launched by hand under `nohup`/tmux, so **nothing
brought it back**. The tunnel then forwarded traffic to a dead port for three days, serving
`Unable to reach the origin service ... connection refused` while looking healthy from the
outside.

Two guards now:
- `Restart=always` on `tensorboard.service` — it self-heals from an OOM kill or a crash.
- `MemoryMax=6G` — TensorBoard can never be the process that pushes the box into a *global*
  OOM. If it exceeds the cap its own cgroup OOMs and it restarts, instead of the whole user
  session getting swept.

**Diagnosing a repeat:** if the site errors, check the origin first — `ss -ltn | grep 6006`.
A live tunnel with a dead origin is the signature failure, and `systemctl --user status
tensorboard` will show it. Note `Linger=no` for this user, so both units still stop on a
full logout; run `loginctl enable-linger goodlad` if you want them to survive that.

### SSH port-forward (alternative, no cloudflared required)

If the tunnel is down, forward over the workstation SSH tunnel instead:
```bash
ssh -p 2222 -L 6006:localhost:6006 goodlad@workstation.g5d.io
# then open http://localhost:6006
```

---

## Model architecture viewer

The Gen3AI delivery digraph is exposed at **https://model.g5d.io**, same tunnel, same shape as
TensorBoard: an origin service on `:6007` plus a `cloudflared` ingress entry.

```bash
systemctl --user status  gen3ai-model-viewer
systemctl --user restart gen3ai-model-viewer
journalctl --user -u gen3ai-model-viewer -f
```

Unit: `~/.config/systemd/user/gen3ai-model-viewer.service`
(reference copy versioned at `scripts/workstation/gen3ai-model-viewer.service`)

### It is not a file — it renders per request

The origin runs `python -m agents.model.build_arch_viewer --serve`, which rebuilds the page from
the checkout at `/home/goodlad/dev/gen3ai` on **every request** (~6 ms; it reads JSON and one
source file and never imports torch). There is deliberately no deployed copy: a copy is a second
artifact that goes stale the moment the first one moves, which is the exact rot the viewer exists
to prevent.

The rendering path is fully live, including the generator itself — `build_arch_viewer.py` is
re-executed into a fresh module object when its mtime changes, so **an architecture change reaches
the endpoint with no restart** (verified: the v60 entity re-home appeared on `model.g5d.io` without
the unit being touched). A file caught mid-save raises, the request 500s with the traceback, and
the last good module keeps serving; it recovers on the next successful read.

The one exception is the SERVER, not the content: `serve()` and its request handler are closures
bound at startup, so a change to the routes — or to what the handler injects, like the snapshot-age
field — does need `systemctl --user restart gen3ai-model-viewer`. Content changes never do.

Responses carry an `ETag` with `Cache-Control: no-cache`, so a browser revalidates on every load
(it can never show a stale architecture) but gets a 304 and zero bytes when nothing changed.

### Restart behaviour

`Restart=always` with **exponential backoff** — `RestartSec=2`, `RestartSteps=6`,
`RestartMaxDelaySec=5min` — and `StartLimitIntervalSec=0`.

That last line is the one that matters. systemd's default start-rate limit (5 starts in 10s) puts a
unit into `failed` and **stops restarting it**, which reproduces the Jul 29 2026 TensorBoard
failure exactly: a dead origin behind a tunnel that still looks healthy. With the limit disabled
the unit always comes back, and the backoff is what keeps a genuinely broken checkout from
busy-looping on a box that is also training (twelve retries an hour at the ceiling, not thousands).

Measured on a probe unit: restart gaps grew 2.25s → 4.25s → 8.25s to the ceiling, and the unit
stayed `activating` rather than `failed` after five consecutive failures.

`MemoryMax=512M` for the same reason TensorBoard has a cap — this process should never be what
pushes the box into a global OOM. `Nice=10` / `CPUWeight=20` keep it from competing with training.

### Diagnosing

Same signature failure as TensorBoard: check the origin first.

```bash
ss -ltn | grep 6007
curl -s localhost:6007/healthz     # -> "ok 58 nodes 487 edges"
```

`/healthz` renders the real payload, so a 200 there means the checkout genuinely builds — not just
that a socket is open. `/graph.json` returns the raw payload, which is a better thing to hand an AI
than 300 KB of HTML.

### Access control

`model.g5d.io` is a public hostname on a public domain, and the page carries the full delivery
digraph, the measured audit numbers and the contamination notes. Put **Cloudflare Access** in front
of it unless that is intended to be world-readable. The origin binds to `127.0.0.1` only, so the
tunnel is the sole path in.

---

## Prober web views — https://prober.g5d.io (`:6008`)

The prober's browser front end (`src/main/prober/web/`, see its `CLAUDE.md`) serves the read-only
forensic views — run summary, battles, `scan`, `triage`, the `falsify_scan` crater bracket, the
`calibration` reliability curve.

⚠️ **Adding an ingress hostname INTERRUPTS the other two.** `cloudflared` does not hot-reload its
ingress: sending `SIGHUP` makes the process EXIT, and `Restart=always` brings it back ~3 s later
(measured 2026-08-10 — PID changed, `NRestarts` went 0→1, and all three hostnames re-registered
their QUIC connections). So editing `config.yml` costs tensorboard.g5d.io and model.g5d.io a few
seconds of downtime whichever way you do it; there is no gentler path, and `systemctl --user
restart cloudflared-tensorboard` is the honest spelling. Validate first — `cd ~/.cloudflared &&
cloudflared tunnel ingress validate` — because a malformed config takes all three down until it is
fixed, not just the new one.

Third endpoint on the same tunnel, same shape as TensorBoard and the model viewer: an origin
service on `:6008` plus a `cloudflared` ingress entry.

```bash
systemctl --user status  gen3ai-prober-web
systemctl --user restart gen3ai-prober-web       # the watchdog does this for you — see below
journalctl --user -u gen3ai-prober-web -f
```

Unit: `~/.config/systemd/user/gen3ai-prober-web.service`
(reference copy versioned at `scripts/workstation/gen3ai-prober-web.service`)

**It serves `models/` from the MAIN CHECKOUT, so it runs whatever code local main is at.** Pushing
to origin is not enough — local main must be fast-forwarded, or the endpoint keeps serving the
previous commit. Unlike the model viewer (which re-executes its generator on mtime change) this is
a FastAPI app whose routes are bound at startup; nothing here reloads.

### The staleness watchdog (why "restart it yourself" was not enough)

**`Restart=always` does not cover the failure that actually happened**, because in that failure
nothing dies. Jinja reloads a changed TEMPLATE from disk while Python cannot reload a changed
MODULE, so a long-lived process ends up serving new templates against old code.

**Measured 2026-08-18:** this unit had been up 5 days and returned **HTTP 500 on every `/battle`**
for a current run — a template shipped two days earlier read a key the running `session.py`
predated. systemd reported the unit healthy the whole time, the tunnel reported the origin up, and
the only signal was a person saying the page would not load. A five-day outage of the main view,
invisible to every check pointed at it.

Two mechanisms now close it:

1. **The app pins its templates to the process** (`auto_reload=False`), so a stale process serves a
   coherent OLD page rather than a broken hybrid.
2. **A timer restarts it when it falls behind local main.**

```bash
cp scripts/workstation/gen3ai-prober-web-watchdog.{service,timer} ~/.config/systemd/user/
cp scripts/workstation/prober_web_watchdog.sh /home/goodlad/dev/gen3ai/scripts/workstation/   # in-repo already
systemctl --user daemon-reload
systemctl --user enable --now gen3ai-prober-web-watchdog.timer

systemctl --user list-timers gen3ai-prober-web-watchdog     # when it next fires
journalctl --user -u gen3ai-prober-web-watchdog -n 30       # what it decided, and why
scripts/workstation/prober_web_watchdog.sh --dry-run        # check without acting
```

It compares `/api/health`'s **`revision`** — the git sha of the source the PROCESS imported,
captured once at import — against `git rev-parse HEAD` in the repo, every 2 minutes. On a mismatch
it restarts the unit and then **verifies the replacement came up on the new revision** (reporting
success into a crash-loop would be the same invisible-failure shape). It **defers while a job is
running**, because a restart kills a multi-minute `falsify_scan` or `calibration`; the next tick
retries. A unit that is deliberately STOPPED is left stopped — that is systemd's business, not the
watchdog's.

Known limit: an **uncommitted** edit does not move HEAD and so does not trigger a restart. That is
right for this box (main only advances by commit — all work happens in worktrees), but if you
hand-edit a file under the service, restart it yourself.

**It is pointed at `models/`, not at one run** — the header carries a run picker — so a new
generation starting does NOT require touching the unit.

### Local-only alternative

If the tunnel is down, or you want a throwaway instance on a different port (note :6008 is taken
by the service — use `--port 6108` for a second one):

```bash
# on the workstation (the main checkout — the editable install covers the import path)
python -m main.prober.web /home/goodlad/dev/gen3ai/models --port 6108
# -> http://127.0.0.1:6108

# from another machine, over the workstation SSH tunnel (no cloudflared involved)
ssh -p 2222 -L 6008:localhost:6008 goodlad@workstation.g5d.io
# then open http://localhost:6008
```

`models/` lives only in the main checkout, so pass an absolute `models/...` path when running from
a git worktree.

### The shared password

Reading is anonymous. The two probes that spend minutes of CPU (`falsify_scan`, `calibration`) are
behind one shared password, handed out in Discord as needed. No usernames, no email: helping out
should not cost anyone personal information.

**The password itself is deliberately not written down in this repository.** This file is
committed, and the repo is public — printing the secret here would publish the thing the Discord
hand-out exists to control. It lives only at `~/.config/gen3ai/prober-password` (mode 0600) on the
workstation; `cat` it when you need to share it.

Nothing exports the password itself — only the PATH to it, from two places so it survives a reboot
for both consumers:

| File | Who reads it |
|---|---|
| `~/.config/environment.d/60-gen3ai-prober.conf` | the systemd **user manager**, at login/boot — so a future `gen3ai-prober-web.service` inherits it |
| `~/.profile` | login shells, **including non-interactive ones** |
| `~/.bashrc` | interactive non-login shells |

Both shell files are needed, and the reason is a trap worth remembering: Ubuntu's `~/.bashrc`
opens with *"If not running interactively, don't do anything"* and returns, so an export appended
to it is invisible to `bash -lc`, cron, and anything else non-interactive. `~/.profile` has no such
guard. (Verified: with only the `.bashrc` export, `bash -lc 'echo $GEN3AI_PROBER_PASSWORD_FILE'`
printed nothing.)

```bash
cat ~/.config/gen3ai/prober-password          # the secret itself
echo $GEN3AI_PROBER_PASSWORD_FILE             # the path, in a fresh login shell
```

`environment.d` is picked up when the user manager starts, so a change needs a re-login (or
`systemctl --user import-environment` for an already-running session). To rotate the password,
edit the file and restart whatever is serving — the app reads it once at startup, and its
cookie-signing key is per-process, so every existing session ends at the same moment.

**It fails closed.** With no password set the probes are switched off entirely rather than left
open, so a misconfigured deploy is read-only, never a public CPU-burn button. `--open` is the
explicit opt-out for a laptop.

### Which runs it will open

It is pointed at `models/` and enumerates the runs inside it; the header carries a picker. A
request names a run by NAME, and the name must be a member of that server-built listing — no
client string is ever joined to a path, so traversal is unrepresentable rather than filtered.

The one asymmetry worth knowing: **a direct child of `models/` may be a symlink** and is followed
(resolved once, at enumeration) — that is how the launcher's worktree runs appear, and on this box
the five newest runs are exactly that. A symlink **inside** a run refuses the whole run. Details
and the attack list: `src/main/prober/web/CLAUDE.md` and `runs_test.py`.

### Access posture, and the deploy's own guards

**No Cloudflare Access — decided by the owner (2026-08-09): this is an open-source model, and its
architecture, training outcomes and forensic traces are all intended to be public.** Same posture
as `model.g5d.io`. The expensive probes are gated instead by the app's own shared password (above),
which protects CPU rather than data. The one thing on these pages that is *not* model content is
incidental: the run header and the battle `id` column print absolute paths under
`/home/goodlad/dev/gen3ai/models/`, so the deploy discloses the box's directory layout and run
names. If that ever matters, it is a rendering change (show run-relative ids), not a reason for an
auth layer.

Because reading is anonymous, **anything an anonymous request can make grow needs a bound**, and
two were found by measuring rather than by review:

- the login-throttle failure map (LRU-bounded; it was 3000 permanent entries per 3000 spoofed
  identities), and
- **the per-run `ProbeSession` cache** — a `scan` of one run leaves ~430 MB behind, and the picker
  offers 81 runs, so walking them all reached ~35 GB on a box with 89 GB that is also training.
  Now LRU-bounded at 3 (`_MAX_CACHED_SESSIONS`), with `MemoryMax=6G` in the unit as the backstop.

`--job-workers 1` is the third bound: the job endpoints are the only way a visitor who has the
password can spend real CPU, and one re-roll job at a time is the cap. Jobs are in-process state,
so a restart silently discards a running scan.

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
