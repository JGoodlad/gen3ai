"""PROVISION a vast.ai GPU box for the cloud-GPU benchmark: search, create (+ the DEADMAN), wait, ssh, destroy.

    export PYTHONPATH=$PWD/src                                   # from the repo root (a worktree is fine)
    python -m tools.cloud.vast.cli credit                        # the account's credit (never the key)
    python -m tools.cloud.vast.cli search [--max-dph 1.0] [--out offers.json]
    python -m tools.cloud.vast.cli ensure-key                    # ~/.ssh/gen3ai_vast (ed25519), on the account
    python -m tools.cloud.vast.cli create --offer <id> --max-minutes 75 --min-credit 7.0
    python -m tools.cloud.vast.cli wait --instance <id>          # running + sshd answering; prints the ssh line
    python -m tools.cloud.vast.cli ssh --instance <id> -- <cmd>  # one remote command (the dedicated key only)
    python -m tools.cloud.vast.cli scp-to --instance <id> <local> <remote> / scp-from ... <remote> <local>
    python -m tools.cloud.vast.cli destroy --instance <id>       # destroy, then VERIFY it is gone
    python -m tools.cloud.vast.cli instances | verify-zero       # verify-zero exits 1 unless the account has none

THE SAFETY RULES (designs/ops/cloud_gpu_benchmark.md "Safety"): ONE instance at a time (``create`` refuses while
any exists); every ``create`` spawns a DETACHED deadman (its own session, survives this process and the agent)
that destroys the instance at ``--max-minutes`` or as soon as the credit falls below ``--min-credit``, and keeps
retrying until the API no longer lists it. ``ssh`` / ``scp`` use ONLY the dedicated key, ignore ~/.ssh/config
(``-F /dev/null``: no agent forwarding, no other identity) and a separate known_hosts. Every create / destroy is
appended to the ledger ``~/.local/state/gen3ai/vast/ledger.jsonl`` with the credit at that moment.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from tools.cloud.vast import api
from tools.cloud.vast import offers as O

KEY_PATH = os.path.expanduser("~/.ssh/gen3ai_vast")
KNOWN_HOSTS = os.path.expanduser("~/.ssh/gen3ai_vast_known_hosts")
STATE_DIR = os.path.expanduser("~/.local/state/gen3ai/vast")
LEDGER = os.path.join(STATE_DIR, "ledger.jsonl")
#: A pinned, small CUDA 12.8 base: torch's cu128 wheels bring their own CUDA libs, Triton its own ptxas; the driver
#: comes from the host. The remote bootstrap installs everything else (remote_bootstrap.sh).
DEFAULT_IMAGE = "nvidia/cuda:12.8.1-base-ubuntu22.04"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ledger(event: str, **kw: Any) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": _now(), "event": event, **kw}) + "\n")


def _say(msg: str) -> None:
    print(f"[vast {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- ssh
def ssh_endpoint(inst: Dict[str, Any]) -> Optional[tuple]:
    """``(host, port)``: the DIRECT port when the instance maps one (faster), else vast's ssh proxy."""
    ports = inst.get("ports") or {}
    direct = ports.get("22/tcp") if isinstance(ports, dict) else None
    if inst.get("public_ipaddr") and direct:
        return str(inst["public_ipaddr"]).strip(), int(direct[0]["HostPort"])
    if inst.get("ssh_host") and inst.get("ssh_port"):
        return str(inst["ssh_host"]).strip(), int(inst["ssh_port"])
    return None


def ssh_base(host: str, port: int, scp: bool = False) -> List[str]:
    opts = ["-F", "/dev/null", "-i", KEY_PATH, "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
            "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
    if scp:
        return ["scp", "-r", *opts, "-P", str(port)]
    return ["ssh", *opts, "-p", str(port), f"root@{host}"]


def _endpoint_or_die(instance_id: int) -> tuple:
    inst = api.instance(instance_id)
    ep = ssh_endpoint(inst or {})
    if not ep:
        raise SystemExit(f"instance {instance_id}: no ssh endpoint yet ({(inst or {}).get('actual_status')})")
    return ep


# ---------------------------------------------------------------------------------------------- commands
def cmd_credit(_a: argparse.Namespace) -> int:
    print(json.dumps(api.account()))
    return 0


def cmd_search(a: argparse.Namespace) -> int:
    c = O.Criteria(gpu_name=a.gpu, max_dph=a.max_dph, min_reliability=a.min_reliability, min_cuda=a.min_cuda)
    raw = api.search_offers(O.search_query(c))
    keep, why = O.select(raw, c)
    _say(f"{len(raw)} offers returned; {len(keep)} pass; rejected: {why}")
    for o in keep[:a.show]:
        print(json.dumps(O.summary(o)))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump([O.summary(o) for o in keep], fh, indent=1)
    return 0 if keep else 1


def cmd_ensure_key(_a: argparse.Namespace) -> int:
    if not os.path.exists(KEY_PATH):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "gen3ai-vast", "-f", KEY_PATH],
                       check=True)
        _say(f"created {KEY_PATH} (ed25519, dedicated to vast.ai)")
    with open(KEY_PATH + ".pub", encoding="utf-8") as fh:
        pub = fh.read().strip()
    body = pub.split()[1]
    have = [k for k in api.ssh_keys() if body in str(k.get("public_key") or k.get("ssh_key") or k.get("key") or "")]
    if have:
        _say("the dedicated key is already on the account")
    else:
        api.add_account_ssh_key(pub)
        _say("registered the dedicated public key on the account")
    return 0


def spawn_deadman(instance_id: int, deadline: float, min_credit: float) -> int:
    os.makedirs(STATE_DIR, exist_ok=True)
    log = open(os.path.join(STATE_DIR, f"deadman_{instance_id}.log"), "a", encoding="utf-8")
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.Popen([sys.executable, "-m", "tools.cloud.vast.cli", "deadman", "--instance", str(instance_id),
                          "--deadline", str(int(deadline)), "--min-credit", str(min_credit)],
                         cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return p.pid


def cmd_create(a: argparse.Namespace) -> int:
    live = api.instances()
    if live:
        raise SystemExit(f"REFUSED: the account already has {len(live)} instance(s) "
                         f"{[i.get('id') for i in live]} — one at a time; destroy first")
    cr = api.credit()
    if cr < a.min_credit + 0.5:
        raise SystemExit(f"REFUSED: credit {cr:.2f} is within $0.50 of the floor {a.min_credit:.2f}")
    crit = O.Criteria(gpu_name=a.gpu, max_dph=a.max_dph, min_cuda=a.min_cuda)
    # Offer ids are UNSTABLE between calls (2026-10-12: a search names ONE representative offer per machine, and
    # the same query named different ids on consecutive calls; two ids from a search were absent from the next).
    # So `--offer auto` searches afresh and tries the ranked survivors in order; a typed id is looked up BY ID.
    if a.offer == "auto":
        cands, _why = O.select(api.search_offers(O.search_query(crit)), crit)
        skip = {int(m) for m in str(a.exclude_machines or "").split(",") if m.strip()}
        cands = [o for o in cands if int(o.get("machine_id") or 0) not in skip][:a.tries]
    else:
        q = dict(O.search_query(crit), id={"eq": int(a.offer)})
        cands = [o for o in api.search_offers(q) if int(o["id"]) == int(a.offer)]
        if not cands:
            raise SystemExit(f"REFUSED: offer {a.offer} is not in a fresh search (taken, or not a {a.gpu})")
    iid, offer, t0 = 0, {}, 0.0
    for offer in cands:
        why = O.reject_reason(offer, crit)
        if why:
            raise SystemExit(f"REFUSED: offer {offer['id']} fails the criteria: {why}")
        t0 = time.time()
        try:
            iid = api.create_instance(int(offer["id"]), image=a.image, disk_gb=a.disk, label=a.label)
            break
        except api.VastError as e:
            _say(f"offer {offer['id']} not taken: {e}")
    if not iid:
        raise SystemExit(f"no instance created ({len(cands)} offer(s) tried)")
    deadline = t0 + a.max_minutes * 60
    try:
        pid = spawn_deadman(iid, deadline, a.min_credit)
        time.sleep(2.0)
        os.kill(pid, 0)                                   # the deadman is alive, or this raises
    except Exception as e:  # noqa: BLE001 — no backstop means no instance
        _say(f"the deadman did not start ({e!r}) — destroying {iid} now")
        ledger("destroy", instance=iid, verified_gone=destroy_and_verify(iid), by="create-no-deadman")
        raise SystemExit(f"REFUSED: instance {iid} destroyed because its deadman did not start")
    ledger("create", instance=iid, offer=O.summary(offer), image=a.image, disk=a.disk, credit_before=cr,
           deadman_pid=pid, deadline=int(deadline))
    _say(f"CREATED instance {iid} on offer {offer['id']} ({offer.get('geolocation')}, {offer.get('cpu_name')}, "
         f"${offer.get('dph_total'):.3f}/h); deadman pid {pid} destroys it by "
         f"{time.strftime('%H:%M:%S', time.localtime(deadline))} or below ${a.min_credit:.2f} credit")
    print(json.dumps({"instance": iid, "created_at": t0, "deadman_pid": pid}))
    return 0


def _sshd_answers(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=5) as s:
            s.settimeout(5)
            return s.recv(4).startswith(b"SSH")
    except OSError:
        return False


def cmd_wait(a: argparse.Namespace) -> int:
    t0 = time.time()
    last = None
    while time.time() - t0 < a.timeout:
        inst = api.instance(a.instance) or {}
        st = (inst.get("actual_status"), inst.get("status_msg"))
        if st != last:
            _say(f"+{time.time() - t0:5.0f}s status={st[0]} msg={str(st[1] or '')[:120]!r}")
            last = st
        ep = ssh_endpoint(inst)
        if inst.get("actual_status") == "running" and ep and _sshd_answers(*ep):
            r = subprocess.run(ssh_base(*ep) + ["echo ok"], capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and "ok" in r.stdout:
                _say(f"READY after {time.time() - t0:.0f}s: ssh root@{ep[0]} -p {ep[1]} (key {KEY_PATH})")
                print(json.dumps({"host": ep[0], "port": ep[1], "ready_s": round(time.time() - t0, 1)}))
                return 0
        time.sleep(10)
    _say(f"NOT READY after {a.timeout}s")
    return 1


def cmd_ssh(a: argparse.Namespace) -> int:
    host, port = _endpoint_or_die(a.instance)
    cmd = " ".join(a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd)
    if a.detach:
        # A plain `nohup setsid ... &` kept the ssh session open until the job ended (2026-10-12): `setsid -f`
        # forks the job into its own session with every fd redirected, so the ssh call returns at once.
        q = cmd.replace("'", "'\"'\"'")
        cmd = f"setsid -f bash -c '{q}' > {a.detach} 2>&1 < /dev/null; echo detached to {a.detach}"
    return subprocess.run(ssh_base(host, port)[:1] + ["-n"] + ssh_base(host, port)[1:] + [cmd]).returncode


def cmd_scp(a: argparse.Namespace) -> int:
    host, port = _endpoint_or_die(a.instance)
    src, dst = ((a.src, f"root@{host}:{a.dst}") if a.direction == "to" else (f"root@{host}:{a.src}", a.dst))
    t0 = time.time()
    rc = subprocess.run(ssh_base(host, port, scp=True) + [src, dst]).returncode
    _say(f"scp {a.direction} rc={rc} in {time.time() - t0:.1f}s")
    return rc


def destroy_and_verify(instance_id: int, tries: int = 40, every_s: float = 15.0) -> bool:
    for k in range(tries):
        ids = [int(i.get("id")) for i in api.instances()]
        if instance_id not in ids:
            return True
        try:
            api.destroy_instance(instance_id)
        except api.VastError as e:
            _say(f"destroy {instance_id} attempt {k + 1}: {e}")
        time.sleep(every_s if k else 3.0)
    return instance_id not in [int(i.get("id")) for i in api.instances()]


def cmd_destroy(a: argparse.Namespace) -> int:
    ok = destroy_and_verify(a.instance)
    cr = api.credit()
    ledger("destroy", instance=a.instance, verified_gone=ok, credit_after=cr, by="cli")
    _say(f"instance {a.instance} {'GONE (verified)' if ok else 'STILL LISTED'}; credit {cr:.4f}")
    return 0 if ok else 1


def cmd_deadman(a: argparse.Namespace) -> int:
    """The backstop. Runs detached; destroys at the deadline or below the credit floor; exits once it is gone."""
    _say(f"deadman armed for {a.instance}: deadline {time.strftime('%H:%M:%S', time.localtime(a.deadline))}, "
         f"credit floor {a.min_credit:.2f}")
    reason = "deadline"
    while time.time() < a.deadline:
        try:
            ids = [int(i.get("id")) for i in api.instances()]
            if a.instance not in ids:
                _say(f"{a.instance} is gone already — deadman exits")
                return 0
            cr = api.credit()
            if cr < a.min_credit:
                reason = f"credit {cr:.4f} < floor {a.min_credit:.2f}"
                break
        except Exception as e:  # noqa: BLE001 — a transient API error must never kill the backstop
            _say(f"poll error (kept going): {e}")
        time.sleep(min(60.0, max(1.0, a.deadline - time.time())))
    _say(f"DESTROYING {a.instance}: {reason}")
    while True:
        try:
            ok = destroy_and_verify(a.instance)
            ledger("destroy", instance=a.instance, verified_gone=ok, by="deadman", reason=reason,
                   credit_after=api.credit())
            if ok:
                _say(f"{a.instance} destroyed (verified)")
                return 0
        except Exception as e:  # noqa: BLE001
            _say(f"destroy error (retrying): {e}")
        time.sleep(30)


def cmd_instances(_a: argparse.Namespace) -> int:
    rows = api.instances()
    for i in rows:
        print(json.dumps({k: i.get(k) for k in ("id", "actual_status", "geolocation", "gpu_name", "dph_total",
                                                 "start_date", "label")}))
    _say(f"{len(rows)} instance(s)")
    return 0


def cmd_verify_zero(_a: argparse.Namespace) -> int:
    rows = api.instances()
    cr = api.credit()
    _say(f"instances: {len(rows)} {[i.get('id') for i in rows]}; credit {cr:.4f}")
    return 0 if not rows else 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.cloud.vast.cli", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("credit").set_defaults(fn=cmd_credit)
    s = sub.add_parser("search")
    s.add_argument("--gpu", default="RTX 5090")
    s.add_argument("--max-dph", type=float, default=1.0)
    s.add_argument("--min-reliability", type=float, default=0.98)
    s.add_argument("--min-cuda", type=float, default=12.8,
                   help="the driver's CUDA floor (12.8 runs the cu128 stack; a cu126 box may take 12.6)")
    s.add_argument("--show", type=int, default=15)
    s.add_argument("--out", default="")
    s.set_defaults(fn=cmd_search)
    sub.add_parser("ensure-key").set_defaults(fn=cmd_ensure_key)
    c = sub.add_parser("create")
    c.add_argument("--offer", required=True, help="an offer id, or 'auto' (fresh search, best ranked first)")
    c.add_argument("--tries", type=int, default=3, help="with --offer auto: how many ranked offers to try")
    c.add_argument("--exclude-machines", default="", metavar="ID[,ID]",
                   help="with --offer auto: never these machine ids (a host that misbehaved this session)")
    c.add_argument("--gpu", default="RTX 5090")
    c.add_argument("--max-dph", type=float, default=1.0)
    c.add_argument("--min-cuda", type=float, default=12.8)
    c.add_argument("--image", default=DEFAULT_IMAGE)
    c.add_argument("--disk", type=float, default=60.0)
    c.add_argument("--label", default="gen3ai-bench")
    c.add_argument("--max-minutes", type=float, required=True, help="the deadman destroys it after this")
    c.add_argument("--min-credit", type=float, required=True, help="the deadman destroys it below this credit")
    c.set_defaults(fn=cmd_create)
    w = sub.add_parser("wait")
    w.add_argument("--instance", type=int, required=True)
    w.add_argument("--timeout", type=float, default=1200.0)
    w.set_defaults(fn=cmd_wait)
    h = sub.add_parser("ssh")
    h.add_argument("--instance", type=int, required=True)
    h.add_argument("--detach", default="", metavar="REMOTE_LOG",
                   help="run the command detached on the box (its own session), output to REMOTE_LOG")
    h.add_argument("cmd", nargs=argparse.REMAINDER)
    h.set_defaults(fn=cmd_ssh)
    for d in ("to", "from"):
        q = sub.add_parser(f"scp-{d}")
        q.add_argument("--instance", type=int, required=True)
        q.add_argument("src")
        q.add_argument("dst")
        q.set_defaults(fn=cmd_scp, direction=d)
    x = sub.add_parser("destroy")
    x.add_argument("--instance", type=int, required=True)
    x.set_defaults(fn=cmd_destroy)
    m = sub.add_parser("deadman")
    m.add_argument("--instance", type=int, required=True)
    m.add_argument("--deadline", type=float, required=True, help="epoch seconds")
    m.add_argument("--min-credit", type=float, required=True)
    m.set_defaults(fn=cmd_deadman)
    sub.add_parser("instances").set_defaults(fn=cmd_instances)
    sub.add_parser("verify-zero").set_defaults(fn=cmd_verify_zero)
    a = p.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
