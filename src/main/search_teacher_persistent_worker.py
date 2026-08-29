"""PERSISTENT search-teacher worker — the supply + pool levers fused.

Unlike the per-cycle worker (`main.search_teacher_worker`, reads eval traces, dies after one slice),
this is a LONG-LIVED loop that GENERATES its own fresh losses (frozen trainee vs a sampled current
opponent, recorded via the eval path) and searches them — so the buffer gets a CONTINUOUS stream
instead of a 2M-step burst, and the workers never sit idle between eval cycles. Because the worker
CHOSE the opponent, the exact-opponent is known directly (no sentinel-resolution fragility).

Lifecycle: poll a control file each iteration — reload the model when the parent RE-FREEZES the
snapshot (so a long-lived worker never drifts far from the live policy), exit on shutdown. Publishes
corrections as numbered shards the parent ingests incrementally. One battle-generation + search slice
per iteration; never touches the training hot path (a frozen-snapshot side activity, like eval).

config: {run_dir, control_path, output_dir, worker_id, opponents:[{label,kind,path?}],
         n_battles, per_iter_budget, depth, beam, top_k, confirm_rollouts, margin_min, gamma,
         impl}   # impl = "node" (default) | "rust": which sim engine the generation battles and
                 # the search/replay children run on (the run's --use-bridge impl, threaded down)
control (parent-written, atomically): {snapshot_path, version, shutdown}
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time

import numpy as np


def _read_control(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _silence(model):
    for mod in model.policy.modules():
        if hasattr(mod, "_debugger"):
            mod._debugger = None


def _publish_shard(output_dir, wid, seq, corrections):
    """Atomically publish one iteration's corrections (obs/mask [+ OPD π'] .npz + scalars .json, .json
    renamed LAST so the parent never ingests a half-written shard)."""
    if not corrections:
        return
    base = os.path.join(output_dir, f"corr_{wid}_{seq}")
    arrays = dict(
        obs=np.stack([c.obs for c in corrections]).astype(np.float32),
        mask=np.stack([c.action_mask for c in corrections]).astype(np.int8))
    # OPD: pack π' as a [n, 11] array (NaN row = None); omitted when no correction has a target (AWR-only).
    if any(c.pi_target is not None for c in corrections):
        arrays["pi_target"] = np.stack([
            c.pi_target if c.pi_target is not None else np.full(11, np.nan, np.float32)
            for c in corrections]).astype(np.float32)
    np.savez(base + ".npz", **arrays)
    tmp = base + ".json.tmp"
    with open(tmp, "w") as f:
        json.dump({"scalars": [c.as_record() for c in corrections]}, f)
    os.replace(tmp, base + ".json")


def run(cfg_path: str) -> None:
    from sb3_contrib import MaskablePPO
    from agents.model.compile_opponents import maybe_compile_extractor
    from poke_env.ps_client import AccountConfiguration
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
    from agents.observation.state_encoder import load_mappings
    from agents.training.eval_callback import build_eval_players, build_eval_opponents
    from agents.training.reward_manager import Gen3RewardManager, RewardConfig
    from agents.inference.player import RLPlayer
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder
    from utils.bridge.search_session import SearchSession
    from agents.training.teacher.generate import generate_loss_traces
    from agents.training.teacher.modes import produce_for_mode, select_for_mode
    from main.prober.session import ProbeSession

    with open(cfg_path) as f:
        cfg = json.load(f)
    run_dir, wid = cfg["run_dir"], cfg["worker_id"]
    control_path, output_dir = cfg["control_path"], cfg["output_dir"]
    opponents = cfg["opponents"]
    mappings = load_mappings()
    pool = TeamLoader().get_all_teams()
    ttb, otb = Gen3Teambuilder(pool), Gen3Teambuilder(pool)
    try:
        with open(os.path.join(run_dir, "model_config.json")) as f:
            rcfg = json.load(f)
    except (OSError, ValueError):
        rcfg = {}
    rfac = functools.partial(Gen3RewardManager, config=RewardConfig.from_dict(rcfg))
    gamma = cfg.get("gamma", rcfg.get("gamma", 0.99))
    # deterministic-but-distinct opponent pick per worker (no Date/random in scripts is a SCRIPT rule,
    # not here — but keep it simple + worker-distinct so the pool is covered).
    rng = np.random.default_rng(1000 + int(wid))

    # A parent crash mid-write can strand a control.json.tmp; never let the poller read it as control.
    try:
        os.remove(os.path.join(os.path.dirname(control_path), "control.json.tmp"))
    except OSError:
        pass

    model = trainee = None
    loaded_version = -1
    snapshot_path = None
    seq = 0
    read_fails = 0
    timeout = float(cfg.get("timeout", 300.0))
    # Follows the run's --compile-opponents; this worker is pure frozen-model inference.
    compile_extractor = bool(cfg.get("compile_extractor", False))
    recycle_every = int(cfg.get("recycle_every", 2000))   # Node V8-heap backstop (launcher 3h owns the rest)
    # Which sim engine every child of this worker runs: the battle GENERATION (run_local_battles),
    # the searches (SearchSession) and the replay/re-roll driver (ProbeSession). One value, so a
    # correction is never half-produced on one engine and half on the other.
    impl = cfg.get("impl", "node")
    # --search-teacher-mode (ai_v12 routes 2+3). Absent ⇒ "crater" — an older parent's
    # config runs exactly as it did.
    mode = cfg.get("mode", "crater")
    wp_band = float(cfg.get("wp_band", 0.15))
    wp_margin = float(cfg.get("wp_margin", 0.02))
    ss = SearchSession(timeout=timeout, impl=impl)
    try:
        while True:
            ctrl = _read_control(control_path)
            if ctrl is None:                        # missing/half-written/corrupt control → wait, but log a stall
                read_fails += 1
                if read_fails in (10, 60) or (read_fails > 60 and read_fails % 120 == 0):
                    print(f"[SearchTeacher worker {wid}] control unreadable for ~{read_fails}s",
                          file=sys.stderr, flush=True)
                time.sleep(1.0); continue
            read_fails = 0
            if ctrl.get("shutdown"):
                break
            version = int(ctrl.get("version", 0))
            if version != loaded_version:           # parent re-froze → reload the trainee snapshot
                snap = ctrl.get("snapshot_path") or ""
                if not snap or not os.path.exists(snap):
                    time.sleep(0.5); continue        # pruned before we loaded it → re-read control next loop
                try:
                    model = MaskablePPO.load(snap, env=None, device="cpu")
                except Exception as e:               # noqa: BLE001 — a corrupt/partial save → wait for the next re-freeze
                    print(f"[SearchTeacher worker {wid}] snapshot load failed: {e}",
                          file=sys.stderr, flush=True)
                    time.sleep(1.0); continue
                _silence(model)
                # Frozen trainee, CPU, B=1 — the same shape as a training opponent, and this worker
                # does nothing BUT forwards. Compile is gated on the run's --compile-opponents,
                # threaded through the control file so the worker follows the parent's setting.
                maybe_compile_extractor(model, compile_extractor, label=f"searchteacher{wid}:trainee",
                                        hide_cuda=True)
                trainee = build_eval_players(
                    model, ["t"], ttb, mappings, LocalhostServerConfiguration, 1, f"TG{wid}",
                    start_listening=False, gamma=gamma, reward_fn_factory=rfac)["t"]
                loaded_version = version
                snapshot_path = snap

            # --- pick an opponent + build it (the worker KNOWS the exact opponent it plays) ---
            spec = opponents[int(rng.integers(len(opponents)))]
            opp = None
            try:
                if spec["kind"] == "bot":
                    opp = build_eval_opponents(LocalhostServerConfiguration, otb, [spec["label"]],
                                               f"TG{wid}", start_listening=False)[0][1]
                    opp_ckpt, opp_src = None, "bot"
                else:                                # 'sentinel' / 'self' snapshot — may have been pruned by now
                    if not os.path.exists(spec["path"]):
                        seq += 1; continue
                    om = MaskablePPO.load(spec["path"], env=None, device="cpu"); _silence(om)
                    # Loaded EVERY iteration, so this leans on two properties: torch.compile keys on
                    # the code object (so it is ~free after the first), and the helper's timing
                    # validation is paid once per process.
                    maybe_compile_extractor(om, compile_extractor,
                                            label=f"searchteacher{wid}:opp", hide_cuda=True)
                    opp = RLPlayer(model=om, team=otb, battle_format="gen3ou",
                                   server_configuration=LocalhostServerConfiguration, mappings=mappings,
                                   account_configuration=AccountConfiguration(f"TGo{wid}", "pw"),
                                   max_concurrent_battles=1, stochastic=True, temperature=1.0,
                                   start_listening=False)
                    opp_ckpt, opp_src = spec["path"], "ckpt"
            except Exception as e:                   # noqa: BLE001 — a stale/corrupt opponent skips the iter, never crashes the worker
                print(f"[SearchTeacher worker {wid}] opponent {spec.get('label')} load failed: {e}",
                      file=sys.stderr, flush=True)
                seq += 1; continue

            iter_dir = os.path.join(output_dir, f"gen_{wid}_{seq}", f"step_{version}", spec["label"])
            traces = generate_loss_traces(
                trainee, opp, out_dir=iter_dir, n_battles=int(cfg["n_battles"]), step=version,
                opponent_label=spec["label"], opponent_ckpt=opp_ckpt, opponent_source=opp_src,
                impl=impl)

            corrections = []
            if traces:
                gen_root = os.path.join(output_dir, f"gen_{wid}_{seq}")
                # one ProbeSession per iter caches a model per checkpoint → close it (context manager) so a
                # long-lived worker doesn't accumulate model objects across thousands of iterations.
                with ProbeSession(gen_root, ckpt_override=snapshot_path, impl=impl) as sess:
                    # plentiful fresh supply → skip the expensive falsify gate; the CONFIRM is the real gate.
                    cands = select_for_mode(mode, gen_root, budget=int(cfg["per_iter_budget"]),
                                            scan_limit=int(cfg["per_iter_budget"]),
                                            falsify_gate=False, window=1, wp_band=wp_band,
                                            session=sess)
                    for c in cands:
                        try:
                            corr, _ = produce_for_mode(
                                mode, sess, c, opponent_ckpt=opp_ckpt, opponent_source=opp_src,
                                confirm_rollouts=int(cfg["confirm_rollouts"]), depth=int(cfg["depth"]),
                                beam=int(cfg["beam"]), top_k=int(cfg["top_k"]),
                                margin_min=float(cfg["margin_min"]), wp_margin=wp_margin,
                                search_session=ss,
                                build_pi_target=bool(cfg.get("opd_build_pi_target", False)),
                                opd_beta=float(cfg.get("opd_beta", 1.0)))
                        except Exception:  # noqa: BLE001 — one bad candidate never kills the loop
                            corr = None
                        if corr is not None:
                            corrections.append(corr)
            _publish_shard(output_dir, wid, seq, corrections)
            _cleanup(os.path.join(output_dir, f"gen_{wid}_{seq}"))
            opp = None                               # drop the opponent player (+ its loaded model) promptly
            seq += 1
            if recycle_every and seq % recycle_every == 0:   # bound the Node child's V8 heap on a no-launcher run
                ss.close(); ss = SearchSession(timeout=timeout, impl=impl)
    finally:
        ss.close()


def _cleanup(d):
    import shutil
    try:
        shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


if __name__ == "__main__":
    run(sys.argv[1])
