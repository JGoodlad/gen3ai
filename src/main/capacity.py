"""main.capacity — the CAPACITY-EVAL BATTERY: is the shared trunk filling up?

The flywheel era piles distilled skills into ONE fixed-capacity network — conditioning was closed
by two independent nulls, so there is no FiLM and no LoRA to grow into. That makes saturation a
thing to WATCH rather than to discover after a long fruitless hunt. This is the watch: an offline,
read-only battery over one checkpoint, ~minutes, emitting a printed table and a JSON artifact
built for GENERATION-OVER-GENERATION differencing at matched step.

    python -m main.capacity models/<run>                       # picks the run's newest checkpoint
    python -m main.capacity models/<run>/legB_final_model.zip   # or name the arm explicitly
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Four probes, all defined in `agents.model.capacity_probes`:

  (a) effective RANK at five taps (participation ratio + srank@0.99), trained vs a fresh net
  (b) TRAINABILITY — Lyle et al. capacity loss: linear fit to K fixed random targets, and the
      trained/fresh ratio that says whether capacity has been consumed
  (c) DECODABILITY — linear probes to ground-truth facts read out of the obs vector itself
  (d) per-phase PARAMETER census

🚨 **No number here licenses a kill or a build on its own, and that rule is bought with a
retraction.** A "conditioning headroom" claim was once derived from a low participation ratio
(`PR(K_ū)=17`); it was a noise artifact and the lever built on it was refuted. The lesson —
*gate a lever on "does this quantity PREDICT performance", never on "is it low"* — is why every
metric ships a `validity` note naming what movement would mean and what PAIRED behavioural
evidence would have to confirm it. This is an early-warning TRIPWIRE whose alarms get
investigated. Full notes: `designs/research_state/capacity_battery.md`.

States come from the run's OWN `eval_traces/` via the shared stratified sampler
(`agents.model.audit_states.collect_states` — round-robin over step dirs × opponents, then a
seeded row subsample), so the sample is reproducible and its coverage is written into the
artifact for a reader to VERIFY rather than trust.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

#: What each metric would MEAN if it moved, and what would have to confirm it. Shipped inside the
#: artifact so a reader six months from now cannot get the number without the caveat.
VALIDITY: Dict[str, Dict[str, str]] = {
    "rank": {
        "movement": "A FALLING participation ratio / srank@0.99 at matched step across "
                    "generations = the representation is collapsing into fewer directions, the "
                    "classic precursor to capacity loss.",
        "confirm": "PAIRED behavioural evidence is required: anchored ELO at matched snapshot "
                   "COUNT (ladder.json, end-of-run), and a per-skill retention read on the "
                   "distilled behaviours. Rank that falls while ELO and retention hold is a "
                   "representation becoming MORE efficient, not a network running out of room.",
        "no_verdict": "No kill/build decision from this number alone. PR(K_u)=17 was 'low', was "
                      "noise, and its lever was refuted by two independent nulls.",
    },
    "trainability": {
        "movement": "capacity_ratio = r2_trained / r2_fresh. Falling well below 1 and CONTINUING "
                    "to fall generation over generation = the trained features have lost the "
                    "ability to express new functions (Lyle et al. capacity loss).",
        "confirm": "A single generation's ratio is uncalibrated — the trained net is not supposed "
                   "to match a random net at random targets. Only the TREND at matched step is "
                   "readable, and it must be paired with a trainability test that has stakes: a "
                   "fresh exploiter fork's learning curve on a NEW skill vs the same fork from an "
                   "earlier generation.",
        "no_verdict": "No kill/build decision from this number alone. A low ratio licenses "
                      "running the exploiter-fork trainability test, nothing else.",
    },
    "decodability": {
        "movement": "These facts are deliberately EASY (each is an obs coordinate or a count over "
                    "six of them), so a high r²/AUC is the EXPECTED reading and says nothing. The "
                    "signal is DRIFT: an established fact becoming less decodable across "
                    "generations at matched step = capacity being reallocated away from it.",
        "confirm": "Decodable != used, and this repo has the measurement — the belief-latent "
                   "role-geometry probe found SPECIES geometry strongly decodable and the "
                   "move-id table not at all, and neither predicted whether the head HELPED. So a "
                   "decodability drop needs an ablation or an intervention showing the policy's "
                   "behaviour actually depends on the dropped fact.",
        "no_verdict": "No kill/build decision from this number alone.",
    },
    "params": {
        "movement": "A phase whose rms is ~unchanged from init while every sibling has grown is a "
                    "phase that is not learning; zero_frac ~1.0 on a zero-init route means it has "
                    "never left identity.",
        "confirm": "Weight norm is not usage. The per-edge-family liveness metrics and the "
                   "route-ablation audits (critic_route_audit, edge_ablation_audit) are the "
                   "instruments that price a route in |dV| / argmax-flip terms; this census only "
                   "says where to point them.",
        "no_verdict": "No kill/build decision from this number alone.",
    },
}


# --------------------------------------------------------------------------- resolution

def resolve_checkpoint(path: str) -> "tuple[str, str]":
    """→ ``(checkpoint_zip, run_dir)``.

    A ``.zip`` is taken verbatim — deliberately, because a run can carry several ARMS at its root
    (`legA_final_model.zip` / `legB_final_model.zip` / `final_model.zip`) and guessing between
    them would silently measure the wrong one. A DIRECTORY resolves through `latest.txt` (the
    run-relative pin), then `final_model.zip`, then the newest `checkpoints/*.zip`.
    """
    if path.endswith(".zip"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"no checkpoint at {path!r}")
        return path, os.path.dirname(os.path.abspath(path))
    run = os.path.abspath(path)
    if not os.path.isdir(run):
        raise FileNotFoundError(f"{path!r} is neither a .zip nor a directory")
    latest = os.path.join(run, "latest.txt")
    if os.path.exists(latest):
        with open(latest) as fh:
            rel = fh.read().strip()
        cand = rel if os.path.isabs(rel) else os.path.join(run, rel)
        if os.path.exists(cand):
            return cand, run
    final = os.path.join(run, "final_model.zip")
    if os.path.exists(final):
        return final, run
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoints", "*.zip")), key=os.path.getmtime)
    if ckpts:
        return ckpts[-1], run
    raise FileNotFoundError(
        f"{run!r} carries no checkpoint (no usable latest.txt, no final_model.zip, no "
        "checkpoints/*.zip). Name the .zip explicitly if the run keeps its arms under other names.")


def load_policy(ckpt: str, run_dir: str, device: str) -> "tuple[Any, Dict[str, Any], str]":
    """Load the checkpoint through `load_model_snapshot`, which runs `check_compatible`.

    ``current_version`` is built from the SAVED config's own arch TOGGLES (matched against
    `current_model_version`'s signature), so the compatibility check is exactly the arch-DRIFT
    question: given the run's own flags, does TODAY'S code build the same architecture? The
    weight-shape fields the toggles do not cover — obs `total_dim`, `arch_signature`, every
    embedding dim — come from live code and are what the check compares. A toggle the current
    code no longer accepts is not passed, and whatever structural difference that implies is
    then caught by the same check rather than papered over.
    """
    from agents.model.model_version import ModelVersionError
    from agents.model.snapshot import current_model_version, load_model_snapshot
    from agents.observation.state_encoder import load_mappings
    import inspect

    cfg_path = None
    for d in (os.path.dirname(os.path.abspath(ckpt)),
              os.path.dirname(os.path.dirname(os.path.abspath(ckpt))), run_dir):
        cand = os.path.join(d, "model_config.json")
        if os.path.exists(cand):
            cfg_path = cand
            break
    if cfg_path is None:
        raise FileNotFoundError(
            f"no model_config.json beside {ckpt!r} or at the run root — the battery refuses to "
            "load a checkpoint whose architecture provenance is unknown, because every number it "
            "emits is labelled with that architecture.")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    sig = inspect.signature(current_model_version).parameters
    toggles = {k: v for k, v in cfg.items() if k in sig and k != "mappings"}
    try:
        version = current_model_version(load_mappings(), **toggles)
        model = load_model_snapshot(ckpt, env=None, current_version=version, device=device)
    except ModelVersionError as exc:
        raise SystemExit(
            "\n[capacity] ARCHITECTURE DRIFT — this checkpoint cannot be loaded by the current "
            f"tree.\n\n{exc}\n\n"
            f"  saved arch_signature : {cfg.get('arch_signature')}\n"
            f"  saved total_dim      : {cfg.get('total_dim')}\n"
            f"  saved config_version : {cfg.get('config_version')}\n\n"
            "The battery is only meaningful under the code that trained the checkpoint. Re-run "
            "from the run's own pinned worktree:\n"
            "  git worktree add /tmp/pin $(python -c \"import json;print(json.load(open('"
            f"{os.path.join(run_dir, 'metadata.json')}'))['git_hash'])\")\n") from exc
    return model, cfg, cfg_path


# --------------------------------------------------------------------------- rendering

def _fmt(v: Any, spec: str = "8.3f") -> str:
    if v is None:
        return f"{'-':>{spec.split('.')[0].strip('<>')}}"
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def render(report: Dict[str, Any]) -> str:
    """The human table. Same content as the JSON, ordered for reading top to bottom."""
    lines: List[str] = []
    m = report["meta"]
    lines.append("=" * 78)
    lines.append(f"CAPACITY BATTERY v{report['battery_version']}   {m['run_name']}")
    lines.append(f"  checkpoint {os.path.basename(m['checkpoint'])}   step {m['num_timesteps']:,}"
                 f"   arch {m['arch_signature']}")
    lines.append(f"  {m['n_states']} states from {m['sampling']['n_files_read']} traces "
                 f"({len(m['sampling']['per_step'])} step dirs x "
                 f"{len(m['sampling']['per_opponent'])} opponents), seed {m['seed']}")
    lines.append("=" * 78)

    lines.append("\n(a) REPRESENTATION EFFECTIVE RANK  (centered; fresh = same config, random init)")
    lines.append(f"  {'tap':<14}{'dim':>5}{'PR':>9}{'PR/dim':>8}{'srank99':>9}"
                 f"{'PR_fresh':>10}{'sr99_fresh':>12}")
    tr, fr = report["rank"]["trained"], report["rank"]["fresh"]
    for tap, r in tr.items():
        f = fr.get(tap, {})
        lines.append(f"  {tap:<14}{r['dim']:>5}{_fmt(r['pr'], '9.2f')}{_fmt(r['pr_frac'], '8.3f')}"
                     f"{_fmt(r['srank99'], '9.0f')}{_fmt(f.get('pr'), '10.2f')}"
                     f"{_fmt(f.get('srank99'), '12.0f')}")

    t = report["trainability"]
    lines.append(f"\n(b) TRAINABILITY / CAPACITY LOSS  (K={t['n_targets']} random targets, "
                 f"{t['folds']}-fold OOF ridge)")
    lines.append(f"  {'tap':<14}{'r2_trained':>12}{'r2_fresh':>10}{'nmse_tr':>10}"
                 f"{'nmse_fresh':>12}{'ratio':>8}")
    for tap, r in t["taps"].items():
        lines.append(f"  {tap:<14}{_fmt(r['r2_trained'], '12.4f')}{_fmt(r['r2_fresh'], '10.4f')}"
                     f"{_fmt(r['nmse_trained'], '10.4f')}{_fmt(r['nmse_fresh'], '12.4f')}"
                     f"{_fmt(r['capacity_ratio'], '8.3f')}")

    d = report["decodability"]
    lines.append("\n(c) PROBE DECODABILITY  (regression = OOF r2, classification = OOF AUC; "
                 "trained | fresh)")
    taps = list(report["rank"]["trained"].keys())
    lines.append(f"  {'fact':<22}{'task':<7}" + "".join(f"{t_[:12]:>15}" for t_ in taps))
    for fact, row in d["facts"].items():
        cells = "".join(
            f"{_fmt(row['taps'][t_]['trained'], '7.3f')}|{_fmt(row['taps'][t_]['fresh'], '7.3f')}"
            for t_ in taps if t_ in row["taps"])
        lines.append(f"  {fact:<22}{row['task'][:5]:<7}{cells}")
    for fact, why in d["skipped"].items():
        lines.append(f"  {fact:<22}SKIP   {why}")

    p = report["params"]
    lines.append(f"\n(d) PARAMETER CENSUS  ({p['n_params_total']:,} params, top 12 phases)")
    lines.append(f"  {'phase':<26}{'params':>12}{'share':>8}{'rms':>9}{'zero%':>8}")
    for name, row in list(p["phases"].items())[:12]:
        lines.append(f"  {name:<26}{row['n_params']:>12,}{_fmt(row['param_share'], '8.3f')}"
                     f"{_fmt(row['rms'], '9.4f')}{_fmt(100 * row['zero_frac'], '8.1f')}")

    lines.append("\n" + "-" * 78)
    lines.append("TRIPWIRE, NOT VERDICT: no kill/build decision from any number above on its own.")
    lines.append("Read a metric only as a DIFFERENCE against another generation at matched step,")
    lines.append("and confirm any alarm with the paired behavioural evidence named in the JSON's")
    lines.append("`validity` block (designs/research_state/capacity_battery.md).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- entry point

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m main.capacity",
        description="Offline capacity/saturation battery over one checkpoint.")
    ap.add_argument("target", help="a run directory, or an explicit checkpoint .zip")
    ap.add_argument("--out", default=None,
                    help="JSON path (default: <run>/capacity_battery.json)")
    ap.add_argument("--states", nargs="+", default=None,
                    help="states.npz glob(s) (default: <run>/eval_traces/**/*_states.npz)")
    ap.add_argument("--max-states", type=int, default=3000)
    ap.add_argument("--n-targets", type=int, default=8, help="K random target functions")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the state subsample, the random targets and the CV folds")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quiet", action="store_true", help="suppress progress lines")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    say = (lambda _m: None) if args.quiet else (lambda m: print(f"[capacity] {m}", flush=True))

    ckpt, run_dir = resolve_checkpoint(args.target)
    say(f"checkpoint {ckpt}")
    model, cfg, cfg_path = load_policy(ckpt, run_dir, args.device)
    policy = model.policy.eval()

    from agents.model.audit_states import collect_states
    from agents.model.capacity_probes import build_fresh_extractor, jsonable, run_battery
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from utils.git import get_git_hash

    patterns = args.states or [os.path.join(run_dir, "eval_traces", "**", "*_states.npz")]
    say(f"sampling states from {patterns[0]}")
    obs, masks, coverage = collect_states(patterns, args.max_states, seed=args.seed)
    say(f"{len(obs)} states, obs dim {obs.shape[1]}")

    fresh = build_fresh_extractor(policy, seed=args.seed)
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    body = run_battery(policy.features_extractor, fresh, obs, masks, layout,
                       n_targets=args.n_targets, folds=args.folds, seed=args.seed,
                       batch=args.batch, device=args.device, progress=say)

    report: Dict[str, Any] = {
        "meta": {
            "run_dir": run_dir,
            "run_name": os.path.basename(run_dir.rstrip("/")),
            "checkpoint": ckpt,
            "model_config": cfg_path,
            "arch_signature": cfg.get("arch_signature"),
            "config_version": cfg.get("config_version"),
            "obs_dim": int(obs.shape[1]),
            "num_timesteps": int(getattr(model, "num_timesteps", 0)),
            "n_states": int(len(obs)),
            "sampling": coverage,
            "seed": int(args.seed),
            "device": args.device,
            "tree_git_hash": get_git_hash(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        **body,
        "validity": VALIDITY,
    }
    report["meta"]["runtime_sec"] = round(time.time() - t0, 1)

    report = jsonable(report)

    print(render(report))
    out = args.out or os.path.join(run_dir, "capacity_battery.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=False, allow_nan=False)
    print(f"\nwrote {out}  ({report['meta']['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
