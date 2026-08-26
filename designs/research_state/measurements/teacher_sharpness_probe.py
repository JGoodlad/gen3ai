"""TEACHER-SHARPNESS PROBE — is a tock teacher's action distribution degenerately sharp?

The era's live working theory (ledger, 2026-08-25) says our tick-1 tock teachers are
DISTRIBUTIONALLY immature: 3M steps of specialization against one frozen opponent improved their
OUTCOMES (+9pp extraction, verified) while collapsing their action distributions into over-sharp
scripts, so KL-matching them injects overconfident narrowness through the shared trunk. The theory
makes ONE cheap falsifiable prediction: on the states the distillation KL actually fires on, a
teacher's policy entropy should sit FAR below the base model's. If it does not, the theory is in
trouble before anyone spends 9M steps on the "better tocks" prescription.

WHAT IT MEASURES — exactly the quantity the loss uses. `DistillTerms._distill_loss` masks the
illegal actions to -inf and normalises BOTH sides over the legal set, so this script does the same:
`softmax(logits + (mask-1)*1e9)` per state, then entropy / top-1 / exp(entropy) / KL(teacher || base).

PAIRING — one pilot generates the states, every model is then forwarded on the STORED obs. So the
teacher-vs-base contrast is paired on identical states, and the bootstrap CI resamples BATTLES (the
cluster), never decisions.

TWO PILOTS, both reported, because they answer different questions and the theory should survive
either:
  * `base`    — the base model pilots. This is the distill-time state distribution: the KL fires on
                states the TRAINEE visits, and the trainee is a base fork. PRIMARY.
  * `teacher` — the teacher pilots its own trajectory. The distribution the teacher actually plays.

TWO POPULATIONS: ON-PIN (the teacher's own pinned `--trainee-teams`, the exact training matchup vs
the frozen rev-1 24M snapshot the tocks were exploiters against) and OFF-PIN (pool teams in no
teacher's slice, same opponent) — so a sharpening result can be told apart from a global one.

Run:
    python designs/research_state/measurements/teacher_sharpness_probe.py \
        --battles 8 --out designs/research_state/measurements/teacher_sharpness_probe.json

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
Reads `models/` through `utils.paths.main_models_dir()`, so it runs from a worktree or the main
checkout. Set `POKESIM_SIM_BRIDGE_BIN` to a prebuilt binary to skip a `cargo build` in a fresh tree.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys

import numpy as np
import torch

from agents.inference.player import RLPlayer
from agents.model.model_version import ModelVersion
from agents.model.snapshot import load_model_snapshot
from agents.observation.state_encoder import load_mappings
from utils.bridge.local_battle_runner import run_local_battles
from utils.paths import main_models_dir, repo_path
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration

BATTLE_FORMAT = "gen3ou"

BASE_RUN = "ai_v9_29_rev1_0823"
# The frozen self-target both tocks trained as exploiters against (`--exploiter ...`).
OPPONENT_REL = f"{BASE_RUN}/snapshots/snapshot_000024000000.zip"

# `--trainee-teams` read verbatim from each tock's metadata.json `original_command`.
TEACHERS = {
    "tock1a_k4": {
        "run": "ai_v9_31_tock1_k4_0824",
        "teams": ["eccfe630ec08de27", "023a2d47648b85e6",
                  "8e768980fc8f3b5f", "710d8d529538ff90"],
    },
    "tock1c_q6": {
        "run": "ai_v9_36_tock1c_q6_0824",
        "teams": ["9eb3abdc52876a63", "e0d97b0ed592889d"],
    },
}
N_OFF_PIN_TEAMS = 6          # pool teams in NO teacher's slice
OFF_PIN_SEED = 20260825


# --------------------------------------------------------------------------- recording player
class _RecordingRLPlayer(RLPlayer):
    """An RLPlayer that appends (battle_tag, turn, obs, mask) for every COMMITTED decision.

    `embed_battle` stashes the obs; `choose_move` commits exactly one row per decision, so a stale
    re-decide (which re-embeds) leaves only its final attempt — the same rule the player's own
    turn-history restore() uses. Zero-mask polls never reach the commit (idx is None there)."""

    def __init__(self, *args, sink: list, **kwargs):
        super().__init__(*args, **kwargs)
        self._sink = sink
        self._stash = None
        self._pending = None

    def embed_battle(self, battle):
        d = super().embed_battle(battle)
        self._stash = (np.asarray(d["observation"], dtype=np.float32),
                       np.asarray(d["action_mask"], dtype=np.int8))
        return d

    def _predict_best_action(self, battle, stochastic=False, need_aux=True, temperature=1.0):
        out = super()._predict_best_action(battle, stochastic=stochastic,
                                           need_aux=need_aux, temperature=temperature)
        idx = out[0]
        if idx is not None and self._stash is not None:
            self._pending = (battle.battle_tag, int(getattr(battle, "turn", -1)),
                             self._stash[0], self._stash[1], int(idx))
        return out

    def choose_move(self, battle):
        self._pending = None
        order = super().choose_move(battle)
        if self._pending is not None:
            self._sink.append(self._pending)
            self._pending = None
        return order


# --------------------------------------------------------------------------- helpers
def _team_str(sha: str) -> str:
    with open(repo_path("data", "teams", "sample", f"{sha}.txt")) as f:
        return f.read().strip()


def _make_player(model, team_str, mappings, tag, stochastic, sink=None):
    cls = _RecordingRLPlayer if sink is not None else RLPlayer
    kw = {"sink": sink} if sink is not None else {}
    return cls(
        model=model,
        team=Gen3Teambuilder([team_str]),
        battle_format=BATTLE_FORMAT,
        server_configuration=localhost_server_configuration,
        mappings=mappings,
        account_configuration=AccountConfiguration(tag, "password"),
        max_concurrent_battles=1,
        stochastic=stochastic,
        temperature=1.0,
        start_listening=False,
        **kw,
    )


def _seed_everything(k: int) -> None:
    random.seed(k)
    np.random.seed(k % (2 ** 31))
    torch.manual_seed(k)


def _masked_probs(model, obs: np.ndarray, mask: np.ndarray, chunk: int = 256) -> np.ndarray:
    """softmax over the LEGAL actions — byte-for-byte the form `_distill_loss` normalises with."""
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), chunk):
            o = torch.as_tensor(obs[i:i + chunk]).to(model.device)
            m = torch.as_tensor(mask[i:i + chunk].astype(np.float32)).to(model.device)
            logits = model.policy.get_distribution({"observation": o, "action_mask": m}).distribution.logits
            masked = logits + (m - 1.0) * 1e9
            out.append(torch.softmax(masked, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Forward KL(p || q) — p = teacher, q = student/base, matching `_distill_loss`."""
    return (p * (np.log(np.clip(p, 1e-12, None)) - np.log(np.clip(q, 1e-12, None)))).sum(-1)


def _boot_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 7):
    """Cluster (battle) bootstrap of a mean. Returns (mean, lo, hi) at 95%."""
    if len(values) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    by = {c: values[clusters == c] for c in uniq}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        stats[b] = np.concatenate([by[c] for c in draw]).mean()
    return float(values.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# --------------------------------------------------------------------------- collection
async def _play_one(pilot_model, opp_model, team_str, mappings, sink, seed_i, impl):
    p1 = _make_player(pilot_model, team_str, mappings, f"PRB{seed_i:04d}", True, sink=sink)
    p2 = _make_player(opp_model, _OPP_TEAM_STR, mappings, f"OPP{seed_i:04d}", True)
    await run_local_battles(
        p1, p2, 1, battle_format=BATTLE_FORMAT,
        # the gen-5 seed is a quadruple of uint16 — a raw product overflows and the bridge REFUSES it
        seed=[(seed_i * m + c) % 65536 for m, c in ((7, 1), (13, 2), (17, 3), (19, 4))],
        impl=impl,
    )


def collect(pilot_model, opp_model, teams, mappings, n_battles, seed0, impl, label):
    """Play `n_battles`, rotating deterministically through `teams`. Returns the recorded rows."""
    sink: list = []
    for i in range(n_battles):
        seed_i = seed0 + i
        _seed_everything(seed_i)
        team_str = teams[i % len(teams)]
        before = len(sink)
        try:
            asyncio.run(_play_one(pilot_model, opp_model, team_str, mappings, sink, seed_i, impl))
        except Exception as exc:                      # a wedged battle must not void the cell
            print(f"    [{label}] battle {i} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"    [{label}] battle {i}: +{len(sink) - before} decisions "
              f"({len(sink)} total)", flush=True)
    return sink


# --------------------------------------------------------------------------- analysis
def summarise(probs: np.ndarray, mask: np.ndarray) -> dict:
    n_legal = mask.sum(-1)
    ent = _entropy(probs)
    top1 = probs.max(-1)
    keep = n_legal >= 2                      # a forced decision carries no distributional signal
    e, t, nl = ent[keep], top1[keep], n_legal[keep]
    return {
        "n_states": int(len(probs)),
        "n_forced": int((~keep).sum()),
        "mean_n_legal": round(float(nl.mean()), 3) if len(nl) else None,
        "mean_entropy_nats": round(float(e.mean()), 4) if len(e) else None,
        "median_entropy_nats": round(float(np.median(e)), 4) if len(e) else None,
        "eff_n_actions": round(float(np.exp(e).mean()), 3) if len(e) else None,
        "entropy_frac_of_uniform": (round(float((e / np.log(nl)).mean()), 4) if len(e) else None),
        "top1_median": round(float(np.median(t)), 4) if len(t) else None,
        "top1_p90": round(float(np.percentile(t, 90)), 4) if len(t) else None,
        "top1_over_0p9_frac": round(float((t > 0.9).mean()), 4) if len(t) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--battles", type=int, default=8, help="battles per (teacher, population, pilot) cell")
    ap.add_argument("--impl", default="rust", choices=("rust", "node"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pilots", default="base,teacher")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models_dir = main_models_dir()
    if models_dir is None:
        print("No models/ archive found (set $GEN3AI_MODELS_DIR).", file=sys.stderr)
        return 2

    mappings = load_mappings()
    # Every run here carries the SAME model_config (gen3_critic_route_wave_v1, v101) — verified from
    # the three model_config.json files. Using the base's as `current_version` therefore keeps
    # `check_compatible` live (it would REFUSE a drifted teacher) without reconstructing 60 toggles.
    cfg = ModelVersion.from_json_file(str(models_dir / BASE_RUN / "model_config.json"))

    def _load(rel):
        m = load_model_snapshot(str(models_dir / rel), env=None,
                                current_version=cfg, device=args.device)
        m.policy.set_training_mode(False)
        # These are `--log-level periodic` checkpoints: each carries an ObservationDebugger that
        # print()s a full decoded board every 30 s of forwards. It is a learner-side diagnostic and
        # would drown this script's own output.
        for mod in m.policy.modules():
            if hasattr(mod, "disable_observation_debugger"):
                mod.disable_observation_debugger()
        return m

    print("loading models ...", flush=True)
    base = _load(f"{BASE_RUN}/final_model.zip")
    opp = _load(OPPONENT_REL)
    teachers = {k: _load(f"{v['run']}/final_model.zip") for k, v in TEACHERS.items()}

    # The frozen opponent pilots the FULL pool exactly as it did in training (`opp_tb` there is the
    # all-teams builder); one deterministic draw keeps the matchup fixed across cells.
    loader = TeamLoader(base_dir=repo_path("data", "teams"))
    all_teams = loader.get_all_teams()
    pinned = {sha for v in TEACHERS.values() for sha in v["teams"]}
    pinned_strs = {_team_str(s) for s in pinned}
    rng = random.Random(OFF_PIN_SEED)
    global _OPP_TEAM_STR
    _OPP_TEAM_STR = rng.choice([t for t in all_teams if t not in pinned_strs])
    off_pin = rng.sample([t for t in all_teams if t not in pinned_strs and t != _OPP_TEAM_STR],
                         N_OFF_PIN_TEAMS)
    print(f"pool={len(all_teams)} teams · off-pin sample={len(off_pin)} · "
          f"pinned excluded={len(pinned)}", flush=True)

    pilots = [p for p in args.pilots.split(",") if p]
    rows: dict[tuple, list] = {}
    seed_cursor = 1000
    for tname, tinfo in TEACHERS.items():
        on_pin_teams = [_team_str(s) for s in tinfo["teams"]]
        for pop, teams in (("on_pin", on_pin_teams), ("off_pin", off_pin)):
            for pilot in pilots:
                pilot_model = base if pilot == "base" else teachers[tname]
                label = f"{tname}/{pop}/pilot={pilot}"
                print(f"  collecting {label} ...", flush=True)
                rows[(tname, pop, pilot)] = collect(
                    pilot_model, opp, teams, mappings, args.battles, seed_cursor, args.impl, label)
                seed_cursor += 1000

    # ---- score every cell with every model on the SAME stored obs ----
    report: dict = {
        # measurements/README.md convention: every file here carries its own scope, so a stale
        # citation is detectable by date rather than by re-deriving the result.
        "provenance": {
            "generation": "tick-1 (gen-18 era)",
            "run": f"{BASE_RUN} + " + " + ".join(v["run"] for v in TEACHERS.values()),
            "checkpoint": "final_model.zip (each); opponent = " + OPPONENT_REL,
            "step": "base 25M · teachers base+3M · opponent 24M",
            "date": __import__("datetime").date.today().isoformat(),
            "producer": "designs/research_state/measurements/teacher_sharpness_probe.py",
            "note": "policy-distribution sharpness only — no outcome/ELO claim is made here",
        },
        "meta": {
            "base_run": BASE_RUN, "opponent": OPPONENT_REL,
            "teachers": {k: v["run"] for k, v in TEACHERS.items()},
            "pinned_teams": {k: v["teams"] for k, v in TEACHERS.items()},
            "battles_per_cell": args.battles, "impl": args.impl, "device": args.device,
            "n_off_pin_teams": N_OFF_PIN_TEAMS, "off_pin_seed": OFF_PIN_SEED,
            "arch_signature": getattr(cfg, "arch_signature", None),
        },
        "cells": {},
    }
    for (tname, pop, pilot), sink in rows.items():
        key = f"{tname}|{pop}|pilot={pilot}"
        if not sink:
            report["cells"][key] = {"error": "no decisions recorded"}
            continue
        tags = np.array([r[0] for r in sink])
        obs = np.stack([r[2] for r in sink])
        mask = np.stack([r[3] for r in sink])
        p_teacher = _masked_probs(teachers[tname], obs, mask)
        p_base = _masked_probs(base, obs, mask)

        n_legal = mask.sum(-1)
        keep = n_legal >= 2
        d_ent = (_entropy(p_teacher) - _entropy(p_base))[keep]
        kl_tb = _kl(p_teacher, p_base)[keep]
        d_top1 = (p_teacher.max(-1) - p_base.max(-1))[keep]
        clusters = tags[keep]

        m, lo, hi = _boot_ci(d_ent, clusters)
        km, klo, khi = _boot_ci(kl_tb, clusters)
        tm, tlo, thi = _boot_ci(d_top1, clusters)
        report["cells"][key] = {
            "n_battles": int(len(np.unique(tags))),
            "teacher": summarise(p_teacher, mask),
            "base": summarise(p_base, mask),
            "paired": {
                "delta_entropy_nats": None if m is None else round(m, 4),
                "delta_entropy_ci95": None if m is None else [round(lo, 4), round(hi, 4)],
                "delta_top1": None if tm is None else round(tm, 4),
                "delta_top1_ci95": None if tm is None else [round(tlo, 4), round(thi, 4)],
                "kl_teacher_base": None if km is None else round(km, 4),
                "kl_teacher_base_ci95": None if km is None else [round(klo, 4), round(khi, 4)],
                "n_scored": int(keep.sum()),
            },
        }
        print(f"  {key}: dH={m:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"KL={km:.4f} n={int(keep.sum())} over {len(np.unique(tags))} battles", flush=True)

    report["provenance"]["n_states"] = sum(len(s) for s in rows.values())
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "teacher_sharpness_probe.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    return 0


_OPP_TEAM_STR = ""

if __name__ == "__main__":
    raise SystemExit(main())
