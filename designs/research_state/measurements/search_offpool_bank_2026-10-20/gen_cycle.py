"""THE BATTLES (``PREREGISTRATION.md`` §3): one generated eval cycle of X26's plateau checkpoint P* against the plateau
cycle's own roster (the nine scripted bots and the five sentinels the live 280M cycle met), with EVERY OPPONENT piloting
a heldout72_hpfix team. The trainee (P*, the traced side) draws its team exactly as the live cycle did (the pool, with
its 10 % sample-team bias).

It is ``main.ops.eval_trace_gen`` (the production Rust eval executor, offline, CPU) with three patches, nothing else:
  1. the sentinels are the plateau cycle's five, by step, from the read-only checkpoint copies (sha256-checked), in the
     live order (``sentinel_0`` = 268000171 ... ``sentinel_4`` = 134000128), not the tool's evenly spaced pick;
  2. the eval builders' opponent pool is the 72 heldout72_hpfix teams (the core's team table holds them);
  3. EVERY opponent (bot or sentinel) draws from that pool. Live, a greedy sentinel draws the trainee's own
     distribution; that is the one regime change, and it is the point of the bank.
Every game is a pure function of (``--seed``, opponent, game index) (``gen3_eval_game_seed_v1``). Capture is ALL; the
live quota is applied when the bank is drawn (``make_bank.py``), by game index.

    python -I gen_cycle.py --code <8fe4fb89 checkout> --cycle cyc1 [--games 100] [--timeout-min 40]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offpool_common as E  # noqa: E402


def ckpt_copies() -> dict:
    """Read-only copies of P* and the five sentinels in ``<archive>/ckpt`` (copied once, sha256-checked every time)."""
    src = E.MAIN / "models" / E.RUN / "checkpoints"
    cfg = E.d("ckpt") / "model_config.json"           # the readers find a checkpoint's config beside it
    if not cfg.exists():
        shutil.copy2(src.parent / "model_config.json", cfg)
    if E.sha256_file(cfg) != E.MODEL_CONFIG_SHA:
        raise SystemExit(f"{cfg}: sha256 differs from the registered model_config")
    cfg.chmod(0o444)
    out = {}
    for step, sha in [(E.PSTAR_STEP, E.PSTAR_SHA)] + list(E.SENTINELS):
        dst = E.d("ckpt") / f"checkpoint_{step}_steps.zip"
        if not dst.exists():
            shutil.copy2(src / dst.name, dst)
        got = E.sha256_file(dst)
        if got != sha:
            raise SystemExit(f"{dst}: sha256 {got} differs from the registered {sha}")
        dst.chmod(0o444)
        out[step] = dst
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--cycle", required=True, choices=sorted({c for c, _p in E.SEEDS}) + ["smoke", "smoke2"])
    ap.add_argument("--part", required=True, choices=sorted(E.PARTS) + ["sentinels"],
                    help="one part of the cycle (offpool_common.PARTS); `sentinels` = all five (the smoke only)")
    ap.add_argument("--games", type=int, default=0, help="the smoke only: games per opponent")
    ap.add_argument("--timeout-min", type=float, default=40.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--n-envs", type=int, default=128)
    a = ap.parse_args()
    smoke = a.cycle.startswith("smoke")
    if not smoke and (a.games or a.part not in E.PARTS):
        raise SystemExit("a bank cycle plays the registered parts and game counts (offpool_common.PARTS / GAMES)")
    games = a.games if smoke else E.GAMES[a.part]
    rng = (0, 5) if a.part == "sentinels" else E.PARTS[a.part]
    sents = [] if rng is None else E.SENTINELS[rng[0]:rng[1]]
    E.use_code(Path(a.code))
    ck = ckpt_copies()
    texts = E.heldout_texts()

    from agents.training import eval_schedule as ES
    from agents.training.rust_eval import build as RB
    from agents.training.rust_eval import executor as EX
    from main.ops import eval_trace_gen as G
    from utils.teambuilder import Gen3Teambuilder

    heldout = Gen3Teambuilder([texts[t] for t in sorted(texts)])
    if len(heldout.packed_teams) != 72:
        raise SystemExit(f"the builder kept {len(heldout.packed_teams)} of 72 heldout teams")
    orig_builders = RB.eval_builders

    def eval_builders(trainee_team_str, fixed_entries):
        tb, _flat, fixed = orig_builders(trainee_team_str, fixed_entries)
        if fixed:
            raise SystemExit("no fixed opponents in this cycle")
        return tb, heldout, fixed

    def opponent_builder(self, item, sentinel_greedy):
        return self.opp_builder

    def pick_sentinels(run_dir, step, k, *, include_current=False):
        if k == 0:
            return [], ["the bots' part: no sentinel"]
        return ([{"label": f"sentinel_{i}", "path": str(ck[s]), "step": s} for i, (s, _sha) in enumerate(sents)],
                [f"OFF-POOL BANK: the plateau cycle's sentinels {[s for s, _ in sents]} (search_offpool_bank_2026-10-20)"])

    RB.eval_builders = eval_builders
    EX.RustEvalCore.opponent_builder = opponent_builder
    G.pick_sentinels = pick_sentinels
    if a.part != "bots":
        ES.eval_opponent_names = lambda: []              # the plan's bot items only; the core still declares them

    seed = E.SMOKE_SEED if smoke else E.SEEDS[(a.cycle, a.part)]
    out = E.ARCH / ("smoke" if smoke else "traces") / f"{a.cycle}_{a.part}"
    argv = [f"{E.RUN}@{E.PSTAR_STEP}", "--out", str(out), "--games", str(games),
            "--sentinels", str(len(sents)),
            "--seed", str(seed), "--n-envs", str(a.n_envs), "--threads", str(a.threads), "--torch-threads",
            str(a.threads), "--nice", "19", "--timeout-min", str(a.timeout_min), "--eval-sentinel-greedy"]
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists: a cycle is never regenerated in place")
    rc = G.main(argv)
    (out / "OFFPOOL_PROVENANCE.txt").write_text(
        f"generated by search_offpool_bank_2026-10-20/gen_cycle.py at code {E.CODE_SHA}, cycle {a.cycle}, part "
        f"{a.part}, seed {seed}\n"
        "opponents (all 14) piloted heldout72_hpfix teams; the trainee drew the pool (10 % sample bias), as live.\n"
        "The manifest's population string says 'pool sentinels' and 'drawing the trainee's own team distribution': "
        "BOTH are overridden here (gen_cycle.py patches 1-3).\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
