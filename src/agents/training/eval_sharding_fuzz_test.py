"""Bridge-backed end-to-end check of battle-level work-stealing — REAL battles, REAL worker.

Runs the actual ``eval_worker._run`` loop (the same code training uses) against the in-process
BattleStream bridge — no server — on a real frozen checkpoint, with an opponent split into several
shard units. Then aggregates via ``ShardedEvalPool.collect`` and asserts the pooled result is
well-formed and exact:

  - every planned unit produced a shard (full coverage, nothing lost);
  - Σ n_finished across an opponent's shards == its nominal game count (Σshards == n_games);
  - win_rate ∈ [0,1] and == Σwon / Σfinished;
  - a second worker over the same claim dir finds the pool already drained (claim-exactly-once
    across processes), so it adds nothing.

Run directly (no pytest collection — no ``test_*`` functions), like the other fuzz suites::

    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/eval_sharding_fuzz_test.py [n_games] [shard_games]

Skips with a clear message if no checkpoint is available (it needs real weights to play).
"""
import os
import sys
import glob
import json
import tempfile


def _find_checkpoint() -> str | None:
    roots = ["/home/goodlad/dev/gen3ai/models", os.path.join(os.getcwd(), "models")]
    pats = ["*/best_model/best_model.zip", "*/checkpoint_*_steps.zip", "*/final_model.zip"]
    for root in roots:
        for pat in pats:
            hits = sorted(glob.glob(os.path.join(root, pat)), key=os.path.getmtime, reverse=True)
            if hits:
                return hits[0]
    return None


def main(n_games: int = 8, shard_games: int = 3) -> int:
    ckpt = _find_checkpoint()
    if not ckpt:
        print("SKIP eval_sharding_fuzz: no checkpoint found under models/ (needs real weights).")
        return 0
    print(f"checkpoint: {ckpt}")

    from agents.training.eval_sharding import EvalItem, ShardedEvalPool, BOT
    import main.eval_worker as ew

    # Two fast scripted bots, each split into ceil(n_games/shard_games) shard units.
    items = [EvalItem("heuristic2", BOT, n_games), EvalItem("aggressive_v2", BOT, n_games)]
    pool = ShardedEvalPool(items, shard_games=shard_games, step=0)
    n_shards_expected = pool.n_units
    print(f"{len(items)} opponents x {n_games} games, shard_games={shard_games} "
          f"→ {n_shards_expected} units")

    with tempfile.TemporaryDirectory() as run_dir:
        claim_dir = os.path.join(run_dir, "claims")
        os.makedirs(claim_dir, exist_ok=True)
        pool.write_plan(run_dir)
        cfg = {
            "snapshot": ckpt, "port": None, "use_showdown_bridge": True,
            "model_dir": None, "step": 0, "claim_dir": claim_dir, "result_dir": run_dir,
            "concurrency": 1, "device": "cpu", "cycle_tag": "fz", "worker_id": 0, "gamma": 0.99,
        }
        # Worker 0 drains the whole pool (the real loop: claim → play via bridge → publish).
        ew._run(cfg)
        # Worker 1 over the same claim dir: every unit already claimed → it plays nothing.
        cfg1 = dict(cfg, worker_id=1, cycle_tag="fz1")
        ew._run(cfg1)

        # Count shard files actually on disk == planned units (claim-exactly-once, full coverage).
        shard_files = glob.glob(os.path.join(run_dir, "shard__*.json"))
        assert len(shard_files) == n_shards_expected, \
            f"expected {n_shards_expected} shard files, got {len(shard_files)}"

        merged, missing = ShardedEvalPool.from_plan(run_dir).collect(run_dir)
        assert not missing, f"unexpected missing opponents: {missing}"
        for it in items:
            key = it.key
            wr = merged["win_rates"][key]
            n_won, n_fin = merged["counts"][key]
            cov = merged["coverage"][key]
            print(f"  {key}: win_rate={wr:.3f} ({n_won}/{n_fin})  coverage={cov:.2f}")
            assert 0.0 <= wr <= 1.0, wr
            assert n_fin == it.n_games, f"{key}: Σshards {n_fin} != n_games {it.n_games}"
            assert wr == (n_won / n_fin), "win_rate must equal Σwon/Σfinished exactly"
            assert cov == 1.0, f"{key}: partial coverage {cov}"

    print(f"\nOK eval_sharding_fuzz: {n_shards_expected} units played + pooled exactly, "
          f"full coverage, claim-exactly-once across 2 workers.")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sg = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    sys.exit(main(n, sg))
