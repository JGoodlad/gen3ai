"""The REAL observation rows `--compile-trainer`'s compile gates (the region gate, the canary) run on (gen3_compile_parity_real_obs_v1).

WHY REAL ROWS, and why this is a committed file rather than zeros. Until 2026-09-28 the gate probed
the compiled learner with an ALL-ZERO observation. Zero obs exercise none of the masking, none of the
top-K seat selection and none of the edge families (no opponent, no moves, no history), and on them
the compiled extractor agreed with eager to 4.8e-7 — while on REAL rows the same compiled graph was
off by up to 7.65 on pi_features, 70.9% argmax agreement, gradient cosine 0.778 (CUDA Inductor
codegen defect on torch 2.5.1, worked around there by the since-deleted gen3_inductor_trunk_split_v1). A probe
that cannot see the failure is not a gate. So the gate runs on rows a real battle produced.

WHERE THE ROWS COME FROM. Reproducible battles on the Rust env core — training's own row path
(`utils.rust_env.fixture_battles`: fixed teams, a fixed battle seed per game, a seeded uniform policy for
p1 and an episode-seeded Rust bot for p2, one thread — the same battles every run), so the fixture can be
REGENERATED at any time, on any box, for any observation layout — including a layout no training run
has produced traces for yet. Rows are taken evenly across each battle so early, mid and late-game
states are all present. (Until P6 of the poke-env retirement, 2026-10-08, the battles were poke-env
bridge battles through the Python encoder — `obs_roundtrip_fuzz_test.record_fixture_battle`; the
committed rows were written that way and stay valid: the gate needs real rows of the right width, and
the Python rows were byte-equal to the core's by slice O.)

WHEN IT GOES STALE. The observation width is recorded in the file; `load_parity_rows` REFUSES a
width that does not match the extractor (a stale fixture must never silently degrade the gate to a
subset or to zeros), and `compile_parity_fixture_test.py` fails the routine gate the moment the
layout changes. Regenerate with:

    python -m agents.model.compile_parity_fixture --write
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

# Ships BESIDE this module (a module locating its own data file, not repo-root discovery).
FIXTURE_PATH = Path(__file__).with_name("compile_parity_obs.npz")

# How many rows the fixture holds (a larger batch tiles them, `compile_trainer.fixture_index`).
N_ROWS = 64
# The reproducible battles the rows come from: this many seeded Rust-core games (`fixture_battles.play_rows`).
N_BATTLES = 8
FIXTURE_SEED = 20261008


class ParityFixtureError(RuntimeError):
    """The committed real-obs fixture is missing or does not match the extractor's layout."""


def load_parity_rows(obs_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(obs [N, obs_dim] float32, action_mask [N, A] bool)`` — or raise `ParityFixtureError`.

    Never falls back: a missing or stale fixture is a REFUSAL, because the fallback (zeros) is
    exactly the probe that hid a 70%-argmax-agreement miscompile.
    """
    if not FIXTURE_PATH.exists():
        raise ParityFixtureError(
            f"the compile parity fixture {FIXTURE_PATH} is missing. Regenerate it with "
            "`python -m agents.model.compile_parity_fixture --write` (or drop --compile-trainer).")
    with np.load(FIXTURE_PATH) as z:
        obs = np.asarray(z["obs"], dtype=np.float32)
        mask = np.asarray(z["action_mask"], dtype=bool)
    if obs.ndim != 2 or obs.shape[1] != int(obs_dim):
        raise ParityFixtureError(
            f"the compile parity fixture is STALE: its rows are {obs.shape[1] if obs.ndim == 2 else obs.shape} "
            f"wide but this extractor reads {obs_dim}. The observation layout changed since it was "
            "written. Regenerate it with `python -m agents.model.compile_parity_fixture --write` "
            "(or drop --compile-trainer) — the gate will not validate on zeros or on a subset.")
    if mask.shape[0] != obs.shape[0] or not mask.any(axis=1).all():
        raise ParityFixtureError(
            "the compile parity fixture's action masks are malformed (row count mismatch, or a row "
            "with no legal action). Regenerate it.")
    return obs, mask


def build_rows(n_rows: int = N_ROWS, n_battles: int = N_BATTLES, seed: int = FIXTURE_SEED
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Play the reproducible Rust-core battles and take ``n_rows`` decision rows spread across them."""
    from utils.rust_env.fixture_battles import spread_rows

    obs, mask = spread_rows(n_rows, n_battles=n_battles, seed=seed)
    if len(obs) < n_rows:
        raise ParityFixtureError(f"only {len(obs)} decision rows from {n_battles} battles; "
                                 f"need {n_rows} — add battles")
    return obs, mask


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help=f"regenerate {FIXTURE_PATH.name} (otherwise: report what is committed)")
    args = ap.parse_args(argv)
    if not args.write:
        if not FIXTURE_PATH.exists():
            print(f"{FIXTURE_PATH}: MISSING")
            return 1
        with np.load(FIXTURE_PATH) as z:
            print(f"{FIXTURE_PATH}: obs {z['obs'].shape}, action_mask {z['action_mask'].shape}")
        return 0
    obs, mask = build_rows()
    np.savez_compressed(FIXTURE_PATH, obs=obs, action_mask=mask,
                        source=np.asarray(f"rust_core:fixture_battles seed={FIXTURE_SEED} n_battles={N_BATTLES}"))
    print(f"wrote {FIXTURE_PATH}: obs {obs.shape}, action_mask {mask.shape}, "
          f"legal/row {mask.sum(1).mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
