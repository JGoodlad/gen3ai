"""THE SNAPSHOT LADDER'S GAMES, ON THE RUST EVAL ENGINE (poke-env retirement P2, ``gen3_ladder_rust_v1``, 2026-10-06).

Until 2026-10-06 every ladder edge (``snapshot_ladder._play_pair``) was played by two poke-env ``RLPlayer`` s over the
in-process bridge, each encoding its observation with the PYTHON encoder, while every trainee since the M5 cutover
trains on RUST rows: the ELO headline was measured on a stack the trainee never saw (survey finding A-F2,
``designs/research_state/measurements/pokeenv_and_hotpath_survey_2026-10-06/``). This module plays the same edge on the
head-to-head meter's engine (``main.h2h.play.H2HEngine``: one T2 service, one Rust eval core, the strict loader, every
slot load byte-checked), so the ladder's games are played exactly as the in-loop eval and ``main.h2h`` play theirs.

WHAT A LADDER EDGE IS NOW (protocol :data:`PROTOCOL`):

* both sides GREEDY (argmax of the served log-probs), as before;
* both sides draw from the default biased pool (10 % sample-team bias), as before — via the eval core's own trainee
  builder; a run that pinned its trainee team is NOT refused (the ladder's yardstick is the default pool for every
  run, as it always was), and the pool is the WORKING DIRECTORY's (a pinned run's ``data/`` is main's);
* ``n_games`` = ``n_games // 2`` MIRRORED team pairs (``gen3_mirrored_pairs_v1``: one team pairing, one battle seed,
  played twice with the teams handed over), so team-draw luck cancels inside a pair;
* SEAT-BALANCED: the eval core keeps its player on seat p1, so ``ceil(P/2)`` pairs are played with ``a`` as the player
  and ``floor(P/2)`` with ``b``, on the SAME schedule (the key is order-independent, so the two directions replay the
  same team pairings and battle seeds with the seats exchanged). The Python ladder put ``a`` — the NEW node, on a
  promotion — on p1 for every game, so any seat effect was credited to the newest node;
* a game that reaches the declared turn limit is a DRAW (as in training). Draws are recorded (``draws``) and
  EXCLUDED from the BT edge: ``games`` = the decisive games, ``wins_a`` = ``a`` 's wins among them (the Python ladder
  folded a draw into ``b`` 's column);
* reproducible: every game is a pure function of (the two checkpoints' content hashes, the edge's batch index), so a
  re-measurement of a pair must take a NEW batch index (``snapshot_ladder._measure_missing`` passes the count of rows
  already recorded for the pair) — an identical replay is not an independent sample.

CPU only (the detached updater never sees the GPU, ``gen3_ladder_off_gpu_v1``).
"""
from __future__ import annotations

import contextlib
import hashlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

#: The transport every NEW ladder row is played on (stamped on the row; ``snapshot_ladder.row_transport``).
TRANSPORT = "rust_eval"
#: The encoder that built both sides' observations.
ENCODER = "rust"
#: What a Rust ladder edge MEASURES (module docstring). A change to it is a new protocol.
PROTOCOL = "gen3_ladder_rust_v1"
SCHEDULE_TAG = "gen3_ladder_schedule_v1"

#: The engine's CPU shape: small, because the updater runs beside a training run.
DEFAULT_N_ENVS = 32
DEFAULT_THREADS = 2
DEFAULT_TORCH_THREADS = 2


class LadderPlayError(RuntimeError):
    """A ladder edge could not be played as declared."""


def seat_split(n_pairs: int) -> Tuple[int, int]:
    """``(pairs with a as the player, pairs with b as the player)``: ``a`` takes the odd one."""
    n = int(n_pairs)
    if n < 1:
        raise LadderPlayError(f"a ladder edge needs at least one mirrored pair, got {n}")
    return (n + 1) // 2, n // 2


def pairs_for(n_games: int) -> int:
    """Mirrored pairs for an ``n_games`` edge (``n_games // 2``; an odd count loses its last game, at least 1 pair)."""
    return max(1, int(n_games) // 2)


def schedule_key(sha_a: str, sha_b: str) -> str:
    """The edge's schedule key: the digest of the two content hashes, ORDER-INDEPENDENT (both directions of a seat-
    balanced edge replay the same team pairings), on the ladder's own namespace (never an ``h2h:`` key)."""
    lo, hi = sorted([sha_a, sha_b])
    return "ladder:" + hashlib.blake2b(f"{SCHEDULE_TAG}:{lo}:{hi}".encode(), digest_size=8).hexdigest()


def combine(score_ab: Any, score_ba: Optional[Any]) -> Dict[str, Any]:
    """Fold the two directions' scores (each from ITS player's side) into ``a`` 's edge. ``score_ba`` is ``None``
    when the edge is a single pair (no b-as-player half)."""
    w1, l1, d1 = int(score_ab.w), int(score_ab.l), int(score_ab.d)
    w2, l2, d2 = (int(score_ba.w), int(score_ba.l), int(score_ba.d)) if score_ba is not None else (0, 0, 0)
    wins_a, losses_a, draws = w1 + l2, l1 + w2, d1 + d2
    return {"wins_a": wins_a, "losses_a": losses_a, "draws": draws, "games": wins_a + losses_a,
            "games_played": wins_a + losses_a + draws,
            # each direction from a's side: [a wins, a losses, draws]
            "by_seat": {"a_p1": [w1, l1, d1], "b_p1": [l2, w2, d2]},
            "pair_counts": {"a_p1": list(score_ab.pair_counts),
                            "b_p1": list(score_ba.pair_counts) if score_ba is not None else None}}


class LadderEngine:
    """ONE ``H2HEngine`` for a whole sweep of one run's frozen pairs (one architecture: a run's snapshots share it);
    each edge only swaps the cell's checkpoints (``H2HEngine.set_cell``, every load byte-checked)."""

    def __init__(self, run_dir: str, first: Tuple[int, int], *, n_envs: int = DEFAULT_N_ENVS,
                 threads: int = DEFAULT_THREADS, torch_threads: int = DEFAULT_TORCH_THREADS,
                 front: Optional[str] = None, profile: Optional[str] = None,
                 emit: Callable[[str], None] = lambda m: print(m, flush=True)):
        from main.h2h import play as PL

        self.run_dir, self.emit = run_dir, emit
        self._refs: Dict[int, Any] = {}
        # `front` / `profile` default to the head-to-head's (the process front end, the release build); a test passes
        # the in-process front and the emission self-check build
        extra = {k: v for k, v in (("front", front), ("profile", profile)) if v is not None}
        self.compute = PL.Compute(device="cpu", backend="eager", n_envs=int(n_envs), threads=int(threads),
                                  torch_threads=int(torch_threads), **extra)
        a, b = (self.ref(s) for s in first)
        self.eng = PL.H2HEngine(a, b, self.compute, emit, team_source_check=False, team_pool="cwd")

    def ref(self, step: int) -> Any:
        from agents.training import snapshot_ladder as sl
        from main.h2h import play as PL

        if step not in self._refs:
            self._refs[step] = PL.resolve_player(sl._snapshot_zip(self.run_dir, step))
        return self._refs[step]

    def _direction(self, player: Any, opponent: Any, pairs: int, seed: int) -> Any:
        from main.h2h import play as PL

        self.eng.set_cell(player, opponent)
        games, st = self.eng.play_batch(pairs, seed)
        score = PL.score_games(games, self.eng.team_packed, pairs)
        if st.get("executor_pair_counts") is not None and list(st["executor_pair_counts"]) != score.pair_counts:
            raise LadderPlayError(f"the executor's pentanomial {st['executor_pair_counts']} != the game log's "
                                  f"{score.pair_counts}")
        return score

    def play(self, step_a: int, step_b: int, n_games: int, batch: int = 0) -> Dict[str, Any]:
        """Play the seat-balanced mirrored edge ``step_a`` vs ``step_b`` (module docstring); returns the row fields."""
        from main.h2h import play as PL

        ra, rb = self.ref(step_a), self.ref(step_b)
        n_pairs = pairs_for(n_games)
        p_ab, p_ba = seat_split(n_pairs)
        key = schedule_key(ra.sha256, rb.sha256)
        seed = PL.cycle_seed(0, key, int(batch))
        s_ab = self._direction(ra, rb, p_ab, seed)
        s_ba = self._direction(rb, ra, p_ba, seed) if p_ba else None
        out = combine(s_ab, s_ba)
        out.update({"transport": TRANSPORT, "encoder": ENCODER, "protocol": PROTOCOL, "mirrored": True,
                    "n_pairs": n_pairs, "seat_split": [p_ab, p_ba], "batch": int(batch), "schedule_key": key,
                    "cycle_seed": int(seed), "regime_id": self.eng.regime["regime_id"],
                    "core_stamp": self.eng.core_stamp, "sha_a": ra.sha256, "sha_b": rb.sha256})
        return out

    def close(self) -> None:
        self.eng.close()


@contextlib.contextmanager
def open_engine(run_dir: str, pairs: Sequence[Tuple[int, int]], **kw: Any) -> Iterator[LadderEngine]:
    """A :class:`LadderEngine` (CPU) declared from the sweep's first pair, closed on exit."""
    if not pairs:
        raise LadderPlayError("nothing to play")
    eng = LadderEngine(run_dir, tuple(pairs[0]), **kw)          # type: ignore[arg-type]
    try:
        yield eng
    finally:
        eng.close()


def engine_kwargs(n_envs: Optional[int] = None, threads: Optional[int] = None,
                  torch_threads: Optional[int] = None) -> Dict[str, Any]:
    """The ``open_engine`` keywords a CLI passes (``None`` = the default)."""
    kw: Dict[str, Any] = {}
    for k, v in (("n_envs", n_envs), ("threads", threads), ("torch_threads", torch_threads)):
        if v is not None:
            kw[k] = int(v)
    return kw


__all__: List[str] = ["TRANSPORT", "ENCODER", "PROTOCOL", "LadderEngine", "LadderPlayError", "open_engine",
                      "seat_split", "pairs_for", "schedule_key", "combine", "engine_kwargs"]
