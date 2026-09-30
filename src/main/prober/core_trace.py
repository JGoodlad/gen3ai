"""EXPAND a Rust-eval CORE TRACE (``gen3_core_trace_v1``) into the full legacy summary.

The Rust eval path (``agents.training.rust_eval.traces.write_core_trace``) writes a trace with no
poke-env battle behind it, so its ``*_summary.json`` carries ``meta`` only — no ``teams``, no
``invocations``. Every prober view reads those two blocks. This module rebuilds them ON READ, by
driving the SAME Python machinery the live eval recorder runs, over the SAME per-side protocol:

1. load the ``<prefix>_reconstruction.json`` and replay it through the rust driver
   (``utils.bridge.reconstruction.replay_battle(rec, impl="rust")``);
2. **CROSS-CHECK** the replayed trainee-side protocol against the stored ``<prefix>.<side>.jsonl.gz``
   record's ``text`` lines (``|t:|`` lines dropped on both sides — wall-clock, state-invisible). The
   RECORD is the authority: any difference is refused with :class:`CoreTraceMismatch` naming the
   first differing line, never "repaired";
3. feed the replayed trainee chunks through ``obs_materializer``'s transport-less replay player —
   the mirror of ``EvalRLPlayer.choose_move`` (stall check first, then the decision cadence, an
   all-zero mask is no decision) — and at each decision drive a real
   :class:`~agents.training.battle_recorder.BattleRecorder` exactly as the live player does:
   ``record(battle, actions[i], softmax(masked logits[i]), mask, {"obs", "logits", "value"})``,
   then ``finalize`` + ``to_summary``;
4. **REFUSE** when the replay's decisions do not line up with the ``states.npz`` rows: the count
   must equal ``meta.invocations`` and each row's legal mask must equal ``action_mask[i]``
   (:class:`CoreTraceMismatch`). The recomputed ``result`` must equal the stored one;
5. replace the recomputed ``meta`` with the STORED one (it keeps ``trace_source``).

Stall forfeits follow the live rule: the trainee forfeits BEFORE the forward at its first decision
with ``turn >= threshold`` (``StallConfig().threshold``, the value production runs the core's
``turn_limit`` at), so that request is not a row — in the core (``rust_env::episode::
stall_forfeit_due``) and in this mirror alike. A trace whose core ran at a different
``turn_limit`` states it in ``trace_source.turn_limit``; absent, the production threshold is used
and a disagreement is refused by the count check. An opponent forfeit simply ends the protocol —
there is no trainee request to mirror.

Cost: one rust replay subprocess + one poke-env feed per battle (tens to hundreds of ms). The result
is cached IN MEMORY per (path, mtimes); nothing is ever written into the run dir.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.training.rust_eval.traces import is_core_trace

__all__ = ["CoreTraceError", "CoreTraceMismatch", "expand", "is_core_trace", "load_summary",
           "protocol_log", "record_text_lines", "protocol_lines"]

_CACHE_CAP = 256
_cache: "OrderedDict[tuple, dict]" = OrderedDict()


class CoreTraceError(RuntimeError):
    """A core trace that cannot be expanded (a missing sibling file, an unreadable record)."""


class CoreTraceMismatch(CoreTraceError):
    """The replay DISAGREES with the stored record / states — refused, never repaired."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_summary(summary_path: str) -> dict:
    """``json.load`` of a summary, EXPANDED when it is a core trace — the one reader to call."""
    with open(summary_path) as f:
        summary = json.load(f)
    return expand(summary_path, summary) if is_core_trace(summary) else summary


def expand(summary_path: str, summary: dict, *, impl: str = "rust",
           run_dir: Optional[str] = None) -> dict:
    """The full legacy summary (``meta`` + ``teams`` + ``invocations``) for a core trace.

    ``summary`` is the stored meta-only dict. A non-core summary is returned unchanged.
    ``run_dir`` (default: inferred from the path, as discovery does) supplies
    ``model_config.json``: the recorder's per-turn ``outcome.reward`` is scored with the run's
    ``RewardConfig``, exactly as the live eval worker builds it (absent ⇒ the default config).
    Cached in memory by path + the input files' mtimes; each call returns a deep COPY (a caller
    mutating its summary must not reach another session's)."""
    if not is_core_trace(summary):
        return summary
    return copy.deepcopy(_entry(summary_path, summary, impl=impl, run_dir=run_dir)["summary"])


def protocol_log(summary_path: str, summary: dict, *, impl: str = "rust",
                 run_dir: Optional[str] = None) -> Tuple[str, ...]:
    """The trainee's protocol log EXACTLY as ``write_battle_record``'s ``*_replay.html`` would carry
    it (``battle._build_replay_events()`` of the expansion's battle, ``|``-lines only) — the stand-in
    for the ``_replay.html`` a core trace does not ship. Empty for a non-core summary."""
    if not is_core_trace(summary):
        return ()
    return _entry(summary_path, summary, impl=impl, run_dir=run_dir)["protocol"]


def _entry(summary_path: str, summary: dict, *, impl: str, run_dir: Optional[str]) -> dict:
    files = _sibling_files(summary_path, summary)
    if run_dir is None:
        from main.prober.discovery import _infer_run_dir

        run_dir = _infer_run_dir(summary_path)
    cfg_path = os.path.join(run_dir, "model_config.json") if run_dir else None
    inputs = list(files.values()) + ([cfg_path] if cfg_path and os.path.exists(cfg_path) else [])
    key = (os.path.abspath(summary_path), impl, tuple((p, os.stat(p).st_mtime_ns) for p in inputs))
    hit = _cache.get(key)
    if hit is None:
        hit = _expand_uncached(summary, files, impl=impl, reward_cfg=_read_json(cfg_path))
        _cache[key] = hit
        while len(_cache) > _CACHE_CAP:
            _cache.popitem(last=False)
    else:
        _cache.move_to_end(key)
    return hit


def _read_json(path: Optional[str]) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record_text_lines(path: str) -> List[str]:
    """The ``text`` of every line of a ``gen3_core_event_v1`` record (header skipped), in order."""
    out: List[str] = []
    with gzip.open(path, "rt") as f:
        for n, raw in enumerate(f):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if n == 0 and "text" not in row:
                continue            # the header line
            out.append(row["text"])
    return out


def protocol_lines(lines: Sequence[str]) -> List[str]:
    """The state-bearing lines: ``|t:|`` (wall-clock, in ``MESSAGES_TO_IGNORE``) and blanks dropped."""
    return [ln for ln in lines if ln and not ln.startswith("|t:|")]


# ---------------------------------------------------------------------------
# The expansion
# ---------------------------------------------------------------------------

def _sibling_files(summary_path: str, summary: dict) -> Dict[str, str]:
    src = summary["meta"]["trace_source"]
    side = src.get("trainee_side", "p1")
    d = os.path.dirname(os.path.abspath(summary_path))
    files = {"summary": os.path.abspath(summary_path),
             "record": os.path.join(d, src["records"][side]),
             "reconstruction": os.path.join(d, src["reconstruction"]),
             "states": os.path.join(d, src["states"])}
    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        raise CoreTraceError(f"core trace {summary_path}: missing {', '.join(missing)} "
                             f"({', '.join(files[k] for k in missing)})")
    return files


def _masked_probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``softmax(logits + (mask - 1) * 1e9)`` — the live player's ``probs``, by the SAME torch ops
    (``RLPlayer._predict_best_action``), so a ``%.1f%%`` label cannot round differently."""
    import torch

    lg = torch.as_tensor(np.asarray(logits, dtype=np.float32)[None])
    mk = torch.as_tensor(np.asarray(mask)[None])
    return torch.softmax(lg + (mk - 1.0) * 1e9, dim=1)[0].numpy()


def _check_protocol(replayed: Sequence[str], stored: Sequence[str], record_path: str) -> None:
    a, b = protocol_lines(replayed), protocol_lines(stored)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            raise CoreTraceMismatch(
                f"core trace record {record_path}: the rust replay DISAGREES with the stored record "
                f"at protocol line {i}: record={y!r} replay={x!r}")
    if len(a) != len(b):
        i = min(len(a), len(b))
        extra = (f"record has {len(b) - i} more, first {b[i]!r}" if len(b) > len(a)
                 else f"replay has {len(a) - i} more, first {a[i]!r}")
        raise CoreTraceMismatch(f"core trace record {record_path}: protocol length differs at line "
                                f"{i} ({extra})")


def _expand_uncached(summary: dict, files: Dict[str, str], *, impl: str, reward_cfg: dict) -> dict:
    import functools

    from agents.training.reward_config import RewardConfig
    from agents.training.reward_manager import Gen3RewardManager
    from agents.training.stall import StallConfig
    from utils.bridge.reconstruction import ReconstructionRecord, replay_battle

    meta = summary["meta"]
    src = meta["trace_source"]
    side = src.get("trainee_side", "p1")
    try:
        rec = ReconstructionRecord.load(files["reconstruction"])
    except (OSError, ValueError, KeyError) as e:
        raise CoreTraceError(f"core trace {files['reconstruction']}: unreadable ({e})") from e
    with np.load(files["states"]) as z:
        npz = {k: z[k] for k in z.files}

    replay = replay_battle(rec, impl=impl)
    chunks = replay.p1_chunks if side == "p1" else replay.p2_chunks
    _check_protocol([ln for c in chunks for ln in c.split("\n")],
                    record_text_lines(files["record"]), files["record"])

    threshold = int(src.get("turn_limit") or StallConfig().threshold)
    recorder, battle, rows = _drive_recorder(
        chunks, username=rec.username(side), packed_team=rec.packed_team(side), side=side,
        battle_tag=str(meta.get("battle_id") or rec.battle_tag or "core"), npz=npz,
        stall_config=StallConfig(threshold=threshold),
        reward_fn_factory=functools.partial(Gen3RewardManager,
                                            config=RewardConfig.from_dict(reward_cfg)))

    n_expected = int(meta.get("invocations", len(npz["actions"])))
    if len(rows) != n_expected or len(rows) != len(npz["actions"]):
        raise CoreTraceMismatch(
            f"core trace {files['summary']}: the replay reached {len(rows)} trainee decisions but "
            f"meta.invocations={n_expected} and states.npz has {len(npz['actions'])} rows "
            f"(stall threshold {threshold}; last replayed turns {[t for t, _ in rows[-3:]]})")
    for i, (turn, mask) in enumerate(rows):
        if not np.array_equal(mask.astype(bool), npz["action_mask"][i].astype(bool)):
            raise CoreTraceMismatch(
                f"core trace {files['summary']}: decision {i} (turn {turn}) legal mask differs — "
                f"replay {mask.astype(int).tolist()} vs states.npz "
                f"{npz['action_mask'][i].astype(int).tolist()}")

    recorder.finalize(battle)
    out = recorder.to_summary(battle, int(meta.get("step", 0)))
    if out["meta"]["result"] != meta.get("result"):
        raise CoreTraceMismatch(
            f"core trace {files['summary']}: the replayed battle reads {out['meta']['result']} "
            f"(turn {out['meta']['turns']}) but the stored meta says {meta.get('result')} "
            f"(turn {meta.get('turns')})")
    out["meta"] = copy.deepcopy(meta)
    # The raw seam `write_battle_record` itself uses (`save_replay` renders exactly these events).
    protocol = tuple(ln for ln in battle._build_replay_events() if ln.startswith("|"))
    return {"summary": out, "protocol": protocol}


def _drive_recorder(chunks: Sequence[str], *, username: str, packed_team: str, side: str,
                    battle_tag: str, npz: Dict[str, np.ndarray], stall_config,
                    reward_fn_factory) -> Tuple[Any, Any, List[Tuple[int, np.ndarray]]]:
    """Feed ``chunks`` through a recording replay player; ``(recorder, battle, [(turn, mask)])``."""
    import asyncio

    from poke_env.concurrency import POKE_LOOP

    from agents.training import obs_materializer as om
    from agents.training.battle_recorder import BattleRecorder

    tag = om._next_tag(battle_tag, "gen3ou")      # a room tag with the format in segment 1
    player, client = om._build_replay_player(
        username=username, packed_team=packed_team, side=side, actions=npz["actions"],
        battle_format="gen3ou", mappings=None, stall_config=stall_config, map_actions_at=None,
        stop_after_decision=None, encode_only_at=set(), player_cls=_recording_player_cls())
    recorder = BattleRecorder(battle_tag, reward_fn_factory)
    player._core_init(recorder, npz)
    om._refuse_poke_loop("core_trace.expand")
    asyncio.run_coroutine_threadsafe(om._feed(client, player, chunks, tag, first=True),
                                     POKE_LOOP).result()
    battles = list(player.battles.values())
    if len(battles) != 1:
        raise CoreTraceError(f"core trace {battle_tag}: the replay produced {len(battles)} battles")
    return recorder, battles[0], player._core_rows


_PLAYER_CLS: Optional[type] = None


def _recording_player_cls() -> type:
    """``_ReplayObsPlayer`` + a live-eval-shaped ``BattleRecorder.record`` at each decision.

    Built lazily so importing this module does not import poke-env."""
    global _PLAYER_CLS
    if _PLAYER_CLS is not None:
        return _PLAYER_CLS
    from agents.training.obs_materializer import _ReplayObsPlayer

    class _CoreRecordingPlayer(_ReplayObsPlayer):
        """Mirrors ``EvalRLPlayer.choose_move``: stall check, then the decision, then ``record``.

        Never stops early (``done`` is always False) so a replay with MORE decisions than the
        states rows is counted, not truncated — the count check then refuses it."""

        def _core_init(self, recorder, npz) -> None:
            self._core_recorder = recorder
            self._core_npz = npz
            self._core_rows: List[Tuple[int, np.ndarray]] = []

        @property
        def done(self) -> bool:
            return False

        def choose_move(self, battle):
            forfeit = self._handle_stall(battle, "CORE_TRACE_STALL")
            if forfeit:
                return forfeit
            i = len(self._core_rows)
            step = self._encode_or_track(battle, i)
            if step is None:
                return self.choose_default_move()
            _obs, mask = step
            mask = np.asarray(mask)
            self._core_rows.append((battle.strict_view().turn, mask))
            npz = self._core_npz
            if i < len(npz["actions"]):
                a = int(npz["actions"][i])
                logits = np.asarray(npz["logits"][i], dtype=np.float32)
                state = {"obs": np.asarray(npz["obs"][i], dtype=np.float32), "logits": logits,
                         "value": float(npz["values"][i])}
                self._core_recorder.record(battle, a, _masked_probs(logits, mask), mask, state=state)
                self._get_tracker(battle).advance(a)
            return self.choose_default_move()

    _PLAYER_CLS = _CoreRecordingPlayer
    return _PLAYER_CLS
