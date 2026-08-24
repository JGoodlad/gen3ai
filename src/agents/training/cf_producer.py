"""cf_producer — the LABEL PRODUCER DRIVER of the counterfactual value-grounding loop.

    python -m agents.training.cf_producer <run_dir> [--rollouts 8] [--top-n 3] ...

**The piece that closes the loop.** `cf_records.py` rings a training episode's reconstruction
record; `cf_label_buffer.py` consumes label rows off disk and folds them into the win-prob head's
auxiliary loss; `cf_audit.py` manufactures labels from EVAL traces for the bias map. Nothing ran
the loop from `<run>/cf_records/` to `<run>/cf_labels/`. This does.

It is a **long-lived standalone process run BESIDE a live trainer** — the detached-sidecar pattern
of `snapshot_ladder` / `bot_matchup_matrix`, launched by hand or by the R1 runbook's launch line.
It is deliberately NOT auto-spawned by the trainer: producer and consumer share only a file format
(`cf_label_buffer`'s whole premise), and a producer the trainer owns would make a label-path
failure a training failure.

Each cycle:

1. **Watch** ``<run_dir>/cf_records/`` for records it has not processed (remembered in
   ``<run_dir>/cf_producer_state.json``, which is written BEFORE the work — see *Crash safety*),
   **NEWEST FIRST**, and READ this cycle's batch into memory at enumeration time — see *The
   producer/retention race* below.
2. **Refresh the snapshot** — the freshest ``checkpoints/`` zip (via ``latest.txt``, else the
   highest-stepped one on disk). Its step is stamped on every label it produces.
3. **Select decisions by a DECLARED priority** — replay each record once (which also yields the
   realized outcome and every decision's obs), forward the snapshot over them, and rank by
   ``critic_surprise`` + ``policy_entropy`` (§ *The sampler*). Label the top ``--top-n``; skip the
   rest. This is a SAMPLER, not a sweep.
4. **Roll out** ``--rollouts`` tight-MC continuations per selected decision: play the RECORDED
   action at that turn, then both sides live to termination on fresh post-divergence dice.
5. **Write** the shared v1 label schema to
   ``<run_dir>/cf_labels/labels_cf_producer_<step>_<seq>.jsonl`` — one NEW file per batch (never a
   rewrite, so the buffer's ``(name, inode)`` offset map can never be wrong about it).

⚠️ **THE ECOLOGY DECISION — read this before quoting any label this producer wrote.**
A training record carries **no opponent identity**. The tap's ``__RECON__`` frame holds the
resolved seed, both packed teams and the committed choices, and nothing that says *which policy*
sat on the other side — a self-play pool snapshot, one of the nine heuristic bots, or the trainee's
own weights. The label therefore cannot name the opponent it was measured against, and a value
claim that cannot name its population is not a value claim (the G0 rule: *never quote "the critic
is optimistic by X" without naming the population — the sign depends on it*).
So v1 makes the approximation **explicit rather than guessed**: every rollout is played by the
**CURRENT snapshot on BOTH sides, sampling stochastically at temperature 1.0** — the regime the
training actor itself plays in. That matches the ~90% self-play share of the training mixture, and
it is wrong in a KNOWN direction for the rest: on an episode whose opponent was a bot, a weaker
opponent is replaced by a stronger self-like one, so that label is biased LOW. Every row carries
``opponent: "self_current"`` — never a bot name it cannot verify — so a reader can always tell a
producer label from a `cf_audit` label, whose opponent IS identified. Closing the approximation
means threading the opponent's identity through the training-side tap; it is not a change to this
file.

The sampler (``cf_producer_priority_v1``)
-----------------------------------------
Declared, versioned, and written into the state file AND every label row, because a silent
priority change is a distribution-shift confound for every downstream readout (design
decision-of-record 3). Per candidate decision:

``critic_surprise`` = ``|P(win|s) − realized outcome|`` — highest first. This is the **conviction
region**: the states where the head was sure and the game disagreed. G0 measured that class at
+0.23 predicted-minus-MC, and measured that a single realized outcome cannot say whether the head
was wrong or the dice were (53% of that class was genuinely winning). Tight-MC labels are the only
instrument that separates them, so they are spent here first.

``policy_entropy`` = the masked action distribution's entropy / ``log(n_legal)`` — the decisions
the policy has not made up its mind about, where a ground-truth value is worth most.

``score = 1.00·critic_surprise + 0.35·policy_entropy``. Surprise dominates deliberately; entropy
is a tie-break that keeps the sample off the degenerate "one legal move" states.

A checkpoint with no win-prob head (``--win-prob-mode none``) has no surprise term at all; the
producer says so once and ranks on entropy alone rather than pretending the term was zero.

Crash safety, and what it costs
-------------------------------
A record is marked processed and the state file is fsync-replaced **before** its rollouts run. So
a crash mid-record loses that record's labels (it will never be retried) and can NEVER
double-label. That direction is chosen on purpose: the buffer dedups on the obs digest, so a
duplicate is survivable — but it is also a silent re-weighting of the declared sampler, and a
record aged out of the ring unprocessed is simply a record that was not labelled. Missing a label
is free; mis-weighting the sampler is not.

The producer/retention race (``records_vanished``)
--------------------------------------------------
The trainer owns ``cf_records/``: every env worker prunes it to the newest ``--cf-records-keep``
(512 by default). This process only reads it. So a record can be ENUMERATED and then DELETED before
it is opened, and three things follow, all of them load-bearing:

* **Records are taken NEWEST FIRST.** The ring deletes from the OLD end, so the oldest pending
  record is the one it has already promised to destroy — and the loop used to walk exactly that end.
  Measured on `ai_v9_29_rev1_0823`: 176 records lost to `FileNotFoundError` across 67 cycles, with
  "538 pending" against a ring of 512 — the excess is a guaranteed loss by arithmetic alone.
* **The batch is READ AT ENUMERATION TIME** (:meth:`CfProducer._load_batch`). The gap the race wins
  is enumerate → anchor (a full scripted replay) → claim + fsync → open. Reading immediately
  collapses it, and everything downstream works from an in-memory ``ReconstructionRecord``.
* **A vanished file is a COUNTED BENIGN SKIP** (``records_vanished`` in the state file, on the
  heartbeat, and one explanatory log line), never an exception path. As an exception it landed in
  ``skip_reasons`` as ``error:FileNotFoundError`` — indistinguishable from a corrupt record — and on
  the anchor path it reached ``anchors_errored``, where it could exit 3 over an ordinary deletion.
  The remedy is a larger ``--cf-records-keep`` on the trainer, which is a flag a restart can raise;
  nothing here changes the ring's semantics.

The anchor discipline (inherited from `cf_audit`)
-------------------------------------------------
On startup and every ``--anchor-every`` records, one record is replayed FULLY SCRIPTED through the
live bridge (``divergence_turn=None`` — the correctness oracle) and must reproduce the winner the
offline replay driver reports. On failure the producer **exits 3 and writes nothing further**: a
factory whose replay is not exact is GIGO, and every label after it would be a measurement of the
bug. Note this anchor is STRONGER than `cf_audit`'s: nothing is played by a policy, so it isolates
the record → replay → bridge → outcome chain with no sampling in it.

⚠️ **ONE declared coverage bound: a record that ends in a FORFEIT.** The 250-turn cap ends a battle
with a ``['forcelose', <side>]`` command that the per-side script drops, so both scripted players
re-derive a forfeit from `_handle_stall` and the winner becomes a race — a faithfulness limit of
live scripted replay, root-caused 2026-08-23 and the whole of the intermittent `ANCHOR REFUSED` the
R1 composition test hit (**4 refusals in 1037 battles, all four of them forfeit records; 0 of 1021
non-forfeit records**). :func:`record_is_full_replay_anchorable` excludes the class; the skip is
counted (``anchors_skipped_unanchorable``) and announced once, never retried.

⚠️ **TWO refusals reach exit 3, and they mean opposite things.** A MISMATCH (the scripted replay
disagreed) is the defect above. An ERROR (the anchor RAISED — a wedged bridge child, a transport
error, a contention ``ProgressTimeout``) never returned a verdict at all, so it is not evidence
about exactness; it still refuses, because an anchor that did not complete has certified nothing.
They are counted apart (``anchors_errored`` vs ``anchors_run - anchors_reproduced``, as `cf_audit`
has always done) and rendered apart by :func:`anchor_refusal_message`, which additionally appends
``describe_contention()`` on a timeout. The single shared message cost a whole investigation on
2026-08-23: one flaky composition-test failure read as a replay-exactness gap because the text
asserted that cause for both.

Observability
-------------
The producer is a separate process, so it has no TensorBoard. Instead it prints one **heartbeat**
line per cycle (rate, labels total, snapshot step, anchor status) and keeps
``<run_dir>/cf_producer_state.json`` human-readable. The trainer-side half of the contract is the
``cf/*`` scalars (see the R1 runbook's launch-window table); ``cf/labels_ingested_total`` going
flat is what a dead producer looks like from over there.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence

import numpy as np

from agents.action.constants import MOVE_START
from agents.training.cf_audit import LabelRow, obs_b64, obs_digest, wilson_ci
from agents.training.obs_materializer import RecordDecision, scan_record
from utils.bridge.counterfactual import replay_counterfactual as _run_one
from utils.bridge.reconstruction import RECON_SUFFIX, ReconstructionRecord, replay_battle
from utils.contention import describe_contention

#: Written into the state file and every label row. Bump it when the weights or the candidate
#: filter below change — a downstream reader comparing two runs' labels needs to know they were
#: drawn from the same distribution.
SAMPLER_VERSION = "cf_producer_priority_v1"

#: The declared priority weights (§ *The sampler*).
PRIORITY_WEIGHTS = {"critic_surprise": 1.00, "policy_entropy": 0.35}

#: Every row this producer writes names this as its opponent. See *THE ECOLOGY DECISION*.
OPPONENT_LABEL = "self_current"

STATE_FILENAME = "cf_producer_state.json"
LABELS_DIRNAME = "cf_labels"
RECORDS_DIRNAME = "cf_records"

#: Skipped and COUNTED, never silently dropped (`cf_audit` declares the same bound).
#:
#: ⚠️ Its ORIGINAL reason is GONE: "the offline replay driver cannot open turn 1" was a rust
#: `search_driver` defect, fixed 2026-08-23 (`gen3_search_turn1_open_v1`), and both impls now
#: open turn 1. The second reason it carried — "a turn-1 divergence has no prefix to be faithful
#: about" — does not survive inspection either: an EMPTY prefix is trivially faithful, which
#: makes turn 1 the easiest case rather than an excluded one.
#:
#: It is left at 2 deliberately, as a SAMPLER choice rather than a capability limit. Lowering it
#: widens the declared candidate distribution by ~3.35% of move decisions, and this module's own
#: rule is that missing a label is free while silently re-weighting the sampler is not — so that
#: is its own change, with its own before/after on the label mix, not a rider on the driver fix.
MIN_LABELABLE_TURN = 2

#: How many processed-record names the state file remembers. The ring's default cap is 512 and
#: files age out of it, so 4096 is ~8 ring turnovers of headroom at a few hundred KB of JSON.
DEFAULT_KEEP_PROCESSED = 4096

#: The TWO names a resumable checkpoint is written under. `_TrackingCheckpointCallback` writes the
#: PERIODIC `checkpoint_<step>_steps.zip`; `train.lifecycle._forced_checkpoint` (SIGUSR1 — the
#: launcher TUI's `c` key) writes `checkpoint_forced_<step:010d>_<HHMMSS>.zip` into the same
#: directory. Reading only the first is not a cosmetic gap: an unparseable name scores `(0, 0, …)`
#: in `resolve_latest_checkpoint`'s key, so ANY periodic checkpoint outranks EVERY forced one —
#: a forced save taken AFTER the last periodic one would be silently passed over in favour of an
#: older snapshot, against this module's own "the highest step wins" contract.
_CKPT_STEP_RES = (
    re.compile(r"checkpoint_(\d+)_steps\.zip$"),
    re.compile(r"checkpoint_forced_(\d+)_\d+\.zip$"),
)


# ---------------------------------------------------------------------------
# Priority scoring — pure, so the arithmetic is testable without a model
# ---------------------------------------------------------------------------

def normalized_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy of a masked action distribution, divided by ``log(n_legal)``.

    Normalizing by the support size is what makes the number comparable ACROSS decisions: a
    3-legal-action decision and a 9-legal-action one have different maximum entropies, and an
    un-normalized entropy would rank "many options, all equal" above "two options, genuinely
    50/50" purely on the option count. Returns 0.0 when there is nothing to be undecided about
    (0 or 1 legal actions).
    """
    p = np.asarray(list(probs), dtype=float)
    p = p[p > 0.0]
    if p.size <= 1:
        return 0.0
    h = float(-(p * np.log(p)).sum())
    return max(0.0, min(1.0, h / math.log(p.size)))


def critic_surprise(win_prob: Optional[float], outcome: float) -> float:
    """``|P(win|s) − realized outcome|`` — 0.0 when the checkpoint has no win-prob head.

    ``outcome`` is 1.0 for a won battle, 0.0 for a lost one and 0.5 for a tie (the turn cap), so a
    tie is maximally uninformative about conviction rather than counted as a loss."""
    if win_prob is None or not np.isfinite(win_prob):
        return 0.0
    return float(abs(float(win_prob) - float(outcome)))


def priority_score(surprise: float, entropy: float,
                   weights: "Optional[dict]" = None) -> float:
    w = weights or PRIORITY_WEIGHTS
    return (w["critic_surprise"] * float(surprise)
            + w["policy_entropy"] * float(entropy))


def is_move_round(mask) -> bool:
    """A start-of-turn MOVE round (as opposed to a mid-turn forced switch).

    The divergence a counterfactual anchors at is a move round, so a forced-switch decision — whose
    mask offers only switches — is not labelable by this path. Read off the mask rather than a
    phase string because a ring record has no invocation metadata to carry one."""
    m = np.asarray(mask)
    return bool(m[MOVE_START:].sum() > 0)


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def step_from_checkpoint_name(path: str) -> Optional[int]:
    """The step a checkpoint FILENAME declares — periodic or forced — else None."""
    base = os.path.basename(path)
    for rx in _CKPT_STEP_RES:
        m = rx.search(base)
        if m:
            return int(m.group(1))
    return None


def resolve_latest_checkpoint(run_dir: str) -> "Optional[tuple[str, Optional[int]]]":
    """The freshest ``(path, step)`` under ``run_dir``, or None when there is not one yet.

    Considers ``latest.txt`` (which holds a run-RELATIVE path) AND the checkpoint glob, and picks
    the highest STEP among them — not simply whatever ``latest.txt`` names. The two disagree
    exactly when a checkpoint landed between the zip write and the pointer update, and in that
    window the higher step is the one whose weights are on disk.

    BOTH resumable checkpoint names count (see ``_CKPT_STEP_RES``): the periodic
    ``checkpoint_<step>_steps.zip`` and the FORCED ``checkpoint_forced_<step>_<HHMMSS>.zip`` that
    SIGUSR1 writes. Globbing only the first made a forced save reachable solely through
    ``latest.txt`` and then, because its step did not parse, rank BELOW every periodic zip — so an
    operator forcing a checkpoint mid-run moved the producer BACKWARDS to an older snapshot.
    """
    cands: "list[str]" = []
    latest = os.path.join(run_dir, "latest.txt")
    try:
        rel = Path(latest).read_text().strip()
    except OSError:
        rel = ""
    if rel:
        p = rel if os.path.isabs(rel) else os.path.join(run_dir, rel)
        if os.path.exists(p):
            cands.append(p)
    for root in (os.path.join(run_dir, "checkpoints"), run_dir):
        cands += glob.glob(os.path.join(root, "checkpoint_*_steps.zip"))
        cands += glob.glob(os.path.join(root, "checkpoint_forced_*.zip"))
    if not cands:
        return None
    # Highest declared step wins; an unparseable name falls back to mtime so a hand-placed
    # checkpoint is still reachable rather than invisible.
    def _key(p: str):
        s = step_from_checkpoint_name(p)
        return (0 if s is None else 1, s or 0, os.path.getmtime(p))
    best = max(sorted(set(cands)), key=_key)
    return best, step_from_checkpoint_name(best)


# ---------------------------------------------------------------------------
# The snapshot — the ONE seam a test substitutes
# ---------------------------------------------------------------------------

class Snapshot:
    """A loaded checkpoint: it scores decisions and it builds the players that roll them out.

    Both jobs live behind one object because they must use the SAME weights — a producer that
    ranked with one snapshot and rolled out with another would be labelling states chosen by a
    policy that is not the one being measured. The `snapshot_loader` seam in :func:`main` exists so
    the end-to-end test can run the REAL bridge rollouts without conjuring a current-architecture
    checkpoint (the same substitution `cf_audit_integration_test` makes, and the same one only).
    """

    def __init__(self, path: str, step: int, model, mappings) -> None:
        self.path = path
        self.step = int(step)
        self.model = model
        self.mappings = mappings
        self._player_seq = 0

    # -- scoring ---------------------------------------------------------------------
    def score(self, obs: np.ndarray, masks: np.ndarray):
        """``(win_probs | None, entropies)`` for a BATCH of decisions — ONE forward.

        ``win_probs`` is None when the checkpoint trained no win-prob head; the caller must fall
        back to entropy-only ranking and SAY so, rather than reading a missing head as a
        confident 0.0.
        """
        import torch as th

        policy = self.model.policy
        ot = th.as_tensor(np.asarray(obs, dtype=np.float32))
        mt = th.as_tensor(np.asarray(masks))
        with th.no_grad():
            dist = policy.get_distribution({"observation": ot, "action_mask": mt})
            logits = dist.distribution.logits
            masked = th.where(mt.bool(), logits, th.full_like(logits, -1e8))
            probs = th.softmax(masked, 1).cpu().numpy()
            # The win-prob head's output is a STASH written by the forward we just ran, so it is
            # read off the same pass rather than paid for with a second one.
            fe = getattr(policy, "features_extractor", None)
            wp_logits = getattr(fe, "last_win_prob_logits", None) if fe is not None else None
            win_probs = (th.sigmoid(wp_logits[:, 0]).cpu().numpy()
                         if wp_logits is not None else None)
        ent = np.asarray([normalized_entropy(row) for row in probs], dtype=float)
        return win_probs, ent

    # -- players ---------------------------------------------------------------------
    def make_player(self, record: ReconstructionRecord, side: str, *, role: str):
        """A stochastic ``RLPlayer`` on ``side`` of ``record`` — see *THE ECOLOGY DECISION*.

        ``stochastic=True`` is the load-bearing half: a greedy copy of a net is strictly stronger
        than a temp-1.0 sample of it, so a greedy rollout would bias every label LOW against the
        regime the training actor actually plays (measured +0.037 [+0.007, +0.066] over 477
        sentinel states when this was got wrong on the prober's path).
        """
        from poke_env.ps_client import AccountConfiguration
        from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
        from agents.inference.player import RLPlayer

        self._player_seq += 1
        return RLPlayer(
            model=self.model, team=record.packed_team(side), battle_format=record.format_id,
            server_configuration=LocalhostServerConfiguration, mappings=self.mappings,
            account_configuration=AccountConfiguration(
                f"Cf{role}{self._player_seq % 100000}", "pw"),
            max_concurrent_battles=1, stochastic=True, start_listening=False)


def load_snapshot(path: str, step: Optional[int], *, device: str = "cpu",
                  compile_extractor: bool = False) -> Snapshot:
    """Load a checkpoint for scoring + rollouts.

    Uses the prober's `sanitized_load_custom_objects` (drop extractor kwargs the CURRENT
    constructor no longer accepts) rather than a bare `MaskablePPO.load`, for the same reason
    every other rollout path does: a checkpoint written one flag-deletion ago otherwise TypeErrors.
    The step is taken from the model's own ``num_timesteps`` — the authoritative number — with the
    filename as the fallback for a checkpoint that does not carry one.
    """
    from sb3_contrib import MaskablePPO
    from agents.observation.state_encoder import load_mappings
    from main.prober.model import sanitized_load_custom_objects

    custom_objects, _dropped = sanitized_load_custom_objects(path, device)
    model = MaskablePPO.load(path, env=None, device=device, custom_objects=custom_objects)
    model.policy.set_training_mode(False)
    # A `--log-level periodic` checkpoint carries an ObservationDebugger that print()s a DEEP TRACE
    # banner on every forward; it would drown the heartbeat this process communicates through.
    for mod in model.policy.modules():
        if hasattr(mod, "_debugger"):
            mod._debugger = None
    if compile_extractor:
        from agents.model.compile_opponents import maybe_compile_extractor
        maybe_compile_extractor(model, True, label=f"cf_producer:{os.path.basename(path)}",
                                hide_cuda=True)
    resolved = int(getattr(model, "num_timesteps", 0) or 0) or int(step or 0)
    return Snapshot(path, resolved, model, load_mappings())


# ---------------------------------------------------------------------------
# The producer state file
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProducerState:
    """``<run_dir>/cf_producer_state.json`` — the crash-safe processed set + the running totals.

    Human-readable on purpose (indented, one record name per entry): the producer has no
    TensorBoard, so this file and the heartbeat are the whole operator surface.
    """

    path: str
    version: int = 1
    sampler_version: str = SAMPLER_VERSION
    sampler_weights: dict = dataclasses.field(default_factory=lambda: dict(PRIORITY_WEIGHTS))
    processed: "List[str]" = dataclasses.field(default_factory=list)
    seq: int = 0
    labels_total: int = 0
    rollouts_total: int = 0
    records_processed: int = 0
    records_skipped: int = 0
    records_since_anchor: int = 0
    anchors_run: int = 0
    anchors_reproduced: int = 0
    #: Anchors that RAISED rather than returning a verdict (a wedged bridge, a contention
    #: timeout, a transport error). Counted apart from a MISMATCH — they refuse alike, but only
    #: one of them is evidence that the replay is inexact. `cf_audit` has always separated these.
    anchors_errored: int = 0
    #: Records the anchor DECLINED to adjudicate because they end in a forfeit — the class a live
    #: scripted replay cannot reproduce (`record_is_full_replay_anchorable`). A declared coverage
    #: bound, counted so it can never become invisible.
    anchors_skipped_unanchorable: int = 0
    #: Records that were ENUMERATED and then DELETED by the trainer's `--cf-records-keep` ring
    #: before this process could read them. A benign, EXPECTED race between two processes that
    #: share only a directory — never an error path, and never invisible. See `_note_vanished`.
    records_vanished: int = 0
    cycles: int = 0
    started_unix: float = dataclasses.field(default_factory=time.time)
    updated_unix: float = 0.0
    last_snapshot: dict = dataclasses.field(default_factory=dict)
    last_heartbeat: str = ""
    skip_reasons: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, run_dir: str) -> "ProducerState":
        path = os.path.join(run_dir, STATE_FILENAME)
        try:
            raw = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return cls(path=path)
        # Unknown keys are DROPPED, not fatal: a state file written by a newer producer must not
        # stop an older one, and the file is a cache of progress, never a contract.
        known = {f.name for f in dataclasses.fields(cls)} - {"path"}
        return cls(path=path, **{k: v for k, v in raw.items() if k in known})

    def __post_init__(self) -> None:
        self._processed_set = set(self.processed)

    def is_processed(self, name: str) -> bool:
        return name in self._processed_set

    def claim(self, name: str, *, keep: int = DEFAULT_KEEP_PROCESSED) -> None:
        """Mark a record processed. Called BEFORE its work — see *Crash safety*."""
        if name in self._processed_set:
            return
        self._processed_set.add(name)
        self.processed.append(name)
        if len(self.processed) > keep:
            drop = self.processed[: len(self.processed) - keep]
            self.processed = self.processed[len(self.processed) - keep:]
            self._processed_set.difference_update(drop)

    def note_skip(self, reason: str) -> None:
        self.records_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def save(self) -> None:
        self.updated_unix = time.time()
        body = {k: v for k, v in dataclasses.asdict(self).items() if k != "path"}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(body, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def write_label_batch(labels_dir: str, rows: "Sequence[dict]", *, step: int, seq: int) -> str:
    """One batch → one NEW file, written tmp-then-rename.

    A new file per batch rather than an append is the inode-safe shape: the buffer keys its byte
    offsets on ``(name, inode)`` precisely because a producer that recreates a file it already
    wrote used to have its first rows silently skipped. A name that is never reused cannot hit
    that path at all, and the atomic rename means a half-written batch is never visible."""
    os.makedirs(labels_dir, exist_ok=True)
    path = os.path.join(labels_dir, f"labels_cf_producer_{step}_{seq}.jsonl")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def label_row(*, record_path: str, decision: RecordDecision, wins: int, n: int,
              step: int, surprise: float, entropy: float, score: float,
              win_prob: Optional[float],
              outcome_label: Optional[float] = None,
              mc_return: Optional[float] = None, mc_return_n: int = 0,
              reward_sha1: str = "", reward_composition: str = "") -> dict:
    """The shared v1 schema, plus additive provenance the buffer ignores by design.

    The extra keys (`sampler_version`, `priority`, `label_regime`) are NOT schema changes — the
    buffer reads a fixed key set and ignores the rest — but they are the difference between a
    label file you can audit a year later and a bag of numbers.

    ``outcome_label`` / ``mc_return`` ARE read by the buffer (`gen3_cf_twin_heads_v1`) and are
    additive-optional at schema v1: an older consumer ignores them, a newer one supervises nothing
    extra when they are absent. See `cf_label_buffer`'s module docstring for why they ride the SAME
    row as the tight-MC label rather than arriving as their own ``kind`` — in one line, the buffer
    dedups on the obs digest, so a second row for one state would collide with the first and one of
    them would vanish, and one-row-per-state additionally makes "heads B and C saw identical states"
    structural rather than hoped-for.

    ``reward_composition`` is the human line beside the digest, for the same reason `format_reward_
    composition` is printed at launch: a hex digest tells a reader that two rewards DIFFER, and
    nothing about how.
    """
    lo, hi = wilson_ci(int(wins), int(n))
    row = LabelRow(
        battle=record_path, decision_idx=int(decision.index),
        obs_sha1=obs_digest(decision.obs), obs_npz=None, obs_inline=obs_b64(decision.obs),
        label=float(wins) / float(n) if n else 0.0, n_rollouts=int(n),
        wilson_lo=round(lo, 6), wilson_hi=round(hi, 6),
        policy_step=int(step), opponent=OPPONENT_LABEL,
    ).to_json()
    row.update(
        sampler_version=SAMPLER_VERSION,
        label_regime="self_current_stochastic_both_sides",
        turn=int(decision.turn),
        recorded_action=int(decision.action),
        priority={"score": round(float(score), 6),
                  "critic_surprise": round(float(surprise), 6),
                  "policy_entropy": round(float(entropy), 6),
                  "win_prob": (None if win_prob is None else round(float(win_prob), 6))},
        # HEAD B's stream: the RECORDED battle's realized outcome for this state — the single
        # Monte-Carlo sample the on-policy BCE already eats, on the states the factory selected.
        # That is the whole coverage arm: same states as C, single-outcome precision.
        outcome_label=(None if outcome_label is None else round(float(outcome_label), 6)),
    )
    if mc_return is not None:
        # THE SHADOW CRITIC's stream. Written only when it was actually measured; a `null` here and
        # an absent key mean the same thing to the buffer, but writing the reward provenance
        # unconditionally would imply a measurement that was not taken.
        row.update(mc_return=round(float(mc_return), 6), mc_return_n=int(mc_return_n),
                   reward_sha1=reward_sha1, reward_composition=reward_composition)
    return row


# ---------------------------------------------------------------------------
# The producer
# ---------------------------------------------------------------------------

def _outcome_scalar(record: ReconstructionRecord, side: str, outcome: dict) -> float:
    winner = outcome.get("winner")
    if not winner:
        return 0.5                      # a tie / the turn cap — maximally uninformative
    return 1.0 if winner == record.username(side) else 0.0


#: The substring that means "the box, not the replay" — it covers `TimeoutError`,
#: `asyncio.TimeoutError` and `utils.contention.ProgressTimeout` alike. A timeout is never a
#: semantic outcome (root `CLAUDE.md` → *Running beside a live training run*), so when the anchor
#: dies of one the message self-diagnoses instead of accusing the replay of being inexact.
_TIMEOUT_MARK = "Timeout"


def record_is_full_replay_anchorable(record: ReconstructionRecord) -> bool:
    """False for a record whose battle ended in a FORFEIT — the one class the anchor cannot judge.

    THE CLASS (root-caused 2026-08-23; this is the intermittent `ANCHOR REFUSED` the R1 composition
    test hit once). A battle that reaches ``StallConfig.threshold`` (= ``MAX_TURNS``, 250) is ended
    by ONE side forfeiting, and the bridge logs that as a ``['forcelose', <side>]`` entry in
    ``record.commands``. `install_scripted_prefix` builds each side's script as
    ``[c for (s, c) in commands if s == side]``, so ``'forcelose'`` matches NEITHER side and is
    silently dropped: the scripted replay has no way to reproduce the recorded forfeit. What it
    does instead is let BOTH players re-derive one from their own `_handle_stall` at turn >= 250,
    and whichever ``FORCELOSE`` the bridge processes first loses. In the recording only one side
    could forfeit at all (the training trainee, or — in the composition test — the `RecordingFuzzPlayer`
    against a plain poke-env `RandomPlayer` that has no stall handling), so the replay can hand the
    win to the side that actually LOST.

    MEASURED, on the saved fixture (`tmp/anchor_probe/fixture_stall_flip.json`, kept out of the
    tree — `tmp/` is gitignored): re-anchoring the SAME record refused **7/12 and 8/12** across two
    batches — a race, not a property of the record, and the only place the live scripted arm is
    non-deterministic (every non-forfeit record re-anchored 40/40 identical). Rebuilding the anchor
    with the OPPONENT's stall threshold set unreachable — mirroring the recording, where only one
    side could forfeit — made it **12/12 correct**, which is the mechanism proof.

    So this is a FAITHFULNESS LIMIT of the live-scripted-replay mechanism, not a bug the anchor can
    report: the offline replay driver gets it right every time because it replays the ORDERED
    command log including the ``forcelose``, while two poke-env players driven concurrently cannot
    reproduce that ordering. Excluding the class is a declared coverage bound — the anchor still
    adjudicates every non-forfeit record, and it is never retried until it passes.

    ⚠️ **The ROLLOUT path inherits the same asymmetry and it is NOT addressed here.** A label's
    rollouts play both sides with `RLPlayer`s that both stall-forfeit, whereas the recorded training
    battle had only the trainee forfeiting — so a rollout that runs to the 250-turn cap can be
    scored a WIN purely because the opponent's forfeit landed first. That biases the labels of long
    games upward. Task-ready, not fixed: see the training leaf.
    """
    return not any(side == "forcelose" for side, _ in record.commands)


def anchor_refusal_message(*, error: Optional[str], mismatch: "Optional[tuple]",
                           state_path: str) -> str:
    """What to print when the anchor refuses — and it must SAY WHICH refusal it was.

    Two failures reach this point and they have OPPOSITE diagnoses:

    * a **MISMATCH** — the scripted full replay reproduced a different outcome from the one the
      offline replay driver reports. Nothing sampled, so this IS a defect: the replay is inexact.
    * an **ERROR** — the anchor raised (a wedged bridge child, a transport error, a contention
      `ProgressTimeout`). It never returned a verdict, so it is not evidence about exactness at
      all; refusing is still correct (an anchor that did not complete has certified nothing), but
      the reader must not be sent hunting for a replay bug.

    The producer used to print the MISMATCH text for both. On 2026-08-23 that turned one flaky
    `cf_producer_integration_test` failure into an investigation of a replay-exactness gap that
    did not exist — the message asserted a cause the code had not established. `cf_audit` has
    always reported `anchor_errors` separately; this brings the stricter anchor into line.
    """
    tail = f"REFUSING to produce further (state: {state_path})."
    if error is not None:
        msg = (f"cf_producer: ANCHOR COULD NOT RUN — {error}\n"
               f"  The anchor RAISED instead of returning a verdict, so it says NOTHING about "
               f"whether the replay is exact. Refusing anyway (an anchor that did not complete "
               f"has certified nothing). {tail}")
        if _TIMEOUT_MARK in error:
            msg += f"\n  {describe_contention()}"
        return msg
    got_outcome, rec_winner, got, want, exhausted = (
        tuple(mismatch) if mismatch else (None, None, None, None, ()))
    why = (f"scripted replay → {got_outcome!r} [{got}], the record's own replay says winner "
           f"{rec_winner!r} [{want}]")
    if exhausted:
        why += (f"; and side(s) {list(exhausted)} RAN OUT of recorded commands and finished on "
                f"the live policy, which a full replay can only do after diverging")
    return (f"cf_producer: ANCHOR MISMATCH — the scripted full replay did not reproduce the "
            f"recorded outcome ({why}).\n"
            f"  Nothing was sampled, so this is a DEFECT, not a die roll: the replay is not "
            f"exact, and every label after this point would be a measurement of that bug. {tail}")


class CfProducer:
    """The cycle loop. Split from :func:`main` so a test can drive one cycle directly."""

    def __init__(self, args, *, snapshot_loader=None) -> None:
        self.args = args
        self.run_dir = os.path.abspath(args.run_dir)
        self.records_dir = os.path.join(self.run_dir, RECORDS_DIRNAME)
        self.labels_dir = os.path.join(self.run_dir, LABELS_DIRNAME)
        self.state = ProducerState.load(self.run_dir)
        self._loader = snapshot_loader or (
            lambda path, step: load_snapshot(path, step, device=args.device,
                                             compile_extractor=args.compile_extractor))
        self.snapshot: Optional[Snapshot] = None
        #: path -> the record READ AT ENUMERATION TIME (see `_load_batch`). `process_record` takes
        #: from here rather than re-opening a file the ring may have deleted in the meantime; a
        #: miss falls back to a real load, which is what every direct/test call still does.
        self._preloaded: "dict[str, ReconstructionRecord]" = {}
        self.label_times: "Deque[float]" = deque()
        self._last_new_ckpt_unix = time.time()
        self._producing = True
        self._said_stale = False
        self._said_no_win_head = False
        self._said_unanchorable = False
        self._warned_lag = False
        self._warned_no_return = False
        self._said_no_return_path = False
        self.anchor_failed = False
        #: Set by :meth:`run_anchor` when the anchor RAISED instead of returning a verdict — the
        #: exception text, so :func:`main` can say which of the two failures happened instead of
        #: asserting the one it cannot know (see `anchor_refusal_message`).
        self.anchor_error: Optional[str] = None
        #: The mismatch itself — ``(scripted outcome, recorded winner, got, want, exhausted)`` —
        #: so the refusal names WHAT disagreed instead of only that something did.
        self.anchor_mismatch: "Optional[tuple]" = None
        self.heartbeat = ""
        # gen3_cf_twin_heads_v1: the SHADOW critic's `mc_return` stream. A shaped return is a fact
        # about a board UNDER A REWARD COMPOSITION, so the config comes from the run's own recorded
        # `cli_args` where possible and the digest is stamped on every row — the consuming buffer
        # refuses a mismatch rather than averaging a foreign value function into the target.
        self._mc_return_on = bool(getattr(args, "mc_return", True))
        (self._reward_factory, self._reward_sha1, self._reward_composition,
         self._gamma) = self._resolve_reward_config()

    # -- the reward, for `mc_return` -------------------------------------------------
    def _resolve_reward_config(self):
        """``(reward_fn_factory, digest, composition_line, gamma)`` for THIS run.

        The GAMMA comes from the resolved `RewardConfig`, not from a CLI flag of its own, and that
        is a correctness property rather than tidiness: `reward_config_digest` hashes EVERY field
        of the config INCLUDING `gamma`, so folding the return at the config's discount puts the
        discount under the same guard as the rest of the reward. A separate `--gamma` would be
        state the digest cannot see — a mistyped one would ship returns folded at the wrong
        discount while `cf/labels_mc_return_rejected_total` stayed 0 and every liveness counter read
        healthy. `--gamma` survives only as an explicit OVERRIDE for a reader who knows better.

        Read from ``<run>/metadata.json``'s recorded ``cli_args`` through the SAME
        ``RewardConfig.from_args`` the trainer uses — never a hand-rebuilt config, because a reward
        assembled two different ways is exactly the drift the digest exists to detect, and a
        producer that reconstructs it independently would eventually detect its own reconstruction.

        When the metadata cannot be read the DEFAULT config is used and the fact is printed LOUDLY.
        That is not a silent fall-back: the digest of the default config will simply not match a
        run whose reward differs, and the trainer-side buffer will reject every ``mc_return`` and
        say so. A wrong label that announces itself twice is the intended failure mode.
        """
        import functools
        from agents.training.reward_manager import (
            Gen3RewardManager, RewardConfig, format_reward_composition, reward_config_digest)

        cfg = None
        meta = os.path.join(self.run_dir, "metadata.json")
        try:
            cli_args = json.loads(Path(meta).read_text()).get("cli_args") or {}
            if cli_args:
                cfg = RewardConfig.from_args(argparse.Namespace(**cli_args))
        except Exception as exc:                                        # noqa: BLE001
            self._log(f"⚠️  could not read this run's reward config from {meta} "
                      f"({type(exc).__name__}) — falling back to the DEFAULT RewardConfig. Any "
                      f"mc_return label will be REJECTED by a trainer whose reward differs "
                      f"(cf/labels_mc_return_rejected_total), which is the intended failure.")
        if cfg is None:
            cfg = RewardConfig()
        _g = getattr(self.args, "gamma", None)
        gamma = float(_g) if _g is not None else float(cfg.gamma)
        return (functools.partial(Gen3RewardManager, config=cfg),
                reward_config_digest(cfg), format_reward_composition(cfg), gamma)

    # -- helpers ---------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        print(f"[cf_producer] {msg}", flush=True)

    def _rate_per_hour(self) -> float:
        now = time.time()
        while self.label_times and now - self.label_times[0] > 3600.0:
            self.label_times.popleft()
        return float(len(self.label_times))

    def _throttled(self) -> bool:
        return self._rate_per_hour() >= self.args.max_labels_per_hour

    def _pending_records(self) -> "List[str]":
        """Unprocessed records, **NEWEST FIRST**.

        ⚠️ THE ORDER IS THE FIX, not a preference. The trainer's `--cf-records-keep` ring prunes
        `cf_records/` to the newest N from every env worker, so the OLDEST pending record is the one
        about to be deleted — and this loop used to walk exactly that end first. Measured on
        `ai_v9_29_rev1_0823`: 176 records died of `FileNotFoundError` in 67 cycles with "538
        pending" against a ring of 512, i.e. the producer spent its cycle on records the ring had
        already guaranteed it would lose. Newest-first puts the deletion end of the ring at the LOW
        end of the work queue, where losing a record costs nothing.

        It is also the right SAMPLER order independently: a newer record was produced by a policy
        closer to the one the label will supervise, and the label's freshness is bounded by
        `--cf-label-lag-steps` at the consumer either way.
        """
        try:
            names = sorted(n for n in os.listdir(self.records_dir) if n.endswith(RECON_SUFFIX))
        except OSError:
            return []
        return [os.path.join(self.records_dir, n) for n in reversed(names)
                if not self.state.is_processed(n)]

    def _note_vanished(self, name: str) -> None:
        """A record that was enumerated and then deleted by the ring. COUNTED, never raised.

        Two processes share one directory and only one of them owns its lifetime, so this is an
        ordinary outcome of the design, not a failure of either side. Reporting it as an exception
        put it in `skip_reasons` as `error:FileNotFoundError` — indistinguishable from a corrupt
        record or a replay crash — and on the anchor path it reached `anchors_errored` and could
        exit 3. It gets its own counter so the operator can see the ring pressure as the number it
        is, and the fix is a bigger `--cf-records-keep`, never a code change here.
        """
        self.state.records_vanished += 1
        if self.state.records_vanished == 1:
            self._log(f"⚠️  {name} was deleted by the trainer's --cf-records-keep ring before this "
                      f"process could read it. That is a BENIGN race (two processes, one "
                      f"directory, one owner) and is counted as `records_vanished`, never an "
                      f"error. If the count grows, raise --cf-records-keep on the trainer or lower "
                      f"--records-per-cycle here. This explanation is printed once.")
        else:
            self._log(f"record {name} vanished (ring deletion #{self.state.records_vanished})")

    def _load_batch(self, paths: "Sequence[str]") -> "List[str]":
        """READ this cycle's records into memory NOW, at enumeration time. Returns the survivors.

        The enumerate → read gap is what the ring wins: a cycle enumerates, then anchors (a FULL
        scripted replay — seconds to minutes), then claims + fsyncs the state file per record, and
        only then opens the file. The ring deletes throughout. Reading immediately shrinks the
        window to the length of this loop, and everything after it works from a `ReconstructionRecord`
        that no longer has a file to lose.
        """
        alive: "List[str]" = []
        for path in paths:
            name = os.path.basename(path)
            try:
                self._preloaded[path] = ReconstructionRecord.load(path)
            except FileNotFoundError:
                # The record is GONE — it can never be retried, so claim it (a name the state file
                # will never see on disk again) and count it. Any OTHER load failure (a corrupt or
                # half-written record) is deliberately NOT caught here: it belongs to
                # `process_record`'s existing claim-then-report path, which names the exception.
                self.state.claim(name, keep=self.args.keep_processed)
                self._note_vanished(name)
                continue
            except Exception:                                           # noqa: BLE001
                alive.append(path)      # let process_record re-raise it and report it by type
                continue
            alive.append(path)
        return alive

    def _newest_record(self) -> Optional[str]:
        """The newest record that the full-replay anchor can actually adjudicate.

        Newest-first, SKIPPING any record that ends in a forfeit — see
        :func:`record_is_full_replay_anchorable` for why that class is not adjudicable by a live
        scripted replay at all. The skip is counted (`anchors_skipped_unanchorable`) and announced
        once; it is a declared COVERAGE BOUND of the oracle, in the same family as `cf_audit`'s
        turn-1 and forced-switch bounds — not a retry, and never a second attempt at the same
        record."""
        try:
            names = sorted(n for n in os.listdir(self.records_dir) if n.endswith(RECON_SUFFIX))
        except OSError:
            return None
        for name in reversed(names):
            path = os.path.join(self.records_dir, name)
            try:
                anchorable = record_is_full_replay_anchorable(ReconstructionRecord.load(path))
            except FileNotFoundError:
                # The ring deleted it between the listdir and this read. NOT an anchor error —
                # that path exits 3 — just an older record to anchor on instead.
                self._note_vanished(name)
                continue
            except Exception:                                           # noqa: BLE001
                return path        # unreadable → let run_anchor report it as an ANCHOR ERROR
            if anchorable:
                return path
            self.state.anchors_skipped_unanchorable += 1
            if not self._said_unanchorable:
                self._said_unanchorable = True
                self._log(f"⚠️  {name} ends in a FORFEIT, which a live scripted replay cannot "
                          f"adjudicate (both sides re-derive the stall forfeit and the winner "
                          f"becomes a race) — skipping it for the anchor and taking an older "
                          f"record. This is a declared coverage bound, not a retry. Said once.")
        return None

    # -- snapshot --------------------------------------------------------------------
    def refresh_snapshot(self) -> bool:
        """Load the freshest checkpoint if it is newer than the one in hand. False = none yet."""
        found = resolve_latest_checkpoint(self.run_dir)
        if found is None:
            return False
        path, step = found
        if self.snapshot is not None and path == self.snapshot.path:
            return True
        try:
            snap = self._loader(path, step)
        except Exception as exc:                                        # noqa: BLE001
            self._log(f"could not load {os.path.basename(path)} "
                      f"({type(exc).__name__}: {str(exc)[:200]}) — retrying next cycle")
            return self.snapshot is not None
        self.snapshot = snap
        self._last_new_ckpt_unix = time.time()
        if not self._producing:
            self._producing = True
            self._said_stale = False
            self._log(f"a NEW checkpoint appeared (step {snap.step}) — resuming production")
        self.state.last_snapshot = {"path": path, "step": snap.step,
                                    "loaded_unix": self._last_new_ckpt_unix}
        self._log(f"snapshot → {os.path.basename(path)} (step {snap.step:,})")
        return True

    def _check_trainer_alive(self) -> None:
        """Stop producing when no NEW checkpoint has appeared for a long time.

        The producer stamps its own snapshot's step on every label, and the buffer expires a row
        whose |age| exceeds ``--cf-label-lag-steps``. So a producer that keeps running against a
        frozen checkpoint is not merely useless — it burns the box filling a buffer whose rows the
        trainer will expire, or worse, teaching the current policy an ancestor's values if the
        trainer is alive but not checkpointing. It keeps WATCHING (a restarted trainer resumes it)
        and says so exactly once in each direction."""
        stale_after = self.args.stale_checkpoint_minutes * 60.0
        if stale_after <= 0 or not self._producing:
            return
        idle = time.time() - self._last_new_ckpt_unix
        if idle > stale_after:
            self._producing = False
            if not self._said_stale:
                self._said_stale = True
                self._log(f"⚠️  NO new checkpoint for {idle / 60:.0f} min "
                          f"(> --stale-checkpoint-minutes {self.args.stale_checkpoint_minutes}) — "
                          f"the trainer is probably gone. PAUSING production rather than filling "
                          f"the buffer with stale-policy labels. Still watching.")

    # -- the anchor ------------------------------------------------------------------
    def _anchor_error(self, path: str, exc: BaseException) -> bool:
        """The anchor RAISED. Always False (an anchor that did not complete has certified
        nothing), counted apart from a MISMATCH, and it says which one it was."""
        self.anchor_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        self._log(f"ANCHOR ERROR on {os.path.basename(path)}: {self.anchor_error}")
        self.state.anchors_run += 1
        self.state.anchors_errored += 1
        return False

    def run_anchor(self) -> Optional[bool]:
        """The full-replay correctness oracle. None = no record to anchor on.

        Fully SCRIPTED (``divergence_turn=None``): no policy acts, so this isolates
        record → offline replay → live bridge → outcome with no sampling in it, and a MISMATCH is
        unambiguously a defect rather than a die roll.

        ⚠️ **A raised exception is NOT a mismatch, and the two must not report as one.** Both
        return ``False`` (an anchor that did not complete has certified nothing, so refusing is
        still right — `cf_audit` counts a crashed anchor as a failure too), but they have opposite
        diagnoses: a mismatch says the replay is inexact, while a `ProgressTimeout` / transport
        error on a saturated box says nothing whatever about exactness. Conflating them cost a
        whole investigation on 2026-08-23 — the test's ONE failure was read as a replay-exactness
        gap because the top-level message asserted that cause for both. The distinction is
        recorded here and rendered by :func:`anchor_refusal_message`; the state file separates
        ``anchors_errored`` from ``anchors_run - anchors_reproduced``, as `cf_audit` always has."""
        path = self._newest_record()
        if path is None or self.snapshot is None:
            return None
        try:
            record = ReconstructionRecord.load(path)
        except FileNotFoundError:
            # Deleted by the ring between `_newest_record`'s read and this one. An anchor that had
            # no record to run on is not an anchor that FAILED — returning False here would exit 3
            # over an ordinary ring deletion.
            self._note_vanished(os.path.basename(path))
            return None
        except Exception as exc:                                        # noqa: BLE001
            # A record that is UNREADABLE for any other reason (corrupt, half-written) is the
            # anchor ERROR it always was — only the vanished case is exempted above.
            return self._anchor_error(path, exc)
        try:
            side = _trainee_side(record)
            rec = dataclasses.replace(record, trainee_username=record.username(side))
            expected = replay_battle(record, impl=self.args.impl).outcome
            want = _outcome_scalar(record, side, expected)
            res = _run_one(
                rec,
                trainee=self.snapshot.make_player(rec, side, role="A"),
                opponent=self.snapshot.make_player(rec, _other(side), role="B"),
                divergence_turn=None, substitute_choice=None,
                seed=record.start_options().get("seed"), impl=self.args.impl)
            got = {"win": 1.0, "loss": 0.0}.get(res["outcome"], 0.5)
        except Exception as exc:                                        # noqa: BLE001
            return self._anchor_error(path, exc)
        self.state.anchors_run += 1
        # THE STRICTER HALF, and the sensitive one. A `divergence_turn=None` replay scripts every
        # decision of both sides, so a side that runs OUT of recorded commands and finishes on the
        # live policy has already diverged from the recording — whatever the winner turned out to
        # be. Comparing outcomes alone catches that only when the random fallback happens to flip
        # the result, i.e. about half the time; this catches the divergence itself. Measured
        # 2026-08-23: `script_exhausted` was empty on all 274 instrumented healthy replays.
        exhausted = list(res.get("script_exhausted") or ())
        ok = (got == want) and not exhausted
        self.state.anchors_reproduced += int(ok)
        self.state.records_since_anchor = 0
        if not ok:
            self.anchor_mismatch = (res["outcome"], expected.get("winner"), got, want, exhausted)
        self._log(f"ANCHOR {'OK' if ok else 'FAILED'} on {os.path.basename(path)} "
                  f"(scripted full replay → {res['outcome']}, record says "
                  f"{expected.get('winner')}"
                  f"{f', script EXHAUSTED on {exhausted}' if exhausted else ''})")
        return ok

    # -- one record ------------------------------------------------------------------
    def process_record(self, path: str) -> "List[dict]":
        """Replay, rank, roll out. Returns the label rows (possibly empty)."""
        assert self.snapshot is not None
        # Preloaded by `_load_batch` at enumeration time; `pop` so the cycle cannot leak memory
        # across records. A direct call (a test, a one-shot) still loads from disk.
        record = self._preloaded.pop(path, None) or ReconstructionRecord.load(path)
        side = _trainee_side(record)
        rep = replay_battle(record, impl=self.args.impl)
        chunks = rep.p1_chunks if side == "p1" else rep.p2_chunks
        outcome = _outcome_scalar(record, side, rep.outcome)
        decisions = scan_record(record, side, chunks=chunks,
                               mappings=self.snapshot.mappings, impl=self.args.impl)

        seen_turns: set = set()
        candidates: "List[RecordDecision]" = []
        for d in decisions:
            if d.obs is None or d.turn < MIN_LABELABLE_TURN or not is_move_round(d.mask):
                continue
            if d.turn in seen_turns:      # a mid-turn round; the divergence anchors at the first
                continue
            seen_turns.add(d.turn)
            candidates.append(d)
        if not candidates:
            self.state.note_skip("no_labelable_decision")
            return []

        win_probs, entropies = self.snapshot.score(
            np.stack([d.obs for d in candidates]),
            np.stack([np.asarray(d.mask) for d in candidates]))
        if win_probs is None and not self._said_no_win_head:
            self._said_no_win_head = True
            self._log("⚠️  this checkpoint has NO win-prob head (--win-prob-mode none) — the "
                      "critic-surprise term is UNAVAILABLE, so decisions are ranked on policy "
                      "entropy alone. Said once.")

        scored = []
        for k, d in enumerate(candidates):
            wp = None if win_probs is None else float(win_probs[k])
            s = critic_surprise(wp, outcome)
            e = float(entropies[k])
            scored.append((priority_score(s, e), s, e, wp, d))
        scored.sort(key=lambda t: (-t[0], t[4].index))

        rows: "List[dict]" = []
        for score, s, e, wp, d in scored[: self.args.top_n]:
            if self._throttled():
                self._log(f"throttle: {self._rate_per_hour():.0f} labels in the last hour "
                          f"(--max-labels-per-hour {self.args.max_labels_per_hour}) — "
                          f"stopping this record early")
                break
            wins, n, returns = self._rollout(record, side, d, tag=os.path.basename(path))
            if n == 0:
                self.state.note_skip("rollouts_all_failed")
                continue
            self.label_times.append(time.time())
            self.state.rollouts_total += n
            rows.append(label_row(
                record_path=path, decision=d, wins=wins, n=n,
                step=self.snapshot.step, surprise=s, entropy=e,
                score=score, win_prob=wp,
                # HEAD B's label is the RECORDED battle's realized outcome — already computed above
                # for the critic-surprise term, so it costs nothing. It is the SAME quantity the
                # on-policy BCE eats, on the states the sampler selected: that identity is what
                # makes B-A a read of COVERAGE alone.
                outcome_label=outcome,
                mc_return=(float(np.mean(returns)) if returns else None),
                mc_return_n=len(returns),
                reward_sha1=self._reward_sha1, reward_composition=self._reward_composition))
        return rows

    def _rollout(self, record: ReconstructionRecord, side: str, d: RecordDecision,
                 *, tag: str) -> "tuple[int, int, List[float]]":
        """R continuations from ``d``: the RECORDED action, then both sides live on fresh dice.

        Returns ``(wins, n, returns)``. ``returns`` is the per-rollout DISCOUNTED SHAPED RETURN from
        the labelled decision (`gen3_cf_twin_heads_v1`, the shadow critic's stream) and is EMPTY
        when `--no-mc-return` is set or when a rollout produced no measurable return — never
        zero-padded, because a zero return is the middle of this reward's range and would read as a
        genuinely neutral game rather than as a missing measurement.

        The tracking is armed at the DIVERGENCE decision through the scripted-prefix hook, not at
        the first live one: our turn-T move is scripted (it IS the substitute), so a live-only hook
        would miss ``r_T`` and shift the whole return by one turn against the state it labels.
        """
        from main.prober.falsifier import fresh_seeds
        from agents.training.cf_mc_return import attach_return_recording

        rec = dataclasses.replace(record, trainee_username=record.username(side))
        seed = record.start_options().get("seed")
        seeds = fresh_seeds(self.args.rollouts, salt=f"{tag}:{d.index}:cfp{self.args.seed}")
        wins = n = 0
        returns: "List[float]" = []
        for ps in seeds:
            trainee = self.snapshot.make_player(rec, side, role="T")
            recorder = None
            hook = None
            if self._mc_return_on:
                recorder = attach_return_recording(
                    trainee, reward_fn_factory=self._reward_factory)
            if recorder is None and self._mc_return_on and not self._said_no_return_path:
                self._said_no_return_path = True
                self._log("⚠️  this snapshot's player exposes no obs/order seam, so no shaped "
                          "RETURN can be measured — rows will carry NO mc_return and the SHADOW "
                          "critic will train on nothing. (Expected only under a test double; a "
                          "real RLPlayer supports it.) Said once.")
            if recorder is not None:
                fired = {"done": False}

                def hook(battle, obs, _choice, _rec=recorder, _t=int(d.turn),
                         _a=int(d.action), _f=fired):
                    # Fires on every OUR-side scripted decision; only the DIVERGENCE turn matters.
                    # The recorder decides nothing — this closure does, which is what keeps reward
                    # tracking out of `utils.bridge.counterfactual` entirely.
                    #
                    # ⚠️ ARM **THEN** NOTE, in that order, and this line is the whole point of the
                    # hook. Turn T's move is SCRIPTED (it IS the substitute), so it never reaches
                    # the live `choose_move` the recorder is otherwise driven by. Arming alone —
                    # leaving the note to the first LIVE decision at T+1 — would make T+1 the armed
                    # decision and drop r_T, which is exactly the off-by-one the hook exists to
                    # close: the label would be G(s_{T+1}) against an obs row for s_T, biased by
                    # whatever happened on the divergence turn (a KO there is the largest single
                    # shaping term), i.e. correlated with the state and shaped like a real signal.
                    if _f["done"] or int(getattr(battle, "turn", 0) or 0) != _t:
                        return
                    mask = (obs or {}).get("action_mask")
                    if mask is None:                       # a non-Gen3 player; nothing to record
                        return
                    _f["done"] = True
                    _rec.arm_at_next()
                    _rec.note(battle, _a, mask)
            try:
                res = _run_one(
                    rec,
                    trainee=trainee,
                    opponent=self.snapshot.make_player(rec, _other(side), role="O"),
                    divergence_turn=int(d.turn), substitute_choice=d.choice,
                    seed=seed, post_t_seed=ps, impl=self.args.impl,
                    trainee_decision_hook=hook)
            except Exception as exc:                                    # noqa: BLE001
                self._log(f"rollout failed ({tag} inv {d.index}): "
                          f"{type(exc).__name__}: {str(exc)[:200]}")
                continue
            n += 1
            wins += int(res["outcome"] == "win")
            if recorder is not None:
                g = recorder.value(self._gamma)
                if g is not None:
                    returns.append(g)
                elif not self._warned_no_return:
                    self._warned_no_return = True
                    self._log("⚠️  a rollout produced NO measurable shaped return (the divergence "
                              "decision was never reached, or the battle ended in the prefix) — "
                              "that rollout contributes no mc_return. Said once; watch "
                              "cf/mc_return_coverage on the trainer side.")
        return wins, n, returns

    # -- one cycle -------------------------------------------------------------------
    def cycle(self) -> int:
        """One pass. Returns an exit code: 0 = keep going, 3 = the anchor refused."""
        t0 = time.perf_counter()
        self.state.cycles += 1
        if not self.refresh_snapshot():
            self._emit_heartbeat(t0, new=0, produced=0, note="waiting for a checkpoint")
            self.state.save()
            return 0
        self._check_trainer_alive()

        if self.state.anchors_run == 0 or (
                self.args.anchor_every > 0
                and self.state.records_since_anchor >= self.args.anchor_every):
            ok = self.run_anchor()
            if ok is False:
                self.anchor_failed = True
                self.state.save()
                return 3

        # Enumerated AFTER the anchor, deliberately: a full scripted replay takes seconds to
        # minutes, and every one of them is ring-deletion time against a list built before it.
        pending = self._pending_records()

        produced = 0
        if self._producing and not self._throttled():
            # NEWEST FIRST (`_pending_records`) and READ NOW (`_load_batch`) — the two halves of
            # the producer/retention race. Everything below works from an in-memory record.
            for path in self._load_batch(pending[: self.args.records_per_cycle]):
                # CLAIM FIRST (see *Crash safety*): the record is recorded as processed and the
                # state is fsync-replaced BEFORE any rollout runs, so a crash can lose this
                # record's labels but can never emit them twice.
                self.state.claim(os.path.basename(path), keep=self.args.keep_processed)
                self.state.records_processed += 1
                self.state.records_since_anchor += 1
                self.state.save()
                try:
                    rows = self.process_record(path)
                except Exception as exc:                                # noqa: BLE001
                    self.state.note_skip(f"error:{type(exc).__name__}")
                    self._log(f"record {os.path.basename(path)} failed: "
                              f"{type(exc).__name__}: {str(exc)[:300]}")
                    continue
                if not rows:
                    continue
                self.state.seq += 1
                out = write_label_batch(self.labels_dir, rows,
                                        step=self.snapshot.step, seq=self.state.seq)
                self.state.labels_total += len(rows)
                produced += len(rows)
                self._log(f"{len(rows)} labels → {os.path.basename(out)}")
                if self._throttled():
                    break

        self._preloaded.clear()      # nothing survives a cycle: the next one re-reads what is live
        self._emit_heartbeat(t0, new=len(pending), produced=produced)
        self.state.save()
        return 0

    def _emit_heartbeat(self, t0: float, *, new: int, produced: int, note: str = "") -> None:
        snap = self.snapshot
        latest = resolve_latest_checkpoint(self.run_dir)
        lag = 0
        if snap is not None and latest is not None and latest[1] is not None:
            lag = max(0, int(latest[1]) - snap.step)
        if (lag > self.args.lag_warn_steps and not self._warned_lag):
            self._warned_lag = True
            self._log(f"⚠️  the snapshot in hand is {lag:,} steps behind the newest checkpoint "
                      f"(> --lag-warn-steps {self.args.lag_warn_steps:,}) — labels stamped at this "
                      f"step may EXPIRE at the consumer's --cf-label-lag-steps bound.")
        anchors = f"{self.state.anchors_reproduced}/{self.state.anchors_run}"
        if self.state.anchors_errored:
            # NEVER folded into the reproduced/run ratio: an anchor that could not run is not an
            # anchor that disagreed, and only the second says the replay is inexact.
            anchors += f" ({self.state.anchors_errored} errored)"
        hb = (f"cycle {self.state.cycles} | "
              f"snapshot {'step ' + format(snap.step, ',') if snap else 'NONE'}"
              f"{f' (lag {lag:,})' if lag else ''} | "
              f"records {new} pending / {self.state.records_processed} done"
              f"{f' / {self.state.records_skipped} skipped' if self.state.records_skipped else ''}"
              # A number, not an error: `records_vanished` is ring PRESSURE (raise
              # --cf-records-keep), and it is on the heartbeat so it can never be invisible.
              f"{f' / {self.state.records_vanished} vanished' if self.state.records_vanished else ''} | "
              f"labels {produced} (+{self.state.labels_total} total, "
              f"{self._rate_per_hour():.0f}/h) | "
              f"anchor {anchors} | "
              f"{'PRODUCING' if self._producing else 'PAUSED'} | "
              f"load {os.getloadavg()[0]:.1f} | {time.perf_counter() - t0:.1f}s"
              + (f" | {note}" if note else ""))
        self.heartbeat = hb
        self.state.last_heartbeat = hb
        self._log(hb)


def _other(side: str) -> str:
    return "p2" if side == "p1" else "p1"


def _trainee_side(record: ReconstructionRecord) -> str:
    """Which side the TRAINEE held.

    A training record names no trainee (the tap writes the raw ``__RECON__``, and only the eval
    path merges a ``trainee_username`` in), so the answer comes from the transport's own
    invariant: ``BridgeSession`` seats ``env.agent1`` — the trainee — on **p1**, always. A record
    that DOES name one (an eval sibling handed to this tool) is honoured instead."""
    if record.trainee_username:
        return record.side_of(record.trainee_username)
    return "p1"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agents.training.cf_producer",
        description="Background tight-MC label producer: <run>/cf_records/ -> <run>/cf_labels/.")
    p.add_argument("run_dir", help="a models/<run> directory carrying cf_records/ (--cf-records)")
    p.add_argument("--rollouts", type=int, default=8,
                   help="R, rollouts per label (default 8 — the per-state SNR is ~2-4:1 there)")
    p.add_argument("--top-n", type=int, default=3,
                   help="decisions labelled per record (default 3). This is a SAMPLER; raising it "
                        "toward the record's decision count turns it into a sweep and re-weights "
                        "long games")
    p.add_argument("--records-per-cycle", type=int, default=4,
                   help="how many new records one cycle consumes (default 4)")
    p.add_argument("--max-labels-per-hour", type=int, default=2000,
                   help="throughput cap (default 2000). The producer shares the box with a "
                        "trainer; this is the knob that keeps it a sidecar")
    p.add_argument("--cycle-seconds", type=float, default=30.0,
                   help="sleep between cycles (default 30)")
    p.add_argument("--anchor-every", type=int, default=50,
                   help="run the recorded-dice anchor every N records (default 50; 0 = only at "
                        "startup). A FAILURE exits 3 — see the module docstring")
    p.add_argument("--stale-checkpoint-minutes", type=float, default=90.0,
                   help="pause production when no NEW checkpoint has appeared for this long "
                        "(default 90; 0 disables). The trainer is probably gone")
    p.add_argument("--lag-warn-steps", type=int, default=150_000,
                   help="warn once when the snapshot in hand falls this far behind the newest "
                        "checkpoint (default 150000 = the buffer's --cf-label-lag-steps default)")
    p.add_argument("--keep-processed", type=int, default=DEFAULT_KEEP_PROCESSED,
                   help="how many processed-record names the state file remembers (default 4096)")
    p.add_argument("--impl", default="rust", choices=["node", "rust"],
                   help="replay driver AND live bridge impl (default rust)")
    p.add_argument("--device", default="cpu", help="torch device for the snapshot (default cpu)")
    p.add_argument("--compile-extractor", action="store_true",
                   help="torch.compile the snapshot's extractor (amortizes over a long session)")
    p.add_argument("--seed", type=int, default=20260822,
                   help="salts the post-divergence dice, so a re-run reproduces")
    p.add_argument("--mc-return", "--mc_return", dest="mc_return",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="also measure the DISCOUNTED SHAPED RETURN of each rollout and ship it as "
                        "the row's `mc_return` (the SHADOW CRITIC's label stream, "
                        "gen3_cf_twin_heads_v1). Default ON — the rollouts already run, so the "
                        "only cost is the server-free reward path. --no-mc-return omits the field "
                        "entirely, which is what a consumer without --cf-shadow-critic wants.")
    p.add_argument("--gamma", type=float, default=None,
                   help="OVERRIDE the discount used to fold a rollout's rewards into a return. "
                        "Default: the run's OWN RewardConfig.gamma, read from its metadata — which "
                        "is the right default because `reward_config_digest` hashes gamma along "
                        "with every other reward field, so the discount rides the same GIGO guard "
                        "as the reward itself. Passing this puts the discount OUTSIDE that guard: "
                        "a wrong value ships returns folded against a different value function "
                        "with every liveness counter reading healthy.")
    p.add_argument("--cycles", type=int, default=0,
                   help="stop after N cycles (default 0 = run forever). --cycles 1 is the smoke")
    return p


def main(argv: "Optional[Sequence[str]]" = None, *, snapshot_loader=None) -> int:
    """``snapshot_loader(path, step) -> Snapshot`` is the ONE substitution point.

    It exists so the end-to-end composition test can run REAL bridge rollouts over a REAL ring
    record without a current-architecture checkpoint on disk — the same single substitution
    `cf_audit_integration_test` makes, and for the same reason. Production always takes the
    default."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.run_dir):
        print(f"cf_producer: {args.run_dir} is not a directory", file=sys.stderr)
        return 2
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:                                                 # pragma: no cover
        pass

    prod = CfProducer(args, snapshot_loader=snapshot_loader)
    records_dir = prod.records_dir
    if not os.path.isdir(records_dir):
        print(f"cf_producer: no {records_dir} — the run was not launched with --cf-records, so "
              f"there is nothing to label.", file=sys.stderr)
        return 2
    prod._log(f"watching {records_dir} → {prod.labels_dir}  "
              f"(R={args.rollouts}, top-{args.top_n}, impl={args.impl}, "
              f"sampler={SAMPLER_VERSION})")
    prod._log("OPPONENT ECOLOGY: rollouts play the CURRENT snapshot on BOTH sides, stochastic — "
              "a training record names no opponent. Every label says opponent=self_current.")

    n = 0
    try:
        while True:
            rc = prod.cycle()
            if rc:
                print("\n" + anchor_refusal_message(
                    error=prod.anchor_error, mismatch=prod.anchor_mismatch,
                    state_path=prod.state.path), file=sys.stderr)
                return rc
            n += 1
            if args.cycles and n >= args.cycles:
                return 0
            time.sleep(max(0.0, args.cycle_seconds))
    except KeyboardInterrupt:
        prod._log("interrupted — state saved")
        prod.state.save()
        return 0


if __name__ == "__main__":
    sys.exit(main())
