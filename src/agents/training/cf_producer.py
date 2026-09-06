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
   action at that turn, then both sides live to termination on fresh post-divergence dice. A
   continuation that reaches the 250-turn stall-forfeit cap is a **DRAW AT CAP** and scores 0.5 —
   see :meth:`CfProducer._play_arm`; the per-label count rides out as ``n_capped``.
4b. **Optionally sweep the SIBLING actions** (``--q-labels``, `gen3_cf_q_labels_v1`, OFF by
   default and byte-identical when off) — the same rollout, once per LEGAL action, on the SAME
   dice, producing the row's ``q_labels`` block. That is the supply side of the v107 Q win-prob
   head, which shipped as a trained consumer of a stream nothing wrote. The arithmetic — the
   common-random-number pairing, the declared ``cf_q_sweep_v1`` selection rule, and why the
   recorded action's arm is free — lives in `cf_q_labels.py`; the cost is in § *Where the time
   goes* and multiplies by the ARM COUNT, so it is metered per row and on the heartbeat.
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

Four modules, one factory
-------------------------
This file owns the LOOP and everything with state in it: the cycle, the record ring's consumer
side, the crash-safe ``ProducerState``, the anchor, the rollout arms, the heartbeat and the CLI.
Three pieces that need nothing the loop knows live beside it (2026-09-06, the file-size ratchet's
third cut of the 1,000-2,000 band, 1899 → 1484 lines). **This module re-imports every public name
from all three**, so ``from agents.training.cf_producer import label_row`` still resolves and no
caller moved:

* **`cf_producer_sampler.py`** — the DECLARED, VERSIONED priority (``cf_producer_priority_v1``):
  ``SAMPLER_VERSION`` / ``PRIORITY_WEIGHTS`` / ``MIN_LABELABLE_TURN`` and the four pure functions
  that rank a candidate (``critic_surprise`` — the conviction region — plus ``normalized_entropy``,
  ``priority_score`` and the ``is_move_round`` filter). It is a SAMPLER, not a sweep, so a silent
  change here is a distribution-shift confound for every downstream readout; the weights and the
  reasoning for them travel together in that module's docstring.
* **`cf_producer_snapshot.py`** — WHICH weights are running (``resolve_latest_checkpoint``,
  ``step_from_checkpoint_name``) and the ``Snapshot`` that scores decisions and builds the rollout
  players from them. Almost everything non-obvious there is a `torch.compile` shape/dtype fact,
  each with its own measurement.
* **`cf_producer_labels.py`** — the v1 label ROW (``label_row``, ``OPPONENT_LABEL``) and the batch
  file it is written into (``write_label_batch``). A CONTRACT with `cf_label_buffer`, which knows
  nothing about this loop.

The extraction changed nothing a caller can see, and that is EVIDENCE rather than a promise:
`cf_producer_test.py`'s extraction-parity golden digests every public entry point of this module —
the arithmetic, the state file, the row, the batch, the refusal texts and the CLI defaults — and
was captured BEFORE the cut and reproduced byte-for-byte after it.

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

Where the time goes (MEASURED 2026-08-23, and it is not where the cost model said)
----------------------------------------------------------------------------------
Profiled on three live `ai_v9_29_rev1_0823` ring records against that run's own checkpoint
(`tmp`-copied, read-only), beside the live trainer at load ~15-25:

===================================================  =========  ======
stage                                                 share      note
===================================================  =========  ======
rollouts (R × `replay_counterfactual`)                **93%**    of which 93% is `choose_move`
`replay_battle` (the offline replay driver)              ~0.2%   ~35 ms/record, once
`scan_record` (replay + obs materialize)                 ~0.3%   ~170 ms/record, once
`score` (the ranking forward)                            ~0.1%
record parse / label write                              <0.1%
===================================================  =========  ======

Inside a rollout, **the live policy forward is essentially the whole thing**: 832 `choose_move`
calls at 15.4 ms each against 0.40 ms of prefix `embed_battle`, 0.14 ms of `_invert_choice` and a
~9 MB rust child spawn. A B=1 CPU decision measured **26.3 ms eager → 4.1 ms compiled (6.4×)**;
the extractor alone is 21.5 → 3.3 ms, i.e. ~82% of it.

Three consequences, all of them corrections to the banked ~162 ms/label cost model:

* **A warm/persistent search-driver session buys ~0.2% here.** That model was built on the
  *materializer* path (one-ply labels, prefix replayed per arm), where driver spawns dominate.
  This producer already replays each record ONCE (`scan_record` takes the `chunks=` from the
  single `replay_battle`) and its labels are ROLLOUTS-TO-END, which spawn no driver at all —
  they play a live bridge battle whose cost is policy forwards.
* **Prefix sharing buys ~3%.** The R arms do each replay the recorded prefix, but a scripted
  prefix decision costs 0.4 ms against the 4-26 ms of every live one, and the arms diverge
  before most of the battle. Cloning the mid-battle state would have to clone two poke-env
  players' trackers as well — a large change for a rounding error.
* **The lever that IS big is `--compile-extractor`** (now default ON, 6.4× on every rollout
  decision; ~40 s once per PROCESS — a later checkpoint refresh reuses the graph in ~1.1 s).
  `--rollout-concurrency` measured a WASH and defaults to 1: every forward *and* every protocol
  parse runs on poke-env's single `POKE_LOOP` thread, so overlapping arms finds almost no idle
  to fill.

Measured end to end, a REAL one-cycle run over the same 6 records against the same checkpoint:
**198.5 s → 99.4 s of cycle wall**, i.e. **10.8 → 3.2 s per label** once the one-time snapshot
load + compile is taken out; interleaved arm-by-arm on the same decisions (the load-fair form)
**8.09 → 1.81 s of rollout wall per label, 4.5×**. The eager arm reproduces the live producer's own
~8.2 s/label. The labels are unchanged — same key set, same decisions, same ranking; the only
difference anywhere is `priority.win_prob` in the 6th decimal (Inductor arithmetic, max|Δ| ~5e-7).

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
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence

import numpy as np

from agents.observation.constants import MAX_TURNS
from agents.training.cf_producer_labels import (LABELS_DIRNAME,  # noqa: F401 (re-export)
                                                OPPONENT_LABEL, label_row, write_label_batch)
from agents.training.cf_producer_sampler import (MIN_LABELABLE_TURN,  # noqa: F401 (re-export)
                                                 PRIORITY_WEIGHTS, SAMPLER_VERSION,
                                                 critic_surprise, is_move_round,
                                                 normalized_entropy, priority_score)
from agents.training.cf_producer_snapshot import (Snapshot,  # noqa: F401 (re-export)
                                                  load_snapshot, resolve_latest_checkpoint,
                                                  step_from_checkpoint_name)
from agents.training.cf_q_labels import (
    Q_SWEEP_VERSION, assert_paired_dice, q_arm_seeds, q_labels_block, q_provenance,
    recorded_arm_is_reusable, select_q_actions)
from agents.training.obs_materializer import RecordDecision, scan_record
from utils.bridge.counterfactual import replay_counterfactual as _run_one
from utils.bridge.reconstruction import RECON_SUFFIX, ReconstructionRecord, replay_battle
from utils.contention import describe_contention

STATE_FILENAME = "cf_producer_state.json"
RECORDS_DIRNAME = "cf_records"

#: How many processed-record names the state file remembers. The ring's default cap is 512 and
#: files age out of it, so 4096 is ~8 ring turnovers of headroom at a few hundred KB of JSON.
DEFAULT_KEEP_PROCESSED = 4096


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
    #: Rollouts that ended at the stall-forfeit TURN CAP and were therefore scored 0.5 rather than
    #: by play (`gen3_cf_draw_at_cap_v1`). It is the operator's only view of how much of the label
    #: corpus is draw-at-cap, and it must never go invisible: before the fix these were scored a
    #: hard 0 (p1's forfeit is always processed first and the trainee is always p1), which biased
    #: stall-shaped positions DOWNWARD with nothing anywhere saying so.
    rollouts_capped: int = 0
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
    #: `gen3_cf_q_labels_v1` — the PER-ACTION sweep's cost meter, and it is a meter rather than a
    #: statistic because the sweep multiplies a label's cost by its ARM COUNT. `q_arms_rolled /
    #: q_rows` IS the measured ~n_legal multiplier an operator sizing this producer needs, and
    #: `q_arms_reused` is how many of those arms the recorded-action identity bought for free.
    q_rows: int = 0
    q_entries_total: int = 0
    q_arms_rolled: int = 0
    q_arms_reused: int = 0
    q_rollouts_total: int = 0
    q_rollouts_capped: int = 0
    q_wall_seconds: float = 0.0
    #: Per-DECISION q failures, kept apart from `skip_reasons` because that one counts RECORDS: a
    #: swept decision that lost an arm has not cost the record its label, and folding the two would
    #: make `records_skipped` unreadable.
    q_skip_reasons: dict = dataclasses.field(default_factory=dict)
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

    def note_q_skip(self, reason: str) -> None:
        """A per-DECISION q-sweep loss. Never touches ``records_skipped`` — see the field."""
        self.q_skip_reasons[reason] = self.q_skip_reasons.get(reason, 0) + 1

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
# The producer
# ---------------------------------------------------------------------------

def _outcome_scalar(record: ReconstructionRecord, side: str, outcome: dict) -> float:
    winner = outcome.get("winner")
    if not winner:
        return 0.5                      # a tie / the turn cap — maximally uninformative
    return 1.0 if winner == record.username(side) else 0.0


def rollout_outcome_score(res: dict) -> float:
    """One ROLLOUT's contribution to a tight-MC P(win) label, in [0, 1].

    ``gen3_cf_draw_at_cap_v1``. Win 1.0, loss 0.0, **draw 0.5 — and a line that ended at the
    stall-forfeit turn cap IS a draw** whatever ``outcome`` says, because at the cap both sides
    forfeit and the winner is decided by which ``FORCELOSE`` the sim processes first. The full
    mechanism (and why it is a systematic 0 rather than a coin flip on a training record) is in
    :meth:`CfProducer._play_arm`.

    ``unfinished`` — the battle never terminated — is a transport pathology rather than a result,
    and it takes the same 0.5 as a draw for want of anything better. It is near-unreachable here
    (``run_local_battles`` returns only when the battle completes, and anything that goes wrong on
    the way RAISES and is caught as a failed arm), but the OLD code scored it a hard 0 alongside
    the tie, and a pathology must not read as a confident loss.

    Deliberately NOT the same function as :func:`_outcome_scalar`, which reads a RECORDED battle's
    referee-view outcome dict off the replay driver. This one reads a LIVE
    :func:`replay_counterfactual` result, which is the only place ``capped`` exists.
    """
    if res.get("capped"):
        return 0.5
    return {"win": 1.0, "loss": 0.0}.get(res.get("outcome"), 0.5)


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
    ``record.commands``.

    ⚠️ **On RUST-written records this exclusion was INERT until 2026-08-24** — the rust
    `sim_bridge` pushed `commands` only in `handle_choose`, so a forfeited battle's record carried
    no `forcelose` entry and this scan returned True on exactly the class it exists to exclude
    (`anchors_skipped_unanchorable` read 0 on the live run for that reason, not for want of
    forfeits). Fixed at the WRITER — `handle_forcelose` now records the entry where node always
    did — so this predicate is impl-agnostic again. Gate:
    `bridge_impl_parity_test::test_a_forfeited_battles_record_logs_the_forcelose_command`, over
    both impls. Records written before that fix are frozen wrong and read as anchorable.

    `install_scripted_prefix` builds each side's script as
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

    The ROLLOUT path inherits the same asymmetry — a label's rollouts play both sides with
    `RLPlayer`s that both stall-forfeit, so a rollout reaching the 250-turn cap has its winner
    decided by forfeit ordering — and that is **FIXED** (`gen3_cf_draw_at_cap_v1`): a capped
    rollout is scored 0.5, never by the ordering. See :meth:`CfProducer._play_arm`. (An older note
    here guessed the bias was UPWARD; it was measured DOWNWARD — p1's forfeit always lands first
    and the trainee is always p1.) The anchor's own exclusion below stands regardless: a capped
    ANCHOR would still have to reproduce a recorded winner it cannot reproduce.
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
        self._said_capped = False
        self._said_no_return_path = False
        self._said_q_sweep = False
        # gen3_cf_q_labels_v1: the PER-ACTION stream. Resolved ONCE here, with `getattr` defaults,
        # so a caller holding an older args namespace (a test double, a saved argv) keeps the OFF
        # path rather than raising — and OFF is byte-identical output.
        self._q_on = bool(getattr(args, "q_labels", False))
        self._q_top_n = max(0, int(getattr(args, "q_top_n", 1) or 0))
        #: R per sibling arm. 0 means "follow --rollouts", which is also the setting that makes the
        #: recorded action's arm FREE (`recorded_arm_is_reusable`) — a default worth having.
        self._q_rollouts = int(getattr(args, "q_rollouts", 0) or 0) or int(args.rollouts)
        self._q_max_actions = max(0, int(getattr(args, "q_max_actions", 0) or 0))
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
                               mappings=self.snapshot.mappings, impl=self.args.impl,
                               # The per-action sweep needs every legal action's choice STRING, and
                               # this replay is already standing at each decision. Asking here costs
                               # ~n_legal pure mapper calls per decision; asking later costs a
                               # prefix replay per labelled decision.
                               capture_choices=self._q_on)

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
        for rank, (score, s, e, wp, d) in enumerate(scored[: self.args.top_n]):
            if self._throttled():
                self._log(f"throttle: {self._rate_per_hour():.0f} labels in the last hour "
                          f"(--max-labels-per-hour {self.args.max_labels_per_hour}) — "
                          f"stopping this record early")
                break
            tag = os.path.basename(path)
            wins, n, n_capped, returns, base_seeds = self._rollout(record, side, d, tag=tag)
            if n == 0:
                self.state.note_skip("rollouts_all_failed")
                continue
            self.label_times.append(time.time())
            self.state.rollouts_total += n
            self.state.rollouts_capped += n_capped
            # `gen3_cf_q_labels_v1`: the per-action sweep, on the TOP `--q-top-n` of this record's
            # labelled decisions. It is ranked-prefix rather than a sampling rate because the
            # ranking is the producer's declared priority — spending the multiplied budget on the
            # decisions the sampler already judged most informative, not on a random subset of them.
            q_labels = q_sweep = None
            if self._q_on and rank < self._q_top_n:
                q_labels, q_sweep = self._q_labels(
                    record, side, d, tag=tag,
                    base=(wins, n, n_capped, base_seeds))
            rows.append(label_row(
                record_path=path, decision=d, wins=wins, n=n, n_capped=n_capped,
                step=self.snapshot.step, surprise=s, entropy=e,
                score=score, win_prob=wp,
                # HEAD B's label is the RECORDED battle's realized outcome — already computed above
                # for the critic-surprise term, so it costs nothing. It is the SAME quantity the
                # on-policy BCE eats, on the states the sampler selected: that identity is what
                # makes B-A a read of COVERAGE alone.
                outcome_label=outcome,
                mc_return=(float(np.mean(returns)) if returns else None),
                mc_return_n=len(returns),
                reward_sha1=self._reward_sha1, reward_composition=self._reward_composition,
                q_labels=q_labels, q_sweep=q_sweep))
        return rows

    def _q_labels(self, record: ReconstructionRecord, side: str, d: RecordDecision, *,
                  tag: str, base: "tuple") -> "tuple[List[dict], dict]":
        """The PER-ACTION sweep for one already-labelled decision (`gen3_cf_q_labels_v1`).

        ``base`` is the per-state label's own ``(wins, n, n_capped, seeds)`` — passed in rather
        than recomputed, because it IS the recorded action's q-label whenever ``--q-rollouts``
        matches ``--rollouts`` (:func:`cf_q_labels.recorded_arm_is_reusable`), and because its
        OBSERVED seed list is what the pairing check adjudicates against. Re-deriving the seeds
        here would make the check prove only that one function is deterministic.

        Returns ``(entries, provenance)`` — the wire list and the additive audit block.

        **THIS IS WHERE THE COST MULTIPLIES.** The per-state path pays R rollouts per label; this
        pays R per LEGAL ACTION, so a decision offering 9 of them costs ~9x, minus the one arm the
        reuse above buys for free. Every arm is metered into the state file, and
        ``q_arms_rolled / q_rows`` is the measured multiplier rather than an estimate of it.
        """
        t0 = time.perf_counter()
        base_wins, base_n, base_capped, base_seeds = base
        choices = d.choices or {}
        if not choices:
            # A record scanned WITHOUT `capture_choices` — the sweep cannot name a sibling action's
            # choice string, and inventing one is a second mapping implementation. Counted, never
            # silently degraded into a one-entry (on-policy) block.
            self.state.note_q_skip("no_choice_map")
            return [], q_provenance(actions=(), rollouts=0, capped=0, reused_recorded=False,
                                    max_actions=self._q_max_actions, wall_seconds=0.0)
        R = self._q_rollouts
        reuse = recorded_arm_is_reusable(q_rollouts=R, rollouts=int(self.args.rollouts))
        actions = select_q_actions(
            d.mask, d.action, max_actions=self._q_max_actions,
            tag=tag, decision_index=int(d.index), producer_seed=int(self.args.seed))
        seeds = q_arm_seeds(tag=tag, decision_index=int(d.index),
                            producer_seed=int(self.args.seed), n=R)

        results: "List[tuple]" = []
        per_action_seeds: "dict[int, tuple]" = {}
        capped = 0
        for a in actions:
            if a == int(d.action) and reuse:
                results.append((a, float(base_wins), int(base_n)))
                # The seeds the base arm REPORTED, not the ones this method just derived — that is
                # what makes the pairing assertion below a measurement.
                per_action_seeds[a] = tuple(base_seeds)
                capped += int(base_capped)
                self.state.q_arms_reused += 1
                continue
            choice = choices.get(a)
            if choice is None:
                self.state.note_q_skip("unmapped_action")
                continue
            if self._throttled():
                # The throttle counts per-action labels too (see `--max-labels-per-hour`), so a
                # sweep can exhaust it mid-decision. Stopping here ships a PARTIAL sweep, which is
                # honest: every entry in it is a real measurement and the consumer masks the rest.
                self.state.note_q_skip("throttled")
                break
            wins, n, arm_capped, _returns, used = self._rollout(
                record, side, d, tag=tag, substitute_choice=choice,
                rollouts=R, seeds=seeds, with_return=False)
            self.state.q_arms_rolled += 1
            self.state.q_rollouts_total += n
            self.state.q_rollouts_capped += arm_capped
            if n == 0:
                self.state.note_q_skip("arm_all_failed")
                continue
            per_action_seeds[a] = tuple(used)
            capped += arm_capped
            results.append((a, wins, n))
            self.label_times.append(time.time())

        # THE SEAM. Every sibling arm must have run on one seed list; see `cf_q_labels`.
        assert_paired_dice(per_action_seeds)

        entries = q_labels_block(results)
        wall = time.perf_counter() - t0
        self.state.q_wall_seconds += wall
        self.state.q_entries_total += len(entries)
        if entries:
            self.state.q_rows += 1
        if not self._said_q_sweep:
            self._said_q_sweep = True
            self._log(f"q-sweep ON ({Q_SWEEP_VERSION}): first decision covered {len(entries)} of "
                      f"{int(np.asarray(d.mask).sum())} legal actions at R={R} in {wall:.1f}s"
                      f"{' (recorded arm reused free)' if reuse else ''}. Cost multiplies by the "
                      f"ARM COUNT — watch `q_arms_rolled` on the heartbeat.")
        return entries, q_provenance(
            actions=actions, rollouts=R, capped=capped, reused_recorded=reuse,
            max_actions=self._q_max_actions, wall_seconds=wall)

    def _rollout(self, record: ReconstructionRecord, side: str, d: RecordDecision,
                 *, tag: str, substitute_choice: "Optional[str]" = None,
                 rollouts: "Optional[int]" = None, seeds: "Optional[Sequence[str]]" = None,
                 with_return: bool = True) -> "tuple[float, int, int, List[float], List[str]]":
        """R continuations from ``d``: the RECORDED action, then both sides live on fresh dice.

        ``substitute_choice`` overrides which action is played at the divergence turn — ``None``
        (the default, and the whole per-state path) plays the RECORDED one. `gen3_cf_q_labels_v1`'s
        per-action sweep passes a SIBLING action's choice string here, which is the only difference
        between the row's own label and one of its q-labels. ``rollouts`` overrides R
        (``--q-rollouts``), ``with_return=False`` skips the shaped-return recorder (a q-label
        carries no ``mc_return``, so measuring one would be work nothing reads), and ``seeds``
        supplies the post-divergence dice explicitly.

        ⚠️ **``seeds`` is how sibling actions are PAIRED, and the returned seed list is how that is
        CHECKED.** The default derivation (:func:`cf_q_labels.q_arm_seeds`) has no action term, so
        passing nothing already pairs — but a sweep must not depend on remembering that, so every
        call reports the dice it actually used and :func:`cf_q_labels.assert_paired_dice` adjudicates
        at the seam. See that module for why an unpaired sweep at R=8 is noise.

        Returns ``(wins, n, n_capped, returns, seeds_used)``. ``wins`` is a FRACTIONAL success
        total — a draw scores 0.5 and a turn-cap forfeit is a draw (`gen3_cf_draw_at_cap_v1`, see
        :meth:`_play_arm`) — and ``n_capped`` is how many of the ``n`` finished arms hit the cap.
        ``returns`` is the per-rollout DISCOUNTED SHAPED RETURN from
        the labelled decision (`gen3_cf_twin_heads_v1`, the shadow critic's stream) and is EMPTY
        when `--no-mc-return` is set or when a rollout produced no measurable return — never
        zero-padded, because a zero return is the middle of this reward's range and would read as a
        genuinely neutral game rather than as a missing measurement.

        The tracking is armed at the DIVERGENCE decision through the scripted-prefix hook, not at
        the first live one: our turn-T move is scripted (it IS the substitute), so a live-only hook
        would miss ``r_T`` and shift the whole return by one turn against the state it labels.

        **THIS IS THE WHOLE COST OF THE PRODUCER** — 93% of the wall, measured (§ *Where the time
        goes*) — so it is also the only place worth parallelising. Every arm is INDEPENDENT (its own
        pair of players, its own bridge child, its own post-divergence dice) and the aggregate is a
        SUM over arms, so running them together changes no label semantics: `wins`/`n` are
        order-free counts and `returns` is reduced by a mean. ``--rollout-concurrency`` picks how
        many overlap; 1 is the byte-identical sequential path.
        """
        rec = dataclasses.replace(record, trainee_username=record.username(side))
        seed = record.start_options().get("seed")
        n_arms = int(self.args.rollouts if rollouts is None else rollouts)
        # The DEFAULT derivation is the per-state path's, unchanged: `q_arm_salt` is verbatim the
        # salt this line has always used, so a run without `--q-labels` draws the same dice it
        # always did.
        seeds = list(seeds) if seeds is not None else q_arm_seeds(
            tag=tag, decision_index=d.index, producer_seed=int(self.args.seed), n=n_arms)
        choice = d.choice if substitute_choice is None else substitute_choice
        # Built up front, on THIS thread: `make_player` hands out account names off a shared
        # counter and the mc_return hook patches the player, so neither belongs in a worker.
        arms = [self._prepare_arm(rec, side, d, ps, with_return=with_return) for ps in seeds]

        conc = max(1, min(int(getattr(self.args, "rollout_concurrency", 1) or 1), len(arms)))
        if conc == 1:
            results = [self._play_arm(a, rec, side, d, seed, tag, choice) for a in arms]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=conc, thread_name_prefix="cf-roll") as ex:
                results = list(ex.map(
                    lambda a: self._play_arm(a, rec, side, d, seed, tag, choice), arms))

        wins = 0.0
        n = n_capped = 0
        returns: "List[float]" = []
        for ok, outcome_score, capped, g in results:
            if not ok:
                continue
            n += 1
            wins += float(outcome_score)
            n_capped += int(capped)
            if g is not None:
                returns.append(g)
        return wins, n, n_capped, returns, seeds

    def _prepare_arm(self, rec: ReconstructionRecord, side: str, d: RecordDecision, ps,
                     *, with_return: bool = True) -> dict:
        """One rollout arm's players + its mc_return hook. Main-thread only — see :meth:`_rollout`.

        ``with_return=False`` builds the arm with no shaped-return recorder at all: a per-action
        q-label has nowhere to carry an ``mc_return`` (the wire object is action/label/n_rollouts),
        so measuring one would be work no consumer reads."""
        from agents.training.cf_mc_return import attach_return_recording

        trainee = self.snapshot.make_player(rec, side, role="T")
        opponent = self.snapshot.make_player(rec, _other(side), role="O")
        recorder = None
        hook = None
        if self._mc_return_on and with_return:
            recorder = attach_return_recording(
                trainee, reward_fn_factory=self._reward_factory)
            if recorder is None and not self._said_no_return_path:
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
                if mask is None:                           # a non-Gen3 player; nothing to record
                    return
                _f["done"] = True
                _rec.arm_at_next()
                _rec.note(battle, _a, mask)
        return {"trainee": trainee, "opponent": opponent, "recorder": recorder,
                "hook": hook, "post_t_seed": ps}

    def _play_arm(self, arm: dict, rec: ReconstructionRecord, side: str, d: RecordDecision,
                  seed, tag: str,
                  substitute_choice: "Optional[str]" = None,
                  ) -> "tuple[bool, float, bool, Optional[float]]":
        """Play ONE prepared arm to a terminal. ``(ok, outcome_score, capped, shaped_return|None)``.

        Runs on a worker thread when ``--rollout-concurrency`` > 1. It touches nothing shared: the
        players, the recorder and the bridge child all belong to this arm, and the only producer
        state it writes is a once-flag whose worst case is the same warning printed twice. A
        rollout that RAISES is reported and returns ``ok=False`` — the label is then computed over
        the arms that finished, exactly as the sequential path always did.

        🚨 **A ROLLOUT THAT REACHES THE 250-TURN CAP IS A DRAW, and scoring it as a win or a loss
        was a label-quality defect** (`gen3_cf_draw_at_cap_v1`, fixed 2026-08-23). Both sides of a
        rollout are `RLPlayer`s that stall-forfeit at ``MAX_TURNS``, so at the cap BOTH forfeit and
        the recorded winner is decided by which ``FORCELOSE`` the sim processes FIRST. That is not
        a fact about the position, and — measured over 16 capped lines across `node` and `rust` —
        it is not even a coin flip: **p1's forfeit is always processed first, so p1 always loses**,
        and `_trainee_side` seats a training record's trainee on p1 ALWAYS. Every capped rollout
        therefore used to score a hard 0, biasing tight-MC P(win) labels DOWNWARD on exactly the
        stall-shaped positions where the cap is reachable.

        So a capped line scores **0.5**, the same as a genuine tie (which was also scored 0 —
        ``outcome == "win"`` is False for a tie — and is likewise a draw). Both possible forfeit
        orderings now map to the same label, which is what makes the ordering unobservable rather
        than merely unlikely. The count rides out to the row as `n_capped`. The scoring itself is
        :func:`rollout_outcome_score`.
        """
        try:
            res = _run_one(
                rec,
                trainee=arm["trainee"],
                opponent=arm["opponent"],
                divergence_turn=int(d.turn),
                substitute_choice=(d.choice if substitute_choice is None else substitute_choice),
                seed=seed, post_t_seed=arm["post_t_seed"], impl=self.args.impl,
                trainee_decision_hook=arm["hook"])
        except Exception as exc:                                        # noqa: BLE001
            self._log(f"rollout failed ({tag} inv {d.index}): "
                      f"{type(exc).__name__}: {str(exc)[:200]}")
            return False, 0.0, False, None
        g = None
        recorder = arm["recorder"]
        if recorder is not None:
            g = recorder.value(self._gamma)
            if g is None and not self._warned_no_return:
                self._warned_no_return = True
                self._log("⚠️  a rollout produced NO measurable shaped return (the divergence "
                          "decision was never reached, or the battle ended in the prefix) — "
                          "that rollout contributes no mc_return. Said once; watch "
                          "cf/mc_return_coverage on the trainer side.")
        capped = bool(res.get("capped"))
        if capped and not self._said_capped:
            self._said_capped = True
            self._log(f"⚠️  a rollout reached the {MAX_TURNS}-turn stall-forfeit CAP ({tag} inv "
                      f"{d.index}) — scored 0.5 (a draw at cap), not by which side's forfeit "
                      f"landed first. Counted per label as `n_capped` and in the state file's "
                      f"`rollouts_capped`. Said once.")
        return True, rollout_outcome_score(res), capped, g

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
        # The sweep's two derived readings, named here so the heartbeat f-string stays legible:
        # the measured ~n_legal MULTIPLIER, and the per-entry wall it implies.
        q_mult = ((self.state.q_arms_rolled + self.state.q_arms_reused)
                  / max(1, self.state.q_rows))
        q_per_entry = self.state.q_wall_seconds / max(1.0, float(self.state.q_entries_total))
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
              # Same rule as `vanished`: a number, not an error. Draw-at-cap rollouts are a real
              # part of the label mix on stall-shaped positions, and a reader stratifying the
              # corpus needs to see the rate here rather than reconstruct it from the rows.
              + (f"capped {self.state.rollouts_capped}/{self.state.rollouts_total} | "
                 if self.state.rollouts_capped else "")
              # The per-action sweep's COST, on the same line as the labels it multiplies. `x` is
              # the measured ~n_legal multiplier — the number that sizes this producer — and it is
              # reported rather than assumed because it depends on how many actions are legal at
              # the decisions the sampler happens to pick.
              + (f"q {self.state.q_entries_total} entries / {self.state.q_rows} rows "
                 f"({self.state.q_arms_rolled} arms rolled, {self.state.q_arms_reused} free, "
                 f"{q_mult:.1f}x, {q_per_entry:.1f}s/entry) | "
                 if self._q_on else "") +
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
                        "trainer; this is the knob that keeps it a sidecar. Under --q-labels "
                        "every per-action ARM IT ACTUALLY ROLLS counts too (the reused recorded arm "
                        "costs nothing, so it does not), which keeps the cap a COST cap instead of "
                        "silently letting a sweep multiply the box load by its arm count")
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
    p.add_argument("--compile-extractor", "--compile_extractor", dest="compile_extractor",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the snapshot's extractor. DEFAULT ON, because it is THE "
                        "lever here: 93%% of this process's wall is the live rollout policy "
                        "forward, and compiling takes a B=1 CPU decision 26.3 -> 4.1 ms (6.4x, "
                        "measured 2026-08-23 on the ai_v9_29_rev1_0823 checkpoint). It costs ~20 s "
                        "ONCE per checkpoint refresh, amortized over the hundreds of rollouts that "
                        "follow. --no-compile-extractor is the fallback when the compile is the "
                        "suspect; a compile FAILURE already degrades to eager on its own")
    p.add_argument("--rollout-concurrency", "--rollout_concurrency", dest="rollout_concurrency",
                   type=int, default=1,
                   help="how many of a decision's --rollouts arms play at once. DEFAULT 1 (the "
                        "sequential path) because it MEASURED A WASH: 10 paired label-arms, "
                        "conc=1 mean 3.86 s vs conc=8 mean 4.17 s, no consistent sign. Every "
                        "policy forward runs on poke-env's single POKE_LOOP thread — and so does "
                        "the protocol parse — so overlapping arms finds almost no idle to fill. "
                        "Kept as a knob for a box where the sim, not the forward, is the wait")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="intra-op torch threads (default 1). This process shares the box with a "
                        "trainer; 1 is the sidecar-shaped choice and the compile is where the "
                        "speed comes from")
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
    # -- gen3_cf_q_labels_v1: the PER-ACTION stream -----------------------------------
    p.add_argument("--q-labels", "--q_labels", dest="q_labels",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="also emit the PER-ACTION counterfactual stream `q_labels` (plus "
                        "`taken_action`) on each swept row — the labels the v107 Q win-prob head "
                        "(--q-winprob-mode/--q-winprob-coef) is trained on. Additive-optional at "
                        "schema v1, so an older trainer reads the new rows unchanged. DEFAULT OFF "
                        "and byte-identical when off. 🚨 ON, a labelled decision costs R rollouts "
                        "PER LEGAL ACTION instead of R total — roughly 9x at a typical move round, "
                        "minus one arm the recorded action gets free. Size it with --q-top-n / "
                        "--q-rollouts / --q-max-actions and watch the `q ...` heartbeat field")
    p.add_argument("--q-top-n", "--q_top_n", dest="q_top_n", type=int, default=1,
                   help="how many of a record's --top-n labelled decisions get the per-action "
                        "sweep (default 1). THE budget knob: total cost scales with this times the "
                        "arm count. The prefix is taken in the sampler's own priority order, so "
                        "the multiplied budget lands on the decisions already judged most "
                        "informative rather than on a random subset of them")
    p.add_argument("--q-rollouts", "--q_rollouts", dest="q_rollouts", type=int, default=0,
                   help="R per SIBLING arm (default 0 = follow --rollouts). Matching --rollouts is "
                        "worth a default: the recorded action's arm is then the row's own label "
                        "measured on the same dice, so it is reused free and `q_labels[recorded] "
                        "== label` exactly. A smaller value buys arms at lower per-arm evidence — "
                        "usually the better trade, since the sweep's product is a RANKING and the "
                        "shared dice already cancel in the differences")
    p.add_argument("--q-max-actions", "--q_max_actions", dest="q_max_actions", type=int, default=0,
                   help="cap the arms per swept decision (default 0 = every legal action). The "
                        "recorded action is always kept; the rest are ordered by a deterministic "
                        "decision-keyed SHUFFLE, never by policy probability (that would rebuild "
                        "the on-policy starvation this stream exists to escape) and never by "
                        "action index (a prefix of [switch x6, move x4, struggle] is a systematic "
                        "preference for switching). Declared as `cf_q_sweep_v1` on every row")
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
        torch.set_num_threads(max(1, int(getattr(args, "torch_threads", 1) or 1)))
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
    prod._log(f"throughput: compile-extractor={'ON' if args.compile_extractor else 'OFF'}, "
              f"rollout-concurrency={args.rollout_concurrency}, torch-threads={args.torch_threads}"
              f" — 93% of this process's wall is the live rollout forward (measured), so those "
              f"three knobs are the whole cost model")
    prod._log("OPPONENT ECOLOGY: rollouts play the CURRENT snapshot on BOTH sides, stochastic — "
              "a training record names no opponent. Every label says opponent=self_current.")
    if prod._q_on:
        prod._log(
            f"PER-ACTION labels ON ({Q_SWEEP_VERSION}): top-{prod._q_top_n} decision(s) per record "
            f"swept at R={prod._q_rollouts}"
            f"{f', max {prod._q_max_actions} arms' if prod._q_max_actions else ', all legal arms'}"
            f". Sibling arms share ONE dice list (paired, asserted at the seam); the recorded "
            f"action's arm is reused free when --q-rollouts == --rollouts. Expect ~n_legal times "
            f"the wall per swept decision.")
    else:
        prod._log("PER-ACTION labels OFF — rows carry no `q_labels`, so a trainer with "
                  "--q-winprob-coef > 0 would train that head on nothing "
                  "(cf/q_label_coverage 0.0). --q-labels turns the stream on.")

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
