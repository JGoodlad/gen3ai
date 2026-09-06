"""cf_producer_labels — the label ROW schema and the batch file it is written into.

The row is a CONTRACT with a training-side consumer (`cf_label_buffer`) that knows nothing about
this producer beyond a fixed key set, so the schema and the writer that lays it down belong
together and apart from the loop that fills them. Both rules the writer carries are correctness
ones rather than style: one NEW file per batch (never a rewrite, because the buffer keys its byte
offsets on ``(name, inode)``) and a tmp-then-rename, so a half-written batch is never visible.

⚠️ **THE ECOLOGY DECISION lives here too, as ``OPPONENT_LABEL``.** A training record carries no
opponent identity — the tap's ``__RECON__`` frame holds the seed, both packed teams and the
committed choices, and nothing that says *which policy* sat on the other side. So every row this
producer writes names ``self_current`` and never a bot name it cannot verify, which is what lets a
reader tell a producer label from a `cf_audit` label, whose opponent IS identified. The full
reasoning (and the known direction of the bias it accepts) is in `cf_producer`'s module docstring.

Extracted verbatim from ``cf_producer.py`` (2026-09-06, the file-size ratchet's third cut of the
1,000-2,000 band) — it was already its own banner section there. ``cf_producer`` re-imports every
name below, so ``from agents.training.cf_producer import label_row`` still resolves, and the
emitted rows are unchanged: ``cf_producer_test.py``'s extraction-parity golden digests the row and
the batch file itself, captured BEFORE the move and reproduced byte-for-byte after it.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

from agents.training.cf_audit import LabelRow, obs_b64, obs_digest, wilson_ci
from agents.training.cf_producer_sampler import SAMPLER_VERSION
from agents.training.obs_materializer import RecordDecision


#: Every row this producer writes names this as its opponent. See *THE ECOLOGY DECISION*.
OPPONENT_LABEL = "self_current"

LABELS_DIRNAME = "cf_labels"



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


def label_row(*, record_path: str, decision: RecordDecision, wins: float, n: int,
              step: int, surprise: float, entropy: float, score: float,
              win_prob: Optional[float], n_capped: int = 0,
              outcome_label: Optional[float] = None,
              mc_return: Optional[float] = None, mc_return_n: int = 0,
              reward_sha1: str = "", reward_composition: str = "",
              q_labels: "Optional[Sequence[dict]]" = None,
              q_sweep: "Optional[dict]" = None) -> dict:
    """The shared v1 schema, plus additive provenance the buffer ignores by design.

    The extra keys (`sampler_version`, `priority`, `label_regime`, `n_capped`) are NOT schema
    changes — the buffer reads a fixed key set and ignores the rest — but they are the difference
    between a label file you can audit a year later and a bag of numbers.

    ``wins`` is a FLOAT, not a count (`gen3_cf_draw_at_cap_v1`). A draw scores 0.5, and the
    turn-cap forfeit is a draw (see :meth:`CfProducer._play_arm`), so a label over R rollouts is a
    mean of ``{0, 0.5, 1}`` rather than of ``{0, 1}``. ``n_capped`` records how many of the R hit
    the cap, because a label of 0.5 built from 8 draws-at-cap and a label of 0.5 built from 4 wins
    and 4 losses are the same number about different positions — and the reader who wants to
    stratify or caveat cannot re-derive it from the row afterwards. ``wilson_lo``/``wilson_hi``
    keep taking the fractional success total: the Wilson interval is a binomial-proportion
    interval, so with draws in the sample it is an APPROXIMATION (the sample is no longer
    Bernoulli) — it errs narrow, which is why `n_capped` sits beside it.

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

    ``q_labels`` is the PER-ACTION stream (`gen3_cf_q_labels_v1`, ``--q-labels``) — additive-optional
    at schema v1 like the two above, and riding the SAME row for the same reason: the buffer dedups
    on the obs digest, so a second row for one state would collide with the first and one of them
    would vanish. Passing it also writes ``taken_action``, the consumer-facing name for the index
    this row has always carried under the provenance key ``recorded_action``; the WEAK on-policy
    fallback term (``--q-winprob-onpolicy-coef``) reads that name beside ``outcome_label``. The two
    travel together deliberately — the free field arrives with the expensive one rather than
    offering a run the on-policy-only regime this whole stream exists to escape.
    """
    lo, hi = wilson_ci(float(wins), int(n))
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
        # How many of `n_rollouts` ended at the 250-turn stall-forfeit cap and were therefore
        # scored 0.5 rather than by play. Additive-only; the buffer never reads it.
        n_capped=int(n_capped),
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
    if q_labels is not None:
        # Written only when the sweep RAN. An absent key and an empty list mean different things
        # here: absent = this producer does not do per-action labels, `[]` = it swept this decision
        # and every arm failed. Both leave the Q head unsupervised on this row; only the second is
        # a fact about the sweep, and `cf/q_label_coverage` should be able to tell them apart.
        row.update(q_labels=list(q_labels), taken_action=int(decision.action))
        if q_sweep is not None:
            row["q_sweep"] = q_sweep
    return row
