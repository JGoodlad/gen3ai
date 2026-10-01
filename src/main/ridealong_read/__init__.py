"""The RIDE-ALONG HEADS' offline reader (`gen3_ridealong_read_v1`).

Reads a checkpoint's detached ride-along heads (``agents.model.ridealong_heads``: the V ensemble, RND
novelty, A and B) on the fixed M5 Lane S turn bank and the Lane S ground truth — forward passes only,
CPU, no games:

    python -m main.ridealong_read --checkpoint models/<run>/<ckpt>.zip [--checkpoint ...] --out <dir>

- :mod:`.reader`     the forward (V, logits, the heads' readout through ``RideAlongBatch.detached``),
                     heads provenance (``trained`` from the checkpoint, or ``fresh-untrained``
                     baseline-spec heads attached here), one JSON per checkpoint.
- :mod:`.meters`     (i) disagreement / novelty vs V's actual error by opponent class and phase;
                     (ii) A vs the truth's per-action values (policy logits as the reference row);
                     (iii) A's member spread on the near-best moves the policy starves;
                     (iv) disagreement / novelty vs |V − V_truth| on the truth turns.
- :mod:`.rnd_choice` (v) the RND INPUT-CHOICE measurement: obs-RND vs feature-RND trained offline on a
                     battle-level split — unseen-team and exploiter novelty, and the representation-
                     drift confound (feature-RND trained on checkpoint A, scored through a later B).
- :mod:`.rnd_states` the STATE-level RND reads (``python -m main.ridealong_read.rnd_states``): a
                     within-cell 70 / 30 battle split, visitation counts of a state key decoded from
                     the observation, off-distribution classes, and the one-ply successors of the
                     moves the policy starves (``rnd_states_2026-09-30/``).

The first committed read (a plumbing smoke on UNTRAINED heads):
``designs/research_state/measurements/ridealong_baseline/smoke_2026-09-30/``.
"""
