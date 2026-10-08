"""The FAINT-CAUSE vocabulary — the labels a KO's cause is encoded with.

Lifted out of ``turn_view.py`` when the Python ``TurnView`` fold was deleted (T27 P6 slice 6d-2): the
fold is the Rust core's now (``src/rust_sim/src/trackers/``), and these tuples are what the MODEL and the
archive decoder still read. ``rust_core_obs_layout_test.py`` holds ``FAINT_CAUSE_VOCAB_LIVE`` equal to the
Rust encoder's ``layout.rs::FAINT_CAUSE_VOCAB``.

* ``FAINT_CAUSE_VOCAB`` / ``FAINT_CAUSE_DIM`` (8) — the ARCHIVE vocabulary: it sizes the frozen
  ``TurnDelta`` lag frame (``turn_delta_encoder``) the prober still decodes from archived runs, so it
  cannot grow.
* ``FAINT_CAUSE_VOCAB_LIVE`` / ``FAINT_CAUSE_DIM_LIVE`` (10) — the event window's (gen3_event_record_v2,
  E12): the archive's eight, then the two a faint with no lethal damage line really has. The event row's
  FAINT_CAUSE column is 1 + the index (0 = not a FAINT row); ``team_transformer.EventSeats`` sizes its
  embedding table from ``FAINT_CAUSE_DIM_LIVE``.
"""
from __future__ import annotations

FAINT_CAUSE_VOCAB: tuple = (
    "attack",    # KO'd by a direct move (no [from] clause on the lethal damage)
    "hazard",    # Entry hazard (Spikes / Stealth Rock)
    "weather",   # Residual weather damage (Sandstorm / Hail)
    "status",    # Residual status damage (Burn / Poison / Toxic)
    "recoil",    # Recoil damage
    "selfko",    # User of Explosion / Selfdestruct kills themselves
    "leechseed",  # Leech Seed residual
    "other",     # Any unclassified [from] source
)
FAINT_CAUSE_DIM: int = len(FAINT_CAUSE_VOCAB)  # 8

FAINT_CAUSE_VOCAB_LIVE: tuple = FAINT_CAUSE_VOCAB + ("destinybond", "perishsong")
FAINT_CAUSE_DIM_LIVE: int = len(FAINT_CAUSE_VOCAB_LIVE)  # 10
