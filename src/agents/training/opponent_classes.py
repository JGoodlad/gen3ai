"""WHICH KIND of opponent an episode faces (gen3_opp_class_v1) and the stable opponents' share of the
challenge bucket — the opponent-mix constants, declared once and free of any env.

Deletion pass U2: these lived on the Python env core's ``MaskableAgentWrapper`` (``wrappers.py``), so
the Rust env core's opponent plan, the self-play callback, the eval-subprocess parser and the value
sidecar reader all imported the Python env core to read four integers and a float. They live here now;
``wrappers`` reads them from here until it is deleted. ``agents.model.opp_intent.OPP_CLASS_NAMES`` (the
model package's plain table) is pinned to agree with ``OPP_CLASS_NAMES`` below.
"""

#: Heuristic / random floor bots — near-deterministic or unpredictable.
OPP_CLASS_BOT = 0
#: Frozen selves — the distribution that actually matters.
OPP_CLASS_POOL = 1
#: Cross-run stable opponents.
OPP_CLASS_STABLE = 2
#: The exploiter's target.
OPP_CLASS_EXPLOITER = 3
#: Width of any per-class array keyed on the four above.
N_OPP_CLASSES = 4
#: code -> the suffix its stratified metrics carry.
OPP_CLASS_NAMES = {OPP_CLASS_BOT: "bot", OPP_CLASS_POOL: "pool", OPP_CLASS_STABLE: "stable",
                   OPP_CLASS_EXPLOITER: "exploiter"}

#: Stable cross-run opponents are a CAPPED minority of the self-play (challenge) bucket — the pool
#: keeps the bulk, so no single fixed opponent can dominate training. Multiple un-mastered stable
#: opponents SHARE this slice (so the total stable share stays <= this regardless of count). Mastered
#: ones leave the challenge bucket entirely for the floor.
STABLE_CHALLENGE_SHARE = 0.20
