"""THE TEAM-STRENGTH RATING meter (TSR, ``python -m main.team_strength``): how strong is a TEAM, measured with a FIXED
pilot on both sides, so that a set of teams can be called weaker or stronger than another without the pilot's own
familiarity standing in for strength.

* :mod:`~main.team_strength.panel` — team sets, team IDENTITY (the six strict mon sets), the frozen REFERENCE panel.
* :mod:`~main.team_strength.register` — the registration: every input frozen by digest before any game.
* :mod:`~main.team_strength.plan` — which team meets which reference team, in which unit, at which seed.
* :mod:`~main.team_strength.scheduled` — the builder that makes ``main.h2h``'s engine play the PLANNED teams.
* :mod:`~main.team_strength.x26_lane` — PILOT A: one checkpoint on both sides, mirrored pairs, the Rust eval core.
* :mod:`~main.team_strength.kakuna_lane` — PILOT B: Metamon's Kakuna on both sides, the anchors' B-D9 pair cell.
* :mod:`~main.team_strength.fit` — Bradley-Terry on team identity, its bootstrap, set means, cross-pilot agreement.
* :mod:`~main.team_strength.report` — the decision classes and the matched-strength piloting gap.

🚨 A rating is the strength of a team AS ITS PILOT PLAYS IT. A team the pilot trained on is played better than one it
never saw, so a pilot-A rating of a familiar team carries the pilot's familiarity; compare sets of EQUAL familiarity
under one pilot, and read strength across familiarity from a pilot that saw neither (or from both pilots together).
Design and first read: ``designs/research_state/measurements/team_strength_2026-10-18/``.
"""
