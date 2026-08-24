"""SEARCH-DIVIDEND probe — does a search on top of the trained policy buy strength?

The registered experiment (ledger ``7ae7e79``): three arms x a budget sweep.

| arm | what it searches | the question it answers |
|---|---|---|
| ``base`` | nothing — the policy's own argmax | the control |
| ``honest`` | K pool-consistent DETERMINIZATIONS of the opponent's never-revealed slots | what a real player could get |
| ``oracle`` | the TRUE hidden state (K=1) | the ceiling: what a perfect belief would be worth |

Budgets are a per-decision wall-clock deadline (0.5 / 1 / 3 s, plus a small 8 s batch) and buy
WIDTH in one registered order — alpha-pruned opponent actions, then determinized worlds, then CRN
dice resamples. The ORDER is the three-axis variance measurement's ranking, not a guess.

**Then they buy DEPTH** (the amendment in ``9599426``, minutes after the registration: "fixed
depth" was shorthand for cheap, not a constraint). Width is planned first and unchanged; whatever
the clock has left is spent deepening the top-``m`` actions a ply at a time, and the REALIZED depth
is recorded per decision — so a 0.5 s cell reporting depth 1 beside an 8 s cell reporting depth 2
is the finding, not a configuration. ``--max-depth`` caps it; the budget governs it.

The ``oracle - honest`` gap is the BELIEF's price and the ``honest - base`` gap is the dividend
actually available today; separating them is why the probe has three arms rather than two.

**Two opponent surfaces, and the second is the sensitive one.** Against the scripted roster every
arm saturates near 90%, which is a ceiling a real dividend can hide inside. ``--opponents self``
plays the searched side against the SAME network with search structurally off — a MIRROR whose
no-effect point is exactly 0.50 by construction. Mirror cells are read against that null, with
``--side-swap`` (on by default there) playing every game in both orientations so the team draw
differences out, and they are excluded from the anchored-ELO fit because ``self`` carries no anchor.

Modules
-------
``determinize`` pool-consistent world sampling + THE prefix byte-identity gate ·
``record`` synthesizing a reconstruction record for a battle still in flight ·
``alpha`` the opponent-marginalization candidate set (the alpha-consumer contract) ·
``budget`` the wall-clock deadline + the width allocator ·
``deepen`` the search tree, its max/alpha-weighted backup, and the beam a ply is planned against ·
``search`` the search itself — the first ply, then iterative deepening under the clock ·
``player`` the search-wrapped eval player ·
``battery`` matched-game driver, side-swap pairing, append-only resumable results ·
``summary`` Wilson intervals, the mirror-vs-null read, and anchored-ELO deltas.
"""
