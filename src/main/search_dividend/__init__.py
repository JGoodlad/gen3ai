"""SEARCH-DIVIDEND probe — does a depth-1 search on top of the trained policy buy strength?

The registered experiment (ledger ``7ae7e79``): three arms x a budget sweep, at FIXED depth 1.

| arm | what it searches | the question it answers |
|---|---|---|
| ``base`` | nothing — the policy's own argmax | the control |
| ``honest`` | K pool-consistent DETERMINIZATIONS of the opponent's never-revealed slots | what a real player could get |
| ``oracle`` | the TRUE hidden state (K=1) | the ceiling: what a perfect belief would be worth |

Budgets are a per-decision wall-clock deadline (0.5 / 1 / 3 s, plus a small 8 s batch) and buy
WIDTH in one registered order — alpha-pruned opponent actions, then determinized worlds, then CRN
dice resamples. The ORDER is the three-axis variance measurement's ranking, not a guess.

The ``oracle - honest`` gap is the BELIEF's price and the ``honest - base`` gap is the dividend
actually available today; separating them is why the probe has three arms rather than two.

Modules
-------
``determinize`` pool-consistent world sampling + THE prefix byte-identity gate ·
``record`` synthesizing a reconstruction record for a battle still in flight ·
``alpha`` the opponent-marginalization candidate set (the alpha-consumer contract) ·
``budget`` the wall-clock deadline + the width allocator ·
``search`` the depth-1 search itself ·
``player`` the search-wrapped eval player ·
``battery`` matched-game driver + append-only resumable results ·
``summary`` Wilson intervals + anchored-ELO deltas.
"""
