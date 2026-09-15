### 4.2 Three findings that belong to the DATASET, before any head is fitted

**(a) On contested decisions the policy's own top-1 and top-2 are outcome-INTERCHANGEABLE.** Over
5,076 forks on identical dice: `top1` wins **0.7082**, `top2` wins **0.7078** — a difference of
**0.0004**. A uniformly random legal alternative wins **0.6795**, i.e. throwing the decision away
entirely costs **2.9 pp**. On the class of decision a searcher is built to fix, the policy's
ranking of its own two best actions carries no measurable value, and its ranking against a coin
carries under three points.

**(b) The policy's BLIND-SPOT rate is 4.5 %** [3.99, 5.16] — in 221 of 4,865 complete forks a
uniformly random legal alternative WON where both of the policy's own top-2 candidates LOST, on
the same dice. Registered band was 8–20 %: **refuted, low.** There is a real but small pocket of
value outside the policy's top-2.

**(c) 79.7 % of branch pairs are TIED** (11,674 of 14,652) — both branches reach the same terminal.
This is the structural tax on any sibling-discrimination objective built from terminal outcomes:
four fifths of the label budget carries no ranking information, and the surviving fifth is what
every number in §5 is computed on.
