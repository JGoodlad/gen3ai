## Turn mechanics, verified at the source (`deps/pokemon-showdown` @ `e0551883f`, gen 3 mod → gen 4 → gen 5 → base)

`data/mods/gen3/scripts.ts:2` inherits `gen4`, `data/mods/gen4/scripts.ts:2` inherits `gen5`; neither
overrides `runAction`, `checkFainted`, `makeRequest`, `getRequests` or `commitChoices`, so the base
`sim/battle.ts` / `sim/side.ts` code below is what gen 3 runs.

| question | answer | source |
|---|---|---|
| After a faint, who chooses a replacement, and when? | Only the side with a fainted active: `checkFainted` sets its `switchFlag` (`sim/battle.ts:2537–2546`), `getRequests('switch')` gives a `forceSwitch` request ONLY to sides whose active has the flag (`:1420–1428`), and every other side gets `{wait: true}` (`:1453–1457`). **In gen ≤ 3 this happens after EVERY action, mid-turn**, not at the end of the turn: `runAction` calls `checkFainted` when the next queued action is a move or the residual (`:2861–2864`, the comment "in gen 3 or earlier, switching in fainted pokemon is done after every move"), then `makeRequest('switch')` (`:2933`). The rest of the turn's queue is SAVED and resumed after the replacement, not re-sorted (`commitChoices`, `:3021–3040`). A fainted Pokémon's still-queued move is dropped (`data/mods/gen3/scripts.ts:157`) | as cited |
| If both sides faint, are the two replacements simultaneous? | **Yes.** One `makeRequest('switch')` issues a `forceSwitch` request to BOTH sides in the same `getRequests` pass (`:1420–1428`); nothing resolves until `allChoicesDone()` (`:3081–3090`, checked in `commitChoices` at `:3028`), and a choice is not revealed to the other side before the commit (the `inputLog` push and the switch-ins run at `:3030–3042`). Each side chooses blind to the other's replacement | as cited |
| Baton Pass mid-turn: is the opponent's move for that turn already locked? | **Yes.** Both sides' turn choices were committed into the queue at the turn's start (`commitChoices`); Baton Pass is `selfSwitch: 'copyvolatile'` (`data/moves.ts:1113`), which sets the user's `switchFlag` (`sim/battle-actions.ts:1332–1333`); `runAction`'s switching block then requests a switch from THAT side only (`sim/battle.ts:2898–2935`, the other side `wait`) and the saved remainder of the queue — the opponent's not-yet-executed move included — is appended after the new switch-in (`:3021–3040`). The opponent cannot change its move; it lands on whatever came in | as cited |
| On the next turn, is the opponent's choice made knowing our switch-in? | **Yes.** Every switch-in is broadcast as it happens (`sim/battle-actions.ts:146–148`, the `switch` line), and the next turn's move request is issued only by `endTurn()` (`sim/battle.ts:1620`, `makeRequest('move')` at `:1797`), i.e. after the whole turn — replacements included — has run and been logged | as cited |
| (found on the way) A hidden trap | A switch into a `maybeTrapped` request is REJECTED and re-requested with `trapped: true` (`sim/side.ts:966–978`); the recorded log then holds TWO opponent commands for one decision (the rejected switch, then the move). One banked turn hit it (below) | as cited |

**The X4 spec's §5.1 ("sequential after a faint; the opponent's reply must be modelled") — PARTLY
CONFIRMED, PARTLY CONTRADICTED.** Confirmed: after a single faint only the fainted side chooses (the
other waits), and the opponent's next action is chosen at the next turn's request knowing the
replacement — so it must be modelled, not held. Contradicted: (1) a DOUBLE faint is SIMULTANEOUS,
not sequential; (2) in gen 3 the replacement is MID-TURN, and when our Pokémon faints before the
opponent has moved (our own Explosion into a Protect, say) the opponent's move for that turn is
already LOCKED and lands on our replacement — there is no reply to model for it, only an action
the observation at the forced-switch decision cannot see.

**How these branches hold or model the opponent, per decision type:**

| decision at the branch point | variant R | variant M | truth (Lane S) |
|---|---|---|---|
| free move turn, opponent answered BEFORE us in the log (its choice is in the replayed prefix) | held at its recorded choice | same | same |
| free move turn, opponent answers AFTER us (its request OPEN at the root) | held at its recorded choice (moved in front of ours; a rejected-then-retried answer moves both) | the checkpoint's GREEDY choice on its own root row (identical across siblings) | the continuation's greedy choice |
| our forced replacement, opponent waiting (single faint) | nothing to hold: the opponent has no decision; any locked move of its stays in the sim's queue | same | same |
| double faint (both replace, simultaneous) | its recorded replacement | the checkpoint's greedy replacement | the continuation's greedy |
| after our action, before OUR next decision: the opponent's own replacement (its mon fainted) | the checkpoint's greedy choice (the dice differ from the record, so the recorded one does not apply) | same | the continuation's greedy |
| our next decision is itself a forced switch (our mon fainted during the turn) | V scored at that forced-switch observation — it cannot see an opponent move still locked in the queue | same | played on |
