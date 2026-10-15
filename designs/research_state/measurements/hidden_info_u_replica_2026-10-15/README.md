# Hidden-info study: the blind U-arm replication (red team E1, 2026-10-15)

## In plain language

The hidden-information study asks how much better a search would decide if it knew the opponent's hidden facts. Its
yardstick is **IMP**: how much the search's pick beats the policy's own pick, in win probability per decision,
judged by a referee that played every action out 24 times in the TRUE battle. The search plays its rollouts either
in the true battle (**O**, the oracle) or in **fair worlds** (**F**): copies of the battle in which every fact the
searched side has not seen is re-drawn from the Smogon usage prior, kept only if replaying the game so far shows the
searched side exactly what it really saw.

This folder adds the two arms that isolate **U**, the opponent's NEVER-SEEN team members (a mon the searched side
has not seen switch or be dragged in before the decision: its species, moves, item, ability, nature, EVs, IVs,
gender and slot):

- **F+U**: each fair world with the never-seen members replaced by the TRUE ones (revealing U alone).
- **O-U**: the TRUE battle with the never-seen members re-drawn from the prior (hiding U alone).

It was built by EXTENDING the independent replica's builder (`../hidden_info_replica_2026-10-11/replica/`),
blind to the first implementation's code and to every prior result. A third party compares it with the first
implementation, under the pre-registration in [`PREREGISTRATION.md`](PREREGISTRATION.md) (committed `8dd25833`
before any arm output existed). **This folder does not compare.**

## Status

RUNNING (see "Runner and handoff" below). The read is not done; the sections marked PENDING fill in when it is.

## What was built (`hiu/`)

| file | what it does |
|---|---|
| `fix.py` | installs two changes into the replica's modules at run time (the replica's files are unchanged): the CURRENT learnset table, and the never-seen draw placed uniformly |
| `arms.py` | the world sets: F / F+M (the replica's `fair.build`), F+U (F's worlds with the true never-seen entries), O-M and O-U (the true team with M or U re-drawn) |
| `guard.py` | the consistency guard: every world's contract (which slots are true, which drawn) and the engine's own view of the team, slot by slot; a mismatch RAISES |
| `run.py` | the resumable runner: per state, all six arms, one JSON row per arm, a done-marker per state |
| `score.py` | this build's own read: IMP per arm and scorer, s_U, n_U, s_M, n_M, R3's numerators with intervals, the identities, the exclusions, the per-state IMPs, the marginals |
| `run_worker.sh`, `snapshot.sh` | the detached worker (rule 5b) and the code snapshot it runs from |

### The learnset fix (every arm)

The replica's move prior and legality check read the learnset table at the checkpoints' PINS, which predates
`64a695dc` (2026-10-14) and lacks 1,500 legal (species, move) pairs: every move inherited from a pre-evolution
(Tyranitar Dragon Dance, Breloom Spore, Swampert Hydro Pump, Umbreon Wish). Measured here: the pinned table (both
pins identical) holds a strict subset of the current one, 1,500 pairs added and none removed. `hiu.fix` replaces
`replica.sampling.learnset` with the current table (`data/pokemon/gen3_learnset.json`, sha256
`c90202a1…a456596`, checked at load; the facade's semantics: an unknown species is unconstrained) for EVERY arm,
F, F+M and O-M included, and asserts at start that Tyranitar Dragon Dance is back in the prior. F, F+M and O-M are
therefore re-run here, paired with F+U and O-U on the same states, so every share is handicap-free.

### The U arms

- **F+U**: world j = this build's F world j with every never-seen slot replaced by the TRUE packed entry (its
  realized gender) at its TRUE slot; the seen mons' draws stay exactly F's (the guard checks byte equality). Gated
  exactly as F (lines AND rows byte-equal, legality, the playout tool's replay). F+U is F's 8 worlds or nothing: a
  world that fails fails the state (reason recorded), never a re-draw.
- **O-U**: the TRUE team; every never-seen slot re-drawn as F draws it (species from the Smogon teammate prior given
  the revealed species, the mean over the revealed species, then a whole set from the Smogon prior), no species
  twice and no gen-3 OU clause broken, the drawn mons placed on the never-seen slots in a uniformly random order;
  every seen mon verbatim true. Gated as F, up to 60 draws for 8 worlds.
- Seeds and scorer as §13.1: world j on dice seed j of `truth.turn_seeds(id, 48)`; LEAF-K1 / K4 / K8; IMP =
  T(argmax, ties toward the policy's order) − T(top1), the referee on truth seeds 16-39.

## Spec ambiguities, and the choice made for each

| # | where the spec is silent | this build's choice |
|---|---|---|
| U1 | the opponent's PENDING root choice (logged before the searched side's, not yet executed): the replica drops it in every constructed world (its A3), O keeps it | the arms built from the TRUE team (O-M, O-U) KEEP it, rendered for the world, as O does: an "O minus x" arm then differs from O only by x, and the registered identity O-U = O at k = 6 holds on the pending states too. The arms built from F (F, F+M, F+U) drop it, as the replica's F does. This changes O-M from the replica's convention on the pending states only (7 of 600 end2 states) |
| U2 | "place the drawn mons on the never-seen slots in a uniformly random order" vs the replica's slot-by-slot draw (not exchangeable: earlier slots get the popular species more often) | the mons are drawn as the replica draws them, then permuted uniformly onto the never-seen slots, in EVERY arm that draws U (F, F+M and their screens' contexts, O-U) |
| U3 | F+U's world that fails the gate | the state fails F+U with the reason; no re-draw (F+U must be F's world j) |
| U4 | the gender of a true never-seen entry | its REALIZED gender (the replica's seed alignment plays explicit genders) |
| U5 | the opponent population of a state (secondary read) | `sentinel_<i>` opponents are our own snapshots (the trainer's eval label for a snapshot); every other label is a scripted bot |

## Identities (the build is invalid unless they hold)

At k = 6 (every opponent mon seen) F+U ≡ F and O-U ≡ O, outcome for outcome (every action, every world).
`hiu.score` checks them on every k = 6 state and records each mismatch. PENDING (the verification run's result is
in "Runner and handoff").

## Runner and handoff

PENDING.

## Blindness (what this build read and did not read)

- **Read:** the replica's `replica/` code and its README; `design_hidden_info_binding.md` §13.1 only (plus, once,
  the document's heading list to find §13.1; no other section's text); the registered states
  `value_disc_pilot_2026-10-11/states_<cp>.json`, the bank and the Phase 1 truth rows they point to (the rows for
  the referee only); the pin checkouts `~/.cache/gen3ai/vdp/pin_{end2,prod}`; `agents.gen3_data.learnset` (current)
  and the learnset fix commit's FILE LIST only (`git show --stat`, message not read); the Rust engine's request
  format (by running it).
- **Not read:** `src/main/value_search/`, `src/agents/info_levers/`, any `determinize.py`,
  `hidden_info_tier1_2026-10-13/`, `~/gen3ai_archive/hidden_info_tier1/`, `design_hidden_info_binding.md` §16 or
  any section but §13.1, `search_ledger.md`, `UNDERSTANDING.md`, `ledger.md`,
  `hidden_info_by_reveal_count_2026-10-12/`; the replica's own rows, `result.json`, `marginals.json` and
  `imp_per_state.json` were not opened. The replica's README holds its own final tables; it was read for the build
  description, and none of its numbers is used here.
