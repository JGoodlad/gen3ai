# Vacuous tests and guards — when green does not mean what it looks like

> **What this is.** A durable explainer for the failure mode where a test, a gate or a defensive
> guard **passes without evaluating the thing it exists to prove**. It is not flakiness (a flaky
> test fails sometimes), it is not a bug in the test (a wrong test fails loudly), and it is not
> low coverage (a missing test is *visibly* missing). It is the one failure whose symptom is
> **exactly indistinguishable from success**. Covers the seven-way taxonomy, the ten specimens
> this repo produced in a single week (2026-08-22/23), the detection method that works for each
> class, the design rules that make each class *unrepresentable*, and the meta-lesson about
> redundant meters. Intuitive first, then technical, no code.
>
> **Companion to [[objective_richness_and_representation]] on the epistemics side.** That note is
> about what a *model* is forced to represent by its targets; this one is about what a *gate* is
> forced to evaluate by its structure. Both are the same move: you do not get a property by
> hoping for it, you get it by making its absence impossible to express.

---

## TL;DR

- **The defining property:** a vacuous test emits the same signal as a passing test. Every other
  kind of broken test is self-announcing. This one recruits your trust and spends it.
- **The economics are inverted.** A missing test costs you coverage. A vacuous test costs you
  coverage **plus** the confidence you would otherwise have spent on getting coverage. It is
  strictly worse than no test — a point this repo now states as a rule: *a gate that keeps
  passing while its subject stops existing is worse than no gate.*
- **Seven classes**, all found for real here: guarded assertions · skip-forever ·
  exception-swallowing setup · guards that cannot fire · presence-not-value · source-scanning on
  moved literals · single-draw verdicts.
- **The common shape is a BRANCH between the runner and the assertion.** Every class is some way
  of putting a conditional in front of the decisive comparison — `if`, `skip`, `except`, an
  impossible predicate, a mis-keyed filter. Delete the branch and the class dies.
- **The costliest measured instance**: `logits > -1e8` as a legal-action recovery returned
  ALL-LEGAL on **0-of-800+ sampled traces back to ai_v5_2**, so every ablation audit ever run
  scored flips/KL over an action space **38.4% wrongly counted legal** — and the audit that owned
  the bug printed "0 zero-legal rows" for a year while measuring phantom actions.
- **The most embarrassing instance** (found 2026-08-23, and the cleanest teaching case): a test
  named `test_prior_logits_hidden_power_sums_typed_usage` selected Hidden Power variants by
  `move.num == 237`, which stopped matching anything the day typed HP got its own dex nums
  355-370. It skipped **on every tree, forever**, with a message blaming the *data* — "no
  HP-running species in the sample". The production code had been right the whole time.
- **Detection is adversarial reading, assisted by AST — never grep.** Comments and docstrings do
  not reach an AST, so an AST-based gate cannot degrade into a documentation argument. But the
  worst specimens here were **cross-scope** (a counter initialised in one function, its floor
  asserted in another) and no scanner found them. Automation narrows; reading convicts.
- **Five design rules make the classes unrepresentable:** assert-the-build · value-not-presence ·
  seams-not-literals · distributions-not-draws · reachability-beside-presence.
- **The meta-lesson: redundant, differently-plumbed meters are why the big decisions survived.**
  When the mask defect detonated, every `|dV|`-keyed verdict stood — because `|dV|` never touches
  the mask. The deletion-class calls had been deliberately keyed on a *different instrument* than
  the flips/KL axis. That redundancy was not free and it paid for itself in one afternoon.

---

## Part 1 — The intuition

### Why this class is special

Imagine a smoke detector. There are three ways it can be wrong.

1. **It goes off when there is no fire.** Annoying, but you learn about it immediately.
2. **You never installed one.** You know you have no detector. You can decide how much that
   worries you.
3. **It is installed, its light is green, and its sensor chamber is sealed shut.**

Only the third one is dangerous, and it is dangerous for a reason that has nothing to do with
fire: it *consumes the attention you would otherwise have spent installing a working detector*.
The green light is not neutral. It actively bids for your trust, and you pay it.

Software gates work identically. A failing test summons someone. A missing test is visible in a
coverage report, or in the uneasy feeling that a subsystem is untested. But a test that runs,
touches nothing, and reports success is **load-bearing in your decision-making while carrying no
load at all**. Every subsequent judgment leans on it.

### The mechanical cause is always the same

Look at the seven classes below and one shape recurs: **there is a branch between the test runner
and the assertion.**

```
runner → [ some condition ] → assertion
```

If the condition can be false, the assertion can be skipped, and if the skip is silent the test
reports green. That is the entire mechanism. The branch can wear many costumes:

| costume | how it skips |
|---|---|
| `if x is not None:` | the assertion is inside the `if` |
| `pytest.skip(...)` | the runner is told not to bother |
| `except: return` | the setup failed and said nothing |
| an impossible predicate | the guard's condition is never true |
| a mis-keyed filter | the loop body never executes |
| a moved literal | the string being searched for is not there any more |
| one random draw | the condition happened not to arise *this time* |

Once you see it as "a branch in front of the assertion", the fix generalises: **either remove the
branch, or assert the branch.** If the condition is a genuine precondition — *this test does not
apply here* — then say so in a way that is itself checkable. If the condition is something the
test *arranged*, then its failure is a bug and must be an assertion, not a skip.

### The distinction that does the work: **arranged vs. encountered**

This is the judgment call, and it is worth stating precisely because it is the only part that
cannot be automated.

- A condition the test **arranged** (it built the config, it pinned the fixture, it asked for the
  flags) failing is a **defect**. Skipping is wrong. Assert.
- A condition the test **encountered** (there is no GPU on this box, chrome is not installed, the
  archive directory does not exist) is a genuine precondition. Skipping is right — *provided the
  skip is falsifiable*, i.e. someone somewhere can and does drive the other branch.

Two live examples from this tree, deliberately resolved in opposite directions:

- `value_entity_pool_test` builds a policy with `value_entity_pool=True,
  value_entity_pool_full=True, value_threat_inject=True` and then skipped "if this build resolved
  without both value parts". It **asked** for both parts. Resolving without them is a flag-resolution
  bug. That is now an assertion.
- `compile_trainer_test` skips without CUDA. The box either has a card or it does not; the test
  arranged nothing. That skip is correct — and it is falsifiable, because this box has a card and
  the test runs here.

### The falsifiability half of a skip

The second bullet hides the subtler rule, and it is the one this repo learned the hard way:

> **A skip that is supposed to happen is indistinguishable from a skip that is not.**

Four tests here sat behind a *correct* skip-if-missing guard on a path literal
(`/home/goodlad/dev/gen3ai/...`). On the owner's box they ran. Everywhere else they skipped — and
a second contributor silently loses the production-config drift gate, the real-trace mask
recovery, the gradient-flow tests and the eval-sharding fuzz, and **is never told**. The defect
was never the skip. It was that the skip was *unfalsifiable*: nothing anywhere exercised the other
branch, so nobody had ever seen the skip path work.

The fix that generalises: make the skip condition **injectable**, then drive it. Here
`$GEN3AI_MODELS_DIR` became authoritative — and set-but-missing yields `None` rather than quietly
falling back to the real archive, because a fall-back would make the override useless as exactly
the test seam it exists to be. *A skip path nothing exercises is a skip path nobody has ever seen
work.*

---

## Part 2 — The taxonomy, with the measured record

Ten specimens, found 2026-08-22/23. They are the evidence base for the whole note; the pattern
numbers are used throughout.

### 1. Guarded assertions — `if x is not None: assert ...`

The decisive comparison sits inside a conditional with no `else`, so an empty/None intermediate
walks past it in silence.

- Both prober integration tests (`lookahead`, `better_line`) could pass **while walking past the
  only thing they exist to prove**; the falsifier asserted on a list that could be empty.
- `counterfactual_fuzz_test` hung its checks 3 and 4 — divergence-to-terminal and the Monte-Carlo
  reseed, **six assertions**, and the larger half of the script's purpose — off
  `if anchor is not None`. A battle with no mid-game move decision skips all of it; the
  full-replay legs still pass and the script still prints `PASSED`. It even printed
  `divergence@turn=n/a` on that line — *an announcement is not a gate*.
- `thread_pinning_test` guarded its ordering assertion — that the BLAS pin runs **before** torch
  is imported, defending a measured **6 fps vs 231** cliff — with `if torch_line is not None`. The
  day the hub's imports stopped matching the detector's root list, the guard would go quietly
  inert. (That day nearly arrived: the 2026-08-22 entry-point decomposition removed the hub's
  direct `import torch` entirely, which is why the detector is now written by **effect** —
  "anything that transitively pulls torch in" — rather than by spelling.)

**Detection:** AST. Walk every `*_test.py`, find `ast.If` nodes whose body contains an `ast.Assert`
and whose `orelse` does not. Rank by assert count — the six-assert one surfaces first. Tree-wide
here: **38 hits, of which the large majority are legitimate** (`else`-arms that assert the other
branch, type narrowing, per-case dispatch). The scan narrows 5,000 tests to 38; reading picks the
four.

### 2. Skip-forever — a skip whose condition is never false

- **The worst specimen of the whole hunt.** `test_prior_logits_hidden_power_sums_typed_usage`
  selected typed Hidden Power by `moves.get(mid).num == HIDDEN_POWER_NUM` (237). That was true
  before `gen3_typed_hidden_power_ids_v1` gave the 16 typed HPs their own dex nums **355-370**.
  Afterwards the filter matched **nothing, for every species**, the sum was always `0.0`, the
  floor test `0.0 > 0.02` was always false, and the test skipped **unconditionally on every tree
  it ever ran on** — announcing "no HP-running species in the sample", i.e. blaming the data.
  Production had been correct throughout: it keys the fold on the move **id** prefix, via
  `_belief_num`, not on the num. So the test was pure dead coverage over a **named GIGO bug
  class** (`project_opp_hp_immune_bug`) — the worst place to lose a gate.
- **The second live one, and the clearest illustration that these arrive in clusters.** The gate
  validating that every team the trainer can be dealt is legal gen3ou opened on
  `data/teams/teams.json` — a manifest that does not exist under the current layout — and skipped
  blaming the operator: "Run sync-teams first." **Three defects were stacked, and all three are the
  same mistake in different costumes**: the stale manifest path (skip-forever); CWD-relative paths
  throughout (the relative sibling of the `/home/...` literal class); and a missing team file that
  printed a warning and `continue`d, so even on the happy path a layout change would have validated
  **zero** teams and reported success. This is the general shape: *a test that has rotted once
  usually rotted three times, because whatever stopped anyone reading it stopped them reading all
  of it.*
- Four tests skipped forever off this box on absolute `/home/goodlad/...` paths (above).
- Two tests skipped on a config *they themselves constructed* — `intent_axis_alignment` (which its
  own docstring calls "**THE** gate", for the named `project_op_move_order_bugclass` that "has
  bitten before") and `value_entity_pool`. Both currently pass, so these are **latent**: a
  trapdoor that opens the day flag resolution changes.

**Detection:** two halves, and you need both.
- *Static*: enumerate every `skipif` / `pytest.skip` / `importorskip` / early `return` and ask, per
  site, **"can this condition ever be false on this box AND on a fresh clone?"** (52 sites here.)
- *Empirical, and far cheaper*: **run the suite with `-rs` and read the skip census.** A skip that
  actually fires shows up as a line. The Hidden Power specimen was invisible to every static scan
  and fell out of `-rs` in four seconds.

The static half is what catches the *latent* ones (the two config skips that don't fire today);
the empirical half is what catches the *live* ones. Neither subsumes the other.

### 3. Exception-swallowing setup

- The canonical specimen, in Rust:
  `bridge_choice_reject_test::a_forced_struggle_substitutes_rather_than_rejecting` opened with
  `match … { Err(_) => return }`, whose comment blamed "a possibly-unmodeled Bide". The real cause
  was its own fixture: a packed team written as a single-mon set with **no trailing `]`**, which
  never unpacks. **That test skipped on every tree it ever ran on.** It read exactly like a green
  gate. It now asserts its build.
- In Python, `server_port_threading_test`'s "no in-process player creation" guard looped over class
  members with `except (OSError, TypeError): continue` around `inspect.getsource`, and then made a
  **negative** assertion (`"server_configuration=" not in src`). A negative assertion over a loop
  that swallows its own setup failure passes just as cheerfully over **zero iterations**.

**The negative-assertion amplifier is worth isolating**, because it inverts the usual safety
margin. `assert "x" in src` fails loudly if `src` is empty or the wrong file — the mistake is
self-announcing. `assert "x" not in src` **passes** on an empty `src`. So source-scanning in the
negative direction needs a companion floor on how much source was actually read; in the positive
direction it does not.

**Detection:** AST for `ast.ExceptHandler` whose body is only `pass`/`return`/`continue`/`break`.
28 hits here. Most are legitimate — the `try/except SomeError: return` + `assert False` idiom for
"this must raise", process-cleanup `except ProcessLookupError: pass`, and production capability
probes like `nvidia-smi` returning `None`. Two were real. **This is the class where the scan's
precision is worst and reading matters most.**

### 4. Guards that cannot fire

A defensive branch whose condition is impossible against the current data shapes.

- **The costly one.** The audit layer recovered the legal-action mask as `logits > -1e8`, but the
  recorder stashes **pre-mask** logits — nothing is ever below `-1e8` — so the recovery returned
  ALL-LEGAL on **0-of-800+ sampled `states.npz` back to ai_v5_2**. Every ablation audit ever run
  scored flips and KL over an action space **38.4% wrongly counted legal** on average (min 18%,
  max 68%; 100% of rows carry at least one illegal action). The guard that was supposed to catch
  this checked for *zero* legal actions — which the broken recovery could never produce. **Its
  missing half was the all-legal case**, and the audit printed "0 zero-legal rows" for a year
  while measuring phantom actions. The real mask had been on disk the whole time.
- A floor assertion that cannot fire: `search_clone_parity_fuzz_test` asserts
  `total >= 1, "no obs check ran"`, where the per-battle counter is **initialised to 1** before any
  observation is compared. All four of its obs checks sit behind `ended`/`None` guards and can be
  skipped together, so a run could report `PASS — N battles ... (N checks bit-for-bit)` having
  compared **zero** observations.
- Five `_resolve(name, default)` lines sat beside a **non-None** argparse default. `_resolve` fires
  only when the value `is None`, so the lines were dead code — while the test asserting they were
  *present* passed happily. Two were production flags, either of which would have made a flagless
  resume of production FATAL.

**Detection:** this is the class automation is worst at, and the record proves it. An AST scan for
"floor asserted at or below the counter's initialiser" found the class **but not the specimen**,
because the counter was initialised in one function and its floor asserted in another. Cross-scope
dataflow beat the scanner. What works instead: for every guard in audit/instrument code, ask
**"what input makes this fire?"** and then *construct* it. If you cannot construct one, that is the
finding.

### 5. Presence-not-value

An integration test asserting that an artifact exists, or that a count is non-zero, without ever
checking a **value** against an independent oracle. A composition test that proves the pipes are
connected but not that anything correct flows through them. (The prober specimen is the canonical
one.) Note this is a **judgment** class, not a mechanical one — plenty of count-checks are exactly
right; the ones that matter are those standing guard over a data pipeline where a wrong value is
the realistic failure and an absent file is not.

**Detection:** grep `assert .*\.exists()` / `len(...) > 0` in `*_integration_test.py`, then judge.
Tree-wide here (excluding the prober) this came back **essentially empty** — one hit, a pinned-team
existence check that is a correct precondition. **That is a real result**, and worth recording as
one: this codebase's integration tests do mostly assert values.

### 6. Source-scanning tests anchored on moved literals

A test that reads source text or AST from a path and asserts a string is in it. Two ways to rot:
the **anchor** moves (the file is renamed or decomposed) or the **literal** moves (the code is
refactored).

- Ten such gates were repointed when `train_rl_agent.py` became the `main/train/` package. The seam
  is `entry_source()` — one function that knows where the entry point's source lives, so the ten
  gates ask *it* rather than each re-deriving a path.
- The general failure has a memorable instance: **a hand-written flag list under a comment claiming
  it was "queried from the registry, not guessed."** It was a literal. It had gone stale.
  *A literal under a comment that says it is not a literal is the failure mode, not the literal.*

**Detection:** AST for `assert <str-literal> in <name>` in files that call `read_text()` /
`getsource()` (80 hits here). Then split by direction: the `in` form fails loudly when it rots, the
`not in` form rots silently (see class 3). Only **two** negative-direction assertions exist
tree-wide, both in the same test, which is why this class was cheap to clear — and the split is the
whole trick, because it turns 80 candidates into 2.

### 7. Single-draw verdicts

A gate whose verdict rides one random artifact.

- **The measurement that settles the argument.** A golden here is three real battles. Across seven
  freshly generated goldens, per-golden divergence counts ran **1 / 0 / 0 / 6 / 8 / 0 / 6**. Three
  of the seven seeds would have reported the gate **green while four rust bugs were live** — and
  one bug class turned up only on the **seventh** seed. Each class needs its own rare board (a
  Return carrier; a Substitute broken near a sampled turn; a battle *ending* on a fire KO of a
  frozen mon; a mon at 0 PP on a sampled turn).
- The recorded "29 divergences" and "1 divergence" in the docs were **single draws read as
  measurements**.
- The same week produced the mirror-image error on the reporting side: a unit test fed
  `timed_out=4, attempted=12` into the *real* warning function, so under `-s` it emitted
  `⚠️ small run: 4/12 battles TIMED OUT (33%)` into the same stream as live measurements — and was
  taken for one **during this very investigation**. Its label is now `SYNTHETIC unit-test sample`.
  *A number that cannot be told apart from a measurement will eventually be read as one.*

**Detection:** for each gate, ask "how many independent draws back this verdict?" If the answer is
one, run it twice with different seeds and compare. **Two seeds is the floor, not the target.**

---

## Part 3 — The five design rules

Detection finds today's instances. These make tomorrow's *unrepresentable* — the same move the
entity-token architecture makes for whole bug classes (see [[entity_tokens_biases_pointers]]): do
not defend against the error, remove the syntax that expresses it.

### Rule 1 — **Assert the build**

Test setup that fails must **fail the test**, never skip it. If the test arranged the condition,
its absence is a defect.

- Rust: never `Err(_) => return` in setup. `unwrap()` or `expect("...")` — the panic message *is*
  the diagnosis, and it is one you will read.
- Python: no bare `except: return` around fixture construction. If you must tolerate a failure in a
  loop, **count the iterations that succeeded and assert a floor** — especially before a negative
  assertion.
- Corollary: **an announcement is not a gate.** Printing `divergence@turn=n/a` and then printing
  `PASSED` is two facts on one screen with no relationship between them. Make the count a floor.

### Rule 2 — **Value, not presence**

Assert what a thing **is**, against an oracle that was computed a different way. `exists()` and
`len(x) > 0` are preconditions, not conclusions. The strongest form is a **cross-implementation**
or **cross-route** comparison — node vs rust, clone vs re-roll, static width vs a real forward —
because the two sides can only agree by both being right.

### Rule 3 — **Seams, not literals**

If a test needs to know where something lives or how something is keyed, it asks the **same
function production asks**. Three shipped instances of this rule, each retrofitted after the
literal rotted:

- `entry_source()` — where the entry point's source is (ten gates).
- `utils.paths` — how deep a file sits in the tree (25 sites converted; **18 deliberately left**
  `__file__`-relative, because a module locating an asset that ships *beside* it is a local fact
  that must not be made to depend on a global one).
- `_belief_num` — how a move id folds to a belief column. The Hidden Power specimen is precisely
  what happens when a test re-derives this rule from a literal instead: production re-keyed, the
  test did not follow, and the divergence expressed itself as a **skip**.

The rule generalises: **a test should share the code under test's *definitions*, and differ from it
only in its *conclusions*.** Sharing a definition makes the test move with the code. Sharing a
conclusion makes the test tautological. Getting these backwards is how you end up with a gate that
is simultaneously brittle and vacuous.

### Rule 4 — **Distributions, not draws**

Two seeds is the floor. If a gate samples, its verdict must survive a re-sample, and its report
must say **how many draws** it rests on. Never quote a per-draw count as a rate. And keep synthetic
numbers typographically distinct from measured ones.

### Rule 5 — **Reachability beside presence**

For every contract with a "the line is there" gate, add a "the line **runs**" gate. This is the
pairing that caught the five dead `_resolve` lines: the presence half passed for years; the
reachability half (assert the argparse default is actually `None`, **against the built parser**,
not the source text, because a default can be an expression and only the constructed object knows
its value) failed immediately.

Generalised: **presence is a property of the source; reachability is a property of the runtime.**
A source-level gate can never see a dead branch. Pair them.

**And the rule that guards the rules:** every gate needs a **vacuity guard** — a test that the
scan *can* fail. `paths_test.py::test_the_scan_can_actually_fail` is the template. It is the only
defence against the gate itself joining this taxonomy, which is not hypothetical: an allowlist
entry here **outlived its own fix** and then misled every reader after, including a subagent
briefed from it. When a deferral dies, **delete it**.

---

## Part 4 — The meta-lesson

### Only adversarial reading finds these

Notice what did *not* find any of the ten: the test suite (5,988 green), code review at the time
of writing, CI, or coverage tooling. Coverage is especially seductive and especially useless here
— **a vacuous test has excellent line coverage.** The `logits > -1e8` recovery executed on every
run for a year. Every line was covered. Every line was wrong.

The three things that did work:

1. **Reading the test as an adversary** — asking, of each assertion, *"what would have to be true
   for this to be skipped?"* rather than *"is this assertion correct?"*
2. **Running the instrument on data whose answer you already know.** The mask defect was convicted
   by pointing the recovery at 800 real traces and observing that it returned all-legal on
   **100%** of them. No amount of reading proves that; one measurement does.
3. **`-rs`.** The cheapest high-yield instrument in the whole hunt. A live skip-forever is four
   seconds away at all times and almost nobody looks.

The corresponding habit is a reading posture, not a tool: **when a gate reports green, ask what it
would have to have seen to report red — and check that it saw it.**

### Redundant, differently-plumbed meters are the actual insurance

This is the part worth internalising, because it is the only thing that limited the damage.

When the mask defect detonated, the blast radius was real and non-uniform: gen-17 `all` kl_mean
moved **−39%**, `h` **−54%**, but `t` **+25%** and `concat_cells` flips **+8%** — `t` and `h`
**swapped rank**. Twenty-four committed pre-fix artifacts became ordinal-only within a single file
and never comparable to a post-fix number.

And yet **every architectural deletion decision stood**. Not by luck: the deletion-class calls had
been deliberately keyed on `|dV|` — the critic delta — and **`|dV|` never touches the mask.** The
policy-side flips/KL axis and the critic-side dV axis are two meters plumbed through different
code. One was broken for a year. The other was not, and it was the one carrying the load-bearing
verdicts.

The general principle: **for a decision you cannot afford to redo, do not merely double-check the
meter — read a second meter that shares no plumbing with the first.** Two instruments that agree
are weak evidence when they share a defect (both the E-battery's pooled numbers and the conditioned
read carried this defect, so their *agreement* proved nothing about the absolute values — though
their *paired deltas* largely cancel it, which is why the relative claims survived and the absolute
ones had to be re-baselined). Two instruments that agree **and cannot share a defect** are strong
evidence.

That redundancy is not free. It looks like waste right up until the afternoon it pays for the
entire research programme.

### The three sentences to keep

> **A test that can pass without evaluating its assertion is indistinguishable from a passing
> test.**

> **A guard that cannot fire is not a guard.**

> **A gate that keeps passing while its subject stops existing is worse than no gate.**

---

## Appendix — the hunt as a procedure

Reusable. Roughly an afternoon over a ~5,000-test tree.

| # | pattern | instrument | precision |
|---|---|---|---|
| 1 | guarded assertions | AST: `If` with `Assert` in body, none in `orelse`; rank by assert count | good — 38 hits, ~4 real |
| 2 | skip-forever | **`pytest -rs` skip census** (live) **+** enumerate every skip site (latent) | excellent live, fair latent |
| 3 | swallowing setup | AST: `ExceptHandler` whose body is only `pass`/`return`/`continue` | poor — 28 hits, ~2 real, all judgment |
| 4 | guards that cannot fire | **reading**, per guard: "what input makes this fire?" then construct it | automation fails (cross-scope) |
| 5 | presence-not-value | grep `exists()` / `len(...) > 0` in integration tests, then judge | n/a — came back empty here |
| 6 | source-scanning | AST: `assert <literal> in <name>` in files calling `read_text`/`getsource`; split by `in` vs `not in` | good; the `not in` direction is the risky one |
| 7 | single-draw | per gate: "how many draws?" — then run it twice | trivial to apply, rarely applied |

Order matters: **run `-rs` first.** It is four seconds and it finds the live ones, which are worth
more than the latent ones. Do the AST scans second to build the candidate list, and spend the
remaining time reading — the scans narrow thousands of tests to dozens, but every conviction in
this hunt came from reading a specific test and asking what it would take to skip it.

**And expect the negative results to be real results.** Presence-not-value came back empty here;
the "floor asserted below its own initialiser" scan came back empty *after* the fix and could not
see the specimen *before* it. Recording both honestly is what stops the next reader from
re-running the same scan and drawing the opposite conclusion from the same silence.
