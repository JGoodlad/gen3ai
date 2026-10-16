# PRE-REGISTRATION — E2, the COMPETENT-OPPONENT state bank (2026-10-16)

**Status:** committed and shipped BEFORE any E2 game is generated, any E2 state is drawn, and any E2 arm is built,
played or scored. A rule below changes only by a dated AMENDMENT appended to this file before the outputs it touches
exist. The scripts and the bank's ids follow in later commits. Each script states the rule it implements.

**What this settles.** The study is `design_hidden_info_binding.md`. Its Tier 1 refuted "the opponent's hidden moves are
THE binding constraint on search" (rule R3). The refutation rests on the never-seen Pokémon (class U) dominating the seen
mons' hidden moves (class M) on both sides. That half replicated in two builds, but only on the registered mix, where
58 % of the states come from games against scripted bots. On the 503 own-snapshot states R3 failed in both builds, and
the bot-minus-snapshot gap was clear: `red_team.md` §2.6 (b), and `third_party_comparison.md` (`da6d024e`). That read
was post hoc. E2 makes it the PRIMARY question on fresh states.

**The question:** on states from games against the network's own snapshots (competent play), which class carries more
of the honest search gap: the never-seen Pokémon (U), or the seen mons' hidden moves (M)?

## 1. The bank (per checkpoint)

The checkpoints are Tier 1's two:

| | run | pin | searched network |
|---|---|---|---|
| end2 | `rb_es_end2_s2001` | `62085750` | `final_model.zip` |
| prod | `rb_es_prod_s2001` | `95d014fa` | `final_model.zip` |

The order is **end2 first** (today's production lineage), then prod.

### 1.1 Games: fresh, generated at the pin

- **The tool.** The pin's own `main.ops.eval_trace_gen` generates them. It is the trainer's eval executor, run offline
  on CPU, and it writes the same core traces a live eval cycle writes.
- **One generated cycle per live cycle that fed the registered bank.** The steps are end2 `10000009`, `12000157`,
  `14000347` and prod `10000002`, `12000178`, `14000168`.
  - **The traced side** is the bit-exact network that played that live cycle (`eval_traces/step_<N>/snapshot.zip`).
  - **The sentinels** are every pool snapshot below the step: 3, 4 and 5 sentinels. These are exactly the live cycles'
    sentinel sets (`metadata.json` `latest_eval.pool.sentinels` per cycle).
  - **The regime** is the one the run recorded: greedy sentinels drawing the trainee's own team distribution.
- **Flags:** `--games 160 --sentinels 5 --opponents random --n-envs 16 --threads 2 --torch-threads 1`, no capture quota
  (every battle traced).
- **Seeds:** `--seed` = 202610161, 202610162, 202610163 for end2's three steps and 202610164, 202610165, 202610166 for
  prod's. Every game is a pure function of (seed, opponent, game index).
- **The `random` bot's games are never drawn.** The tool cannot generate a sentinel-only cycle, so they are a by-product.
- **Where they are written:** `~/gen3ai_archive/hidden_info_e2/<cp>/gen/e2_<cp>_<step>/` (a shadow run dir). Nothing is
  written under `models/`.
- **The checkout:** `~/.cache/gen3ai/hib_e2/pin_<cp>`. It is a detached checkout of the same pin commit, plus the
  Showdown submodule (the team validator needs it; the vdp pin checkouts lack it).

### 1.2 Capture: the live outcome quota's shape, at 2.4×

A live cycle traced the first 10 losses, 5 draws and 5 wins per opponent: loss-enriched by design. The registered bank
inherits that enrichment. E2 mirrors its 2 : 1 : 1 shape so that the two populations differ only in the opponent.

- **Per (step, sentinel) cell:** order the battles by `sha256("hib_e2_capture_v1:" + battle id)`. Take the first 24
  losses, 12 draws and 12 wins, by the traced side's recorded outcome.
- **A cell short of a cap** keeps what it has. The rule is deterministic and is never relaxed.

### 1.3 The pool and the draw: the registered bank's procedure (the pilot's `make_states.py`)

The procedure is unchanged except for the source (§1.1, §1.2) and the salt.

- **The pool.** Every captured battle is replayed (`core_events --trackers --obs`). Gate ① runs per battle; more than
  1 % of battles failing refuses the bank. The clock-decode turn gate must agree on every decision.
- **Eligibility:** a free move round, turn 2–40, ≥ 3 legal actions.
- **The root forward** uses the searched network: `logits_and_values`, batch 1024. The gap is the top-2 logit gap. The
  quintile edges are this pool's own 20 / 40 / 60 / 80 % quantiles.
- **Rule-8 exclusion (the registered one):** a state with |gap − edge| < 1e-6, or gap < 1e-6.
- **The draw:** **120 states per gap quintile, 600 per checkpoint**, in `sha256("hib_e2_states_v1:" + id)` order, at
  most **2 per battle**.
- **A quintile that cannot be filled STOPS the draw.** The one pre-registered extension is then a second generation batch
  per step, with seed + 1000 and the same flags. The capture rule (§1.2) is applied to the union (caps 48 / 24 / 24),
  and the draw is re-run from scratch on the union.
- **Disjoint from Tier 1 by construction.** These are new games, so no E2 battle or state is a Tier-1 battle or state.
  The script asserts it.
- **Outputs:** the bank (`main.policy_spectrum.bank` format) at `~/gen3ai_archive/hidden_info_e2/<cp>/bank/`, and the
  committed `states_e2_<cp>.json`. Per state that file records the gap, V, the logits, top-3, the quintile, k (opponent
  species seen), the turn, material, the traced outcome, the sentinel's step and the battle.
- **Teeth:** the same forward over the REGISTERED bank's 600 roots reproduces `states_<cp>.json`'s logits and V within
  1e-4. This checks the pin, the model and the environment.

## 2. The arms (the ORIGINAL implementation's harness, as Tier 1 ran)

| arm | what | built / played |
|---|---|---|
| **O** | the truth (Phase 1): every legal action × the pilot's 40 seeds (`truth.turn_seeds(id, 48)`: leaf 0–15, referee 16–39) + the one-ply critic (40–47) | the pilot's `run_playouts.one_state`, at the UNMODIFIED pin |
| **F** | the fully honest world (`WorldSampler(reveal="off")`) | built at HEAD, played at the PAD pin |
| **F+M** / **O−M** | the seen mons' unrevealed moves told / hidden | the same |
| **F+U** / **O−U** | the never-seen mons told / hidden | the same |

- **Worlds.** 8 per state and arm, from `main.value_search.worlds.WorldSampler(reveal=<spec>, max_tries=60)` at the
  E2 HEAD commit.
  - The HEAD commit is the commit that ships the E2 harness, checked out detached at `~/.cache/gen3ai/hib_e2/head`.
    `key` = the decision id. F+U shares F's seen-mon screen (common random numbers).
  - The **world consistency guard** (`world_guard.check_worlds`, landed `8702a56f`; `631db7d8`) runs inside every
    sample.
  - The slot / alias fixes since Tier 1 (`8702a56f`, `631db7d8`) are in. Tier 1 built at an earlier HEAD. Any
    builder-level difference is reported, not corrected.
- **Playouts.** The pilot's `run_det_playouts.one_state` plays them unchanged: world j on dice seed j, the checkpoint
  greedy on both sides, the honest one-ply critic on seed 40 + j.
  - **Where:** the Tier-1 PAD pins (`67f3f7fa` / `9a698801`: the pin + the engine's gender draw pad).
  - **Their engine source equals HEAD's** except comments and the oracle-reveal encoder. `git diff 67f3f7fa HEAD --
    src/rust_sim/src src/rust_env/src` names the same 7 files as Tier 1's README §2; this was checked on 2026-10-16.
- **IMP** = T(the scorer's argmax) − T(the policy's top1). T is the mean outcome over the TRUE world's referee seeds
  16–39, read by no scorer.

## 3. The PRIMARY read

### 3.1 Quantities

**Per-state IMPs**, for each source X ∈ {O, F, F+M, F+U, O−M, O−U}: IMP_X.

**The two R3 numerators (PRIMARY):**
- reveal-only: **Δs = IMP(F+U) − IMP(F+M)**;
- hide-only: **Δn = IMP(O−M) − IMP(O−U)**.

**The class contrasts:**
- σ_X = IMP(F+X) − IMP(F);
- ν_X = IMP(O) − IMP(O−X), for X ∈ {M, U}.

**Gap and shares:** G = IMP(O) − IMP(F); s_X = σ_X / G and n_X = ν_X / G (ratios of means on the same draws).

**Pairing.** Every contrast is computed on the states that hold every source it involves. Δn needs neither O's playouts
nor F, but every IMP needs the state's Phase 1 referee.

### 3.2 Statistics

- **The bootstrap** is Tier 1's `t1_stats.Boot` (imported, never copied): 2,000 draws, `default_rng(20261011)`,
  percentile 95 % intervals, battle-clustered, every contrast paired on the same draws.
- **Pooled** = battles resampled WITHIN each checkpoint, states weighted equally.

### 3.3 Rule 8: the band, resolved before any data

- **The band is ±0.0005** (`t1_stats.EDGE`). A bound "> 0" is clear only above +0.0005, and "< 0" only below −0.0005,
  on the numerators (p-units) and on the shares alike.
- **Why ±0.0005, and not ±0.005:**
  1. **It is the band of the rule E2 tests.** Tier 1's registered R3 was judged at ±0.0005, and so was the third
     party's primary reading. E2's verdict must be commensurable with both.
  2. **The ±0.005 of the E1 pre-registration was a different thing.** It was a cross-implementation POINT-agreement
     tolerance (|Δ| < 0.005 on σ_U, ν_U between two builds). That is a materiality bound on a difference between two
     estimates of one quantity, not a significance band. Read as a band, it would re-write the original's registered
     R3 (`third_party_comparison.md` finding 3).
  3. **Rule 8 guards against a rounding error at a decision boundary.** The IMPs carry about 1e-5 of arithmetic noise
     (Tier 1's identity check: max |ΔV| 1e-5). ±0.0005 is 50 times that.
- **Materiality is reported separately, never folded into detection.** A descriptive **MATERIAL** flag on a numerator
  means its lower bound is above +0.005 (or its upper bound below −0.005): at least half a point of win probability per
  decision.

### 3.4 The decision rules (PRIMARY: pooled end2 + prod, LEAF-K8 with the registered tie rule)

**LEAF-K8:** the mean of the 8 world outcomes; ties within 1e-6 go to the first action in the policy's order. Evaluate
in this order:

1. **U-FIRST** iff lower(Δs) > +0.0005 AND lower(Δn) > +0.0005. This is R3 met with X = U.
2. **M-FIRST** iff upper(Δs) < −0.0005 AND upper(Δn) < −0.0005. The moves dominate the never-seen mons on both sides.
3. **SHARED** iff neither of 1–2, AND **both classes are detected**. A class X is detected iff lower(σ_X) > +0.0005
   or lower(ν_X) > +0.0005.
4. **NOT DETECTED** otherwise: no ranking, and at most one class is detected. The report names which class, if any.

**Qualifiers, printed beside the verdict (descriptive):**
- **Each side's direction:** U, M, or none, by its clear bound. SHARED is labelled **SIDES-DISAGREE** when one numerator
  is clear positive and the other clear negative.
- **The MATERIAL flags** (§3.3).
- **CLOSE:** both numerators' intervals lie inside ±0.01 (rule 8: lower > −0.0095 and upper < +0.0095).
- **GAP:** lower(G) > +0.0005. If the gap is not detected, the verdict still stands, because the numerators need no G.
  The shares are then reported as uninformative.

**What each verdict means for the study (fixed now):**
- **U-FIRST.** The never-seen Pokémon are the largest class on competent play too. The red team's population attack does
  not replicate on fresh states. "U first" then holds on both populations.
- **M-FIRST.** On competent play the seen mons' moves carry more than the never-seen mons. Tier 1's R3 is
  population-specific. Whether the moves are THE binding constraint still needs B3 / B4 on this bank: s_M, n_M, B3 and
  B4 are reported, descriptive here, never a new verdict on R-MOVES.
- **SHARED.** Both carry detectable value, and neither dominates on both sides. "U dominates" does not extend to
  competent play.
- **NOT DETECTED.** This bank cannot rank the classes. Its interval widths are reported, so the next bank can be sized.

## 4. SECONDARY reads (each reported; none overrides the primary)

1. **Per checkpoint.** The same rules on end2 alone and on prod alone. The end2 verdict is quoted beside the pooled one
   because end2 is the production lineage.
2. **The TIE-FREE scorer (the scorer-dependence check).** The same rules under **LEAF-K8|RAND**: ties broken uniformly at
   random, as the exact expectation of T over the tied set (`red_team.py` `scorer_imps`).
   - When it and LEAF-K8 give different verdicts, the primary verdict is printed with **SCORER-DEPENDENT** beside it.
   - Reported without a verdict: LEAF-K8|CRITIC, LEAF-K8|TRUNC4, TRUNC-4-K8, CRITIC, LEAF-K4 and LEAF-K1, each with Δs,
     Δn and the R3 conditions. The agreements of `third_party_comparison.md` held at LEAF-K8 and flipped at K4 / K1;
     this makes that dependence visible.
3. **The shares:** G, s_M, n_M, s_U, n_U, s_U − s_M and n_U − n_M (the registered §12.3 form), and B3 / B4 for M.
   - All are descriptive. The §6.1 10 % rule applies to F: above 10 % F exclusions, every F-based share is flagged as
     read on the included states.
4. **Against Tier 1's populations (descriptive).** E2's Δs and Δn minus the same numerators on Tier 1's bot-game states
   and on Tier 1's snapshot-game states (registered 600 + supplement, the red team's classification by the
   `cycle_opp` prefix).
   - These are independent samples on disjoint battles, so the bootstrap draws are independent: a difference of the two
     draw vectors.
   - Also reported: a k-adjusted E2 Δs / Δn (per-state numerators regressed on the never-seen count and the hidden move
     slots, as `red_team.py` §E / §H).
5. **By k bin** (1–2, 3, 4, 5, 6) and by traced outcome: Δs and Δn, descriptive.

## 5. Exclusions, validity and stopping rules (mirroring Tier 1, with the guard on)

- **An exclusion** is a state the builder cannot fit (a `WorldError`). It is recorded as a row with `ok: false` and its
  reason. It removes the state from the contrasts involving that arm only, and it is counted per arm.
- **A world-guard raise** (`WorldConsistencyError`) is NOT an exclusion: it is a defect. The build driver records it as a
  `crash` row, the state leaves that arm's contrasts, and every such state is listed as a FINDING with its cause.
  - **If guard raises exceed 1 % of any arm's states,** the read is computed but quoted only as "computed, guard
    defects open" until the cause is root-caused.
- **A state without a Phase 1 row** (truth refused) leaves every contrast, because the referee is Phase 1. Counted.
- **The k = 6 identities must hold**, else the build is INVALID and nothing is read: on every k = 6 state, F+U ≡ F and
  O−U ≡ O outcome for outcome (no never-seen mon exists, so the arms must equal their bases).
- **The cross-commit identity must hold:** `oracle` worlds built at the E2 HEAD and played at the pad pin reproduce
  Phase 1's outcomes exactly on 12 states per checkpoint. Values must agree within 1e-5 (Tier 1's `identity_read.py`
  check).
- **The pin re-gate (descriptive, as Tier 1):** every world re-gated at the pad pin's own `core_events`, rows and
  protocol byte-equal to the true prefix. Failures are counted.
- **Size rule:** none. 600 per checkpoint is fixed. E2 does not extend the bank after reading any outcome.
- **Operations (no read):** the runner prints no outcome. It reports progress as counts of done rows only.

## 6. CPU (root rule 5b; the X26 run is live)

- Every E2 process runs under `nice -n 19` (and SCHED_IDLE where the Tier-1 runner used it), pinned with
  `taskset -c 5-7,13-15`, each inside its own `scripts/ops/mem_cap.sh` scope, with `OMP_NUM_THREADS=1`.
- At most 2 capped jobs run at once. No GPU (`CUDA_VISIBLE_DEVICES=""`).
- The red team's estimate is ≈ 40 CPU-h for end2 and ≈ 65 for both. The pin-side cost is measured and reported.
