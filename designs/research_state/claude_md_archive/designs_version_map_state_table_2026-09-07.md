# The designs/ state table — the run and chapter cells, 2026-09-07

> **This directory is HISTORY — additive only.** A file here is written once, when narrative is
> lifted out of a `CLAUDE.md`, and is never updated afterwards; when the world changes, the
> `CLAUDE.md` changes and this file stays as the record of what was believed then.

Lifted verbatim from `designs/CLAUDE.md`'s **Current state** table on **2026-09-07**, when the
table was cut to one status line + pointer per row. These are the four remaining long cells: the
live arm and the generation history behind it, and the ai_v12 / ai_v10 / ai_v11 chapter cells.

🚨 **Nothing here is current** — the live arm moves, the generations behind it are finished, and
the chapter status lines in `designs/CLAUDE.md` are the maintained ones.

---

## Active training run

🟢 **`ai_v12_02_winprob_critic` — the first VALID ai_v12 arm, relaunched 2026-09-06 20:27** (its
predecessor `ai_v12_01_winprob_critic` ran ~7 h on a STRIPPED architecture and is dead — ledger
`81016942`; the first `ai_v12_02` launch at a 4096 micro-batch OOMed at iteration 1 — ledger
`ce43b4e9`, dir kept as `ai_v12_02_winprob_critic.OOM_4096`)

**FRESH WEIGHTS** (`lineage.role = "fresh"`, `fork_parent: null` — the era has no warm start),
config **v110** at launch, `arch_signature` `gen3_critic_route_wave_v1`, pinned by `--pin-commit` to
**`f971caf2`**, `--steps 75000000` (~33 h at the full architecture's measured rate), `--n-envs 48
--batch-size 2048 --grad-accum-steps 32 --n-epochs 10 --n-steps 2048 --lr 0.0003 --ent-coef 0.02
--device cuda --self-play`. **The clean-world arm on the PRODUCTION architecture surface (49 derived
toggles diffed against `production_config.json`, 0 differing) plus
[`ai_v12/design_winprob_only_critic.md`](../../ai_v12/design_winprob_only_critic.md) §5.4's critic
block:** `--critic winprob --no-hand-shaping --terminal-indicator --victory-value 1.0 --draw-penalty
0 --vf-coef 0.5`, recorded as `critic: "winprob"`, `hand_shaping: false`, `use_popart: false`,
`value_from_dist: false`, `value_dist_mode: "none"`, `victory_value: 1.0`. It is the **SPARSE** rung
of the runbook's SPARSE / SELF-φ / FROZEN-φ ladder (there is no potential: under this critic both
`--win-prob-pbrs-*` are refused, not merely unset). **It is the only run in the archive whose config
records `critic: "winprob"`** — every other run is `shaped` by migration. 🚨 **Read the stall rate
and mean episode length as PRIMARY endpoints, not monitored ones**: a critic bounded in [0,1] cannot
represent "a timeout is worse than a loss", so the `−35 < −30` ordering is unrepresentable and all
the anti-stall pressure comes from the obs deadline clock plus `--arm-no-progress-tax` (NOT passed
on this arm). Immediately prior, and complete: the **G5 continuation controls**
`ai_v9_195/196/197_G5PLAIN{A,B,C}_0906` (config **v107**) — *our parent gains NOTHING from a plain
continuation*, **−1.92pp [−3.98, +0.46]**, so the frozen-parent baseline stands on our side while
the same cell on v8's 277M parent read **+3.45pp [+0.46, +6.48]**. `designs/production_config.json`
still mirrors **config v97** (`ai_v9_21_gen17_pfspoff_0820`, the last full pre-ai_v12 generation)
and therefore predates the v98–v109 fields — it is the drift gate's reference during a bump window,
not a description of the newest runs, and `arch_tables --check` is green against it. ⚠️ It also
still carries `threat_prob_outspeed: false`, a key **deleted at v108**. Historical:

**gen-14**

`ai_v9_16_gen14_framedel_0817` (launched 2026-08-17, config **v90** `gen3_frame_deletion_v1`, FRESH
WEIGHTS — the signature bump forbids a warm start; pinned to `fe910ee`). **One behavioural change:
the 7×159 TurnDelta lag frames and the 11-dim prev-turn action mask are DELETED** (obs 3529 → 2437),
licensed by gen-13.5 §4 (`event_seats` dV 2.7714 vs `frames` 1.3015). Flag delta off gen-13:
`--value-intent` DROPPED (dV 0.1560); `--intent-value-reduce` (0.3826) and `--value-clock` (0.3370)
KEPT ON deliberately — they sit in the registered TIE-BREAK zone and must stay live to be re-audited
at ≥2× sample. Its battery is `designs/research_state/gen14_endofrun_runbook.md` (pre-registered
BEFORE launch), and its open reconciliation is
[`ai_v9/design_frame_deletion_coverage_gaps.md`](../../ai_v9/design_frame_deletion_coverage_gaps.md). The
C5 TD-aux control fork `ai_v9_16_c5fork_control_0817` (27.1M, complete) is the banked baseline for
the λ arms, which feed **gen-15**, not gen-14 — the frame deletion rides alone. Predecessor gen-13
`ai_v9_15_gen13_hb_events_stack_0817` (25.07M, config v89). Earlier: gen-12
`ai_v9_14_gen12_h_entitypool_shaping_0816` (launched 2026-08-16, config **v80** at save time; a
pre-floor config — v90 raised `MIGRATION_FLOOR` to 90, so it no longer migrates and is read from its
own git_hash). Four changes off the gen-11 base: the `h` pair-history edge family ON,
`--value-entity-pool` ON (the Stage-3 critic pool trains live), `--intent-value-reduce` ON, and the
win-prob shaping REVERTED to `--win-prob-coef 0.05` (the gen-11 verdict: ELO tied but the species
belief regressed under the heavier shaping). Its end-of-run battery is `python -m main.endofrun
models/ai_v9_14_gen12_h_entitypool_shaping_0816 --ref models/ai_v9_13_gen11_labelonly_winprob_0815`.
Predecessor gen-11 `ai_v9_13_gen11_labelonly_winprob_0815` (launched 2026-08-15, config v77, pinned
to its own commit — NOT `--sync-to-main`); its two changes off the gen-10 base: `--belief-grad-mode
label_only` (the belief heads are trained by their SUPERVISED LABELS ALONE — no policy/value
gradient reaches them; their trunk read stays live so the label loss still shapes the trunk) and
`--win-prob-mode shaping --win-prob-coef 0.05` (the win-probability side readout now also shapes the
trunk). Predecessors: gen-10 `ai_v9_12_gen10_t0prior_0814` (the T0-species-prior arm) and
`ai_v9_11_gen10_intentfull_compiled_0814` (its A/B partner without the prior). ⚠️ Both were launched
with a FULL explicit flag dump, so their `original_command` passes flags v78 deleted (`--zarch-*`,
`--seed-quantile-coef`, `--value-seed-vicreg-coef`, `--film-grad-accum-steps`) — a `--sync-to-main`
resume of either needs those stripped from the command first; a normal (git-hash-pinned) resume is
unaffected. Earlier: gen-9 `ai_v9_10_gen9_intent_distcritic_0813` (the intent +
distributional-critic arm), gen-8 `ai_v9_09_gen8_beliefs_threat_inject_0811` (beliefs +
threat-inject; live sparse eval 2087±31 @26M, offline tail-4 ladder read it BELOW gen-4/5 —
sparse-vs-dense unresolved), gen-7 (seed-quantile arm — quantiles learned, rank 1.157/4 ⇒ seed line
closed), gen-6 (seed-VICReg arm — every term satisfied, rank ~1 ⇒ repulsion refuted), gen-5
`ai_v9_06_gen5_no_concat_0809` (25M, concat deletion at ELO parity with gen-4's 2096), gen-4
`run_20260808_212910` (v60 entity re-home; its stratified audits justified the concat deletion),
gen-3 `run_20260807_135637_gen3` (40M, 15 edge families, ELO 2094@32M), gen-2, gen-1. The old ai_v8
lineage sits behind the ai_v9 signature wall.

## ai_v12 — 🟢 **THE LIVE CHAPTER — code landed, ARM 1 RUNNING**

The **clean-world / win-probability-critic** chapter, opened **2026-08-29** (`7d1a8517`). Where
ai_v9 is the entity graph inside one battle, ai_v10 what transfers between teams and ai_v11 what an
external action distribution teaches, **ai_v12 is what the win-probability head becomes when it
stops being a barometer**. Two designs and a runbook. (1)
[`design_winprob_behavior_coupling.md`](../../ai_v12/design_winprob_behavior_coupling.md) — the plan of
record, three routes turning the head into behavioural force, all **BUILT and OFF** (route 1
`--win-prob-pbrs-coef`, v104; routes 2+3 `--search-teacher-mode winprob_oneply`). Its probe **L** is
what fires the chapter: the head ranks an alternative above the played action on **96.4% of immune
whiffs** (+0.213 over the tightest control, dice-invariant) while the policy samples that
alternative at a median **p = 0.002** — *the head KNOWS and the policy does not act on it*. (2)
[`design_winprob_only_critic.md`](../../ai_v12/design_winprob_only_critic.md), the **DESIGN OF RECORD,
landed 2026-09-06** (`b242e2e3`), implemented the same day as **`gen3_winprob_critic_mode_v1` /
config v109** (`cbcb0bfb`) in its stated §6 order — A2 census, then B2/B3, then the mode.
**`--critic {shaped,winprob}` exists, is tested, and DEFAULTS to `shaped`**, so nothing about
today's runs changed; the default flip, the `ARCH_SIGNATURE` bump it forces, and §5.3's deletion
list are a LATER commit, **after an arm has run**. (3)
[`launch_runbook.md`](../../ai_v12/launch_runbook.md) — the terminal-only ladder registered
**2026-08-29/30** (ledger `e22bd08` three-arm ladder · `627ab58` no launch bias · `cfbc9bf` draw =
−1 · `2d38a4a` PopArt retirement · `db9bb5c` the V_shaped-constancy prediction · `4d22ae4` the
pure-sparse control + 5M pre-test · `132d198` wave A's landed flag surface): three generation-scale
arms **SPARSE / SELF-φ / FROZEN-φ**, identical but for where the potential comes from, all at `{win
+1, loss −1, draw −1}`, ahead of them a paired 5M pre-test that sizes the full runs in GPU-hours
rather than generations. Every argv in it passes `main.checkargs` at exit 0 and has run as a
`--debug --steps 8000` CPU smoke; `src/main/launch_runbook_test.py` parses the blocks OUT OF the
document through the live parser, so a flag deleted anywhere fails a test naming this doc. Also
here: [`probe_risk_modulation_capstone.md`](../../ai_v12/probe_risk_modulation_capstone.md) (owner-ordered
2026-08-30 — does a P(win) value function buy correct risk modulation? three offline instruments
with **frozen per-arm predictions**, and a FLAT sparse slope falsifies "P(win) buys risk for free"
and must be reported as loudly as a pass), the 40-team slate (`team_slate_40.{json,md}`,
`team_slate_build.py`) and `promotion_exclusions.json` (the exclusion list `main.promote_teams
--regenerate-exclusions` rebuilds from run metadata — it was first built from FROZEN ARGVS and went
stale on all three rev-4 arms **with its union SIZE unchanged**, so no count-shaped check saw it).
**ARM 1 IS LIVE** — `ai_v12_02_winprob_critic`, the production surface plus §5.4's critic block,
fresh weights, pinned to `f971caf2`, relaunched 2026-09-06 20:27 at 2048×32 (see the *Active
training run* row); `ai_v12_01_winprob_critic` is the dead stripped-architecture arm and is not
evidence about anything but the launch guard. **The era boundary the default flip implies is
CENSUSED, NOT DECIDED**:
[`research_state/era_boundary_deprecation_2026-09-06.md`](../era_boundary_deprecation_2026-09-06.md)
prices it — which flags become deletable, which of the **112 currently-loadable runs** stop loading,
what an era checkout still buys, and the ordered commit list with a gate per commit. Nothing in it
is a decision; §5.3 of the design doc is the deletion list, and the flip waits on this arm.

## ai_v10 — **OPEN — nothing built**

The **exploiter-SCALING** chapter, opened 2026-08-16: why competence collapses between N=10 and N=20
teams when N=1..5 is trivial. One doc,
[`ai_v10/design_exploiter_scaling.md`](../../ai_v10/design_exploiter_scaling.md) — the hypothesis (**no
transferable team-scoped abstraction ⇒ sample cost linear in N**), the four competing accounts it
must beat (H_rate / H_capacity / H_conflict / H_coverage), and a **pre-registered, unrun** test
battery whose Tier 0 needs no extra GPU. Two NEW gen-12 measurements carry it: "it's just 6 1v1s" is
**REFUTED** (bench→MOVE logit ratio **0.262**, n=4058 over 8 real exploiter teams) but the profile
is FLAT, so the bench enters as a **SCALAR not as structure**; and team PACE class decodes from the
raw obs at **0.456 on UNSEEN teams** while `pi_features` **0.211** and `value_pooled` **0.199** sit
at chance (0.200) — **the abstraction is free in the input and the trunk discards it**. Where ai_v9
is the entity graph *inside* one battle, ai_v10 is *what transfers between teams*. The chapter also
carries two owner-era operational/forward docs:
[`design_flywheel_tick_tock.md`](../../ai_v10/design_flywheel_tick_tock.md) (the exploiter–generalist
loop, decisions of record — needs refinement, not implementation-ready) and
[`design_outcome_latent.md`](../../ai_v10/design_outcome_latent.md) (FORWARD, 2026-08-19: the per-action
LEARNED outcome latent — route-3 delivery with learned content, the Spikes mechanism/horizon
factorization, the G0→R3 ladder gated on behavioral deltas, and the richness-pressure menu ranked by
this codebase's own body count — post-gen-16, unscheduled). A third forward doc,
[`design_counterfactual_value_grounding.md`](../../ai_v10/design_counterfactual_value_grounding.md)
(2026-08-22): the counterfactual label factory + the three reroll-based attacks on CRITIC BIAS (R1
tight-MC re-labels on visited states / R2 MC labels on counterfactual successors — the
optimizer's-curse interruption and the bait cure claim / R3 k-step grounded targets), priced by the
2026-08-21 probe triad (162→28.4→~7.7 ms/label; opponent branches first; the prefix-sharing
materializer is the one build item), gated G0–G4 on bias METERS not loss curves — the pre-registered
new mechanism ledger C6 requires, attacking the critic's TARGETS while its closed delivery line
stays closed. A fourth forward doc,
[`design_advantage_gated_distillation.md`](../../ai_v10/design_advantage_gated_distillation.md)
(2026-08-25): the DEEP-BRANCH fix for the exploiter fold, ordered by the ARM E verdict. Five
code-matched +3M arms plus tick-1 eliminated ecology (fdC null), coefficient (12.50 at 0.3 *and*
1.0), the state-gate (fdE hard-gated ≡ fdB) and teacher count — leaving `pi_features` **BINARY:
21.87 with no KL, 12.5–13.6 with ANY KL**, with IN-gate and OUT-gate states both damaged. Every arm
moved *where* or *how hard*; none moved **what the KL asks for**. The doc separates TARGET FORM
(action-level CE / top-K vs full-distribution KL — the one axis never manipulated, and the one that
formally contests flywheel **D-F**) from JUDGE (the student's own advantage SIGN, free, vs
paired-CRN rollouts, priced in its §A at ~580–1,330 usable verdicts/hour and therefore deferred to
flywheel cadence), makes a `rank/policy_pr` tripwire a first-class element (20% drop = fires on all
five bad arms, on none of the controls), states how each arm-F outcome edits it, and pre-registers a
dose-matched G1/G2 arm pair against the already-run fdC/fdB.

## ai_v11 — **OPEN — nothing built**

The **human-ladder-replay** chapter, opened 2026-08-18: what we can learn from an EXTERNAL action
distribution, and what survives the fact that spectator replays are **partial information**. One
doc, [`ai_v11/design_human_replay_objectives.md`](../../ai_v11/design_human_replay_objectives.md) — the
**OOD taxonomy** (our own team is fully known live and only partially recoverable from a replay; the
request stream does not exist, so the mask is synthesised and systematically over-permissive) and a
**pre-registered, unrun** four-rung objective ladder ordered by OOD-robustness: α/β on the human
OPPONENT's actions (robust — that half of the obs is hidden live too) → outcome/value on human
states → BC-regularization on the faithful subset (**gen-17 candidate**) → offline RL with
team-completed acting sides. Its Phase-0 census is **RUN** (`tmp/replay_faithfulness_census.py`;
263,159 logs / 2.8 GB / 2026-05-18→2026-08-02): tier-A (6/6 bench, 4/4 moves) is **16.70%** of
30,146 ≥1500 decisions, own **item known on 3.93%** of own mons, own **spread is FABRICATED and
flagged `spread_known=1`** (all-31 IVs / 0 EVs / neutral nature — a wrong value asserted as known,
feeding `d1`/`d2`), the faithful stratum is **loss-enriched 1.29×**, α-label pairing is **92.04%**,
and the human switch share reproduces model-free at **28.96%**.
