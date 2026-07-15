# The generalist–specialist amortization gap (why our self-play net won't learn per-team play)

**TL;DR.** Our self-play generalist can *see* its own team (the obs carries the full 6-mon
"Our team" block) and the trunk attends over it — so its inability to pilot a specific team
expertly is **not** an obs or architecture bug. It's the **amortization gap**: one policy trained
on a *distribution* of ~700 teams under an *averaged* objective, with a *tail-blind critic* and no
architectural pressure to specialize, converges to a coarse team-**agnostic** policy that plays
every team adequately and none expertly. A specialist exploiter beats it not with a better team but
by being a **dedicated per-team solution** (a fine-tune) that never has to *condition* at all.
Measured proof it isn't conditioning: on the free axis (switching) behavior is flat ~30% across
archetypes; the equal-pilot mirror shows the generalist pilots a canonical SkarmBliss team to only
0.375 vs the pool. The fix is credit-assignment / architecture / curriculum / distillation — not a
wiring fix.

---

## Intuitive level

Two chess players. One has drilled a **single opening** 10,000 times and knows its deep 20-move
plans cold; the other plays **every opening** competently but shallowly. Hand the specialist her pet
opening and she crushes the generalist — not because her pieces are better, but because she is a
*dedicated solution to that position*.

Our generalist is the second player. One network, one set of weights, must pilot SkarmBliss balance,
hyper-offense, trap, CM-pass — everything in the pool. In self-play its gradient is an **average
over all those teams**. The update that would make it a patient SkarmBliss expert ("Spikes turn 1 →
preserve Blissey → Toxic-stall over 30 turns") is neutral-or-wrong for a hyper-offense team that just
wants to click its strongest attack. Averaged across the team distribution those specialist
directions **cancel**, and what survives is the team-agnostic "play a locally sensible move" policy.

The exploiter has a far easier problem: it pilots **one** team every game. It never has to recognize
"which archetype am I?" and switch plans — it simply *is* the SkarmBliss specialist. All its capacity
and every gradient step point at one team's win-conditions. That's why it executes the balance
win-conditions the generalist never does.

## Technical level

Three mechanisms stack; none is a wiring bug.

**1. Self-play averaging → the fixed point is the task-*marginal* optimum.** An amortized policy
π_θ(a|s) trained on a task (team) distribution p(T) optimizes E_{T~p}[J_T(θ)]. Its optimum is
generally **not** the set of per-task optima {argmax_θ J_T} — it's the compromise that is "okay
everywhere." Vanilla PPO self-play never *forces* per-team experts, so it doesn't build them. This is
the amortization gap: best-response (fine-tuning on one task) beats the amortized policy on that task
by construction.

**2. Long-horizon team plans need a critic that can credit them — ours can't.** Reinforcing a 25-turn
balance line requires the critic to assign rising value to *patient positional* moves (Spikes,
Protect-to-scout, Toxic-then-stall) that pay off many turns later. Our critic is **tail-blind**
(`td_resid_tail` stuck negative across runs; loss-analysis traces losses to critic tail-blindness on
recovery/attrition coverage). If the critic under-values the patient line, the **policy gradient
never credits it**, so the policy defaults to greedy immediate-value moves — precisely the "adequate,
no long game" behavior.

**3. The representation has collapsed coarse.** A rank probe found the 128-dim trunk runs in only
**~3–5 effective dimensions** — far fewer than a policy conditioning on *(my archetype × opponent
archetype × board)* would need. The coarse "who's ahead + immediate threat" summary is *sufficient*
for the averaged objective, so that's where it settles. *(Open/unverified: the rank number is from
the ai_v7_02-era trunk, not re-measured on ai_v7_14 specifically; _14 shares the architecture.)*

### Measured evidence it isn't conditioning (not theory)

- **Equal-pilot mirror** (`tmp/team_matchup_mirror.py`, 2026-07-14): ai_v7_14 pilots a canonical
  SkarmBliss team to only **0.375** vs the pool at equal skill — it genuinely pilots balance *poorly*
  (trap 0.483, CM-pass 0.383). See `project_exploiter_no_team_advantage`.
- **Archetype competence gradient:** behavioral differences across teams are **mostly
  move-availability** (a stall team can't click a sweep move → the move-mix differs *mechanically*),
  but on the **free axis — switching — it is flat ~30% regardless of archetype.** Where the model is
  free to express team-specific strategy, it does the same thing on every team: one averaged policy
  filtered through each team's legal moves, not genuine conditioning. See
  `project_archetype_competence_gradient`.
- **Turn-1 P(win) stratifies** (offense 0.69 / stall 0.55 / balance 0.46): systematically weakest on
  the patient archetypes it never learned to pilot.

## Where this lives in our architecture

- **Team is in the obs — no fix needed there.** `OFFSET_OUR_TEAM` → the "Our team (6 × 110)" block
  (`src/agents/observation/state_encoder.py`). The 6 our-team tokens pass through `TeamTransformer`
  alongside the 6 opp tokens into the CLS pools (`src/agents/model/features_extractor.py`), so both
  the input and the attention path to *condition* already exist.
- **Critic-side levers (the credit-assignment half):** `--value-dist-mode` (v29 `ValueDistHead`), the
  tail-weighted value loss (`--value-tail-weight`), and the search-teacher AWR line
  (`project_search_teacher`) all target valuing patient/long-horizon lines.
- **Specialist-side levers:** `--exploiter` is a per-team fine-tune by construction; the league
  fold-back and on-policy self-distillation (`--opd-coef`, `on_policy_self_distillation.md`) inject a
  specialist's conditioning back into the generalist.
- **Data for a real fix exists:** `data/teams/gen3_team_archetypes.json` labels every pool team by
  archetype — raw material for archetype-conditioned routing or archetype-aware sampling, neither
  built yet.

### Four levers to close the gap (increasing ambition)
1. **Fix the critic (deepest).** The **tail-weighted value loss** (`--value-tail-weight`) + the
   search-teacher, so patient team-specific lines get valued in the *scalar* critic — otherwise the
   gradient *can't* reinforce a long game. NOTE: the **distributional value head** (`--value-dist-mode`,
   `ValueDistHead`) is NOT this lever — it is a side readout, never in pi/vf or GAE; even in `shaping`
   mode it only backprops as a representation-aux. The value PPO credits is the scalar mean V and the
   reward is scalar per turn, so the distribution does nothing for *credit assignment* (see the
   distributional caveat below).
2. **Architectural conditioning.** An archetype embedding / FiLM / mixture-of-experts head keyed on
   team composition, so the net *routes* to a per-archetype sub-policy instead of amortizing.
3. **Curriculum / DRO.** Up-weight the teams it pilots *worst* (balance/stall) so it can't hide
   mediocrity behind the offense teams it's good at — PFSP, but over our own teams.
4. **Distill the specialists back.** League fold-back: each exploiter's per-team policy distilled into
   the generalist — literally injecting the conditioning it won't discover alone.

## How to actually make it *want* per-task optima

The key unlock: **the gap is an RL-objective artifact, not a representational limit.** Averaged
self-play cancels specialist gradients — but **supervised multi-task learning does not.** Give the
net a *per-team target* ("on team-T states, output specialist_T's distribution") and the loss
explicitly demands matching each specialist; there is nothing to average away. Conditioning stops
being something the policy must *discover* under a weak reward and becomes a classification the
network is good at. So the highest-ceiling path is **distillation, not cleverer self-play**:

1. **Distill specialists back as per-team *teachers* (primary lever).** Fold each exploiter back
   NOT as an opponent but as a teacher: when the generalist pilots team T, add a KL/BC term pulling
   its policy toward exploiter_T on those states. The specialist already solved the credit
   assignment; the generalist just copies the mapping. This is AlphaStar-league / **Distral**
   ("distill & transfer"). We have the machinery — `--opd-coef` (KL-to-teacher), search-teacher AWR,
   fold-back plumbing — but today fold-back wires exploiters as **adversarial opponents** (harden _14
   to *resist* them), the OPPOSITE of the **teacher** wiring (make _14 *play like* them). The dream is
   the teacher wiring. *(Open: teacher-side fold-back is not built; the opponent-side is.)*
2. **Routing capacity — only if interference bites.** Distillation can hit an *interference* wall
   (specialist targets fight over shared weights and re-average). Fix = archetype FiLM / MoE routing.
   But the ~3–5-dim rank collapse says capacity is NOT the current bind → try plain distillation
   first; add routing only if per-team WR plateaus below the specialists.
3. **Per-archetype advantage normalization.** PPO advantage-norm (and PopArt) use GLOBAL batch
   statistics. When archetypes differ in *variance*, the pooled std is dominated by the loud, swingy
   archetype (hyper-offense, big |A|), so dividing everything by it shrinks the QUIET archetype's
   advantages (stall/balance) to near-zero → they get the weakest gradient and are learned slowest — a
   concrete driver of the observed competence gradient. Fix: normalize advantages (or PopArt) WITHIN
   each archetype so each contributes gradient of comparable scale. Caveat: only helps if the variance
   heterogeneity is real — MEASURE per-archetype |A| / return variance first.
   - **Soft, self-learned archetypes (the better design).** Hard 5-bin labels are lossy — teams are a
     spectrum. Use a **soft-gated mixture-of-experts** (or a learned continuous "style" latent off the
     6-mon set): the gate reads the team, outputs a soft weight over experts, and the policy is the
     weighted BLEND (a 60/40 team → 0.6/0.4 mix). Experts can DISCOVER their own factorization
     end-to-end, so `gen3_team_archetypes.json` becomes optional (gate warm-start / per-archetype norm
     only). Catch: MoE reduces INTERFERENCE but the gate still needs a *reason* to differentiate — it
     can collapse under pure averaged self-play. So it composes with distillation: teacher = the
     specialization pressure, MoE = the non-interfering capacity, soft gate = the blend. Distillation
     is the engine; MoE is the transmission.
4. **The loop that reaches the dream — expert-iteration / PSRO.** Iterate: (a) train fresh exploiters
   vs the current generalist, (b) distill them back, (c) repeat. Each round absorbs the latest
   specialist knowledge; fresh exploiters must find NEW weaknesses. Convergence (fresh exploiters
   can't beat it much) IS "generalist ≈ every specialist."

### Scaling distillation to N archetypes (the cardinality problem)

One exploiter = one team, but there are ~700 pool teams / infinitely many. You do NOT train one per
team. **Coverage is a sampling problem, not an enumeration problem; generalization is the STUDENT's
job.** The generalist is one function conditioned on team features — feed it `(team → specialist
action)` for a sparse *covering set* and it interpolates to the teams between (like any supervised
generalization). Levers that shrink the covering set + make it work:

- **Exploiter per ARCHETYPE, not per team.** Pin the exploiter to a *distribution* of teams within an
  archetype (widen `--trainee-team` from one file to an archetype sample; `gen3_team_archetypes.json`
  defines the sets) → it becomes a mini-generalist over that sub-distribution and distilling it covers
  the archetype. N drops to ~5–8. Within-archetype averaging is MILD (strategies align: all balance
  wants spikes+recovery+status+pivot); the gap only bit across *opposed* archetypes. Curriculum trick:
  train on ONE team first (clean discovery/exploration), then WIDEN to the archetype (generalize the
  skills), then distill.
- **JOINT multi-teacher distillation (the "how for N" mechanism).** Do NOT distill teachers
  sequentially — later ones overwrite earlier (catastrophic forgetting). Every batch mixes states from
  several teams, each supervised by ITS OWN archetype-teacher (KL/BC), with **per-archetype loss
  normalization** so none dominates. The student learns all N mappings at once — this is where the
  dense-trunk superposition capacity is spent (N sparse teacher-programs in one trunk). Needs the
  teachers kept in a small snapshot pool (we have that machinery).
- **Blending = learned soft conditioning, never hand-assigned.** A 60/40 team's policy lands *between*
  the offense/balance regions the student learned. Mixed teams: either pure interpolation (distill on
  archetype-pure teams, generalize) or a membership-weighted **soft-mixture target**
  (0.6·π_off + 0.4·π_bal — a prior, not a theorem: the hybrid optimum isn't provably the mixture). A
  soft-gate/MoE makes the weighting explicit + learnable, but optional (dense trunk can do it implicitly).
- **PSRO makes cardinality self-solving.** Iterate: train generalist → train an exploiter whose team is
  SAMPLED to maximize exploitation (gravitates to the worst-covered region) → distill jointly → repeat.
  The loop DISCOVERS which archetypes need coverage instead of you enumerating them, and signals done
  when no exploiter can gain anywhere.

Open problems: mixed-team distillation target (mixture-of-policies not provably optimal); forgetting
(mitigate with joint-not-sequential distill + snapshot pool); teacher-mode fold-back not built (today's
fold-back is opponent-mode; `--opd-coef` already does KL-to-frozen-teacher, so the missing piece is
per-team teacher *selection*, not a new loss).

### The simplest MVP (one teacher, one team) + how the aux loss works

**MVP needs NO new exploiter — the specialists already exist on disk** (`ai_v7_10`, `ai_v7_15`). Test
"can one dense trunk absorb a specialist without breaking the rest" on N=1:
1. Resume generalist training (fork _14).
2. Bias team sampling toward `tss_starmie.txt` (~30–50% of episodes); the rest of the pool is the
   REHEARSAL that prevents forgetting.
3. On TSS episodes, add the distillation aux loss toward the frozen TSS exploiter; non-TSS episodes =
   plain PPO.
4. Measure: TSS-piloting WR (should climb toward the exploiter) AND bots/ELO/non-TSS WR (should HOLD —
   the whole experiment is this no-forgetting check). Rest regresses ⇒ interference wall found cheaply
   ⇒ *then* per-archetype norm / soft gate earn their place.

The co-evolution (autocurriculum): MVP = round 1. Re-train an exploiter vs the improved generalist — the
easy 0.78 line is patched, so it must find HARDER lines; distill those; repeat. Each crank forces both up.

**How the distillation aux loss works (policy, NOT value):**
1. Standard PPO rollout — the student plays, collecting states/actions/rewards/advantages as normal.
2. Per state, ONE extra forward of the FROZEN teacher → `π_teacher(·|s)` (logits over the 11 actions,
   masked to legal); teacher is inference-only, no grad.
3. `L = L_PPO + β · KL(π_teacher(·|s) ‖ π_student(·|s))`, backprop student only. β = distillation coef.

Key points: **on-policy** (teacher queried on the STUDENT's own visited states) is why it beats plain BC
— no compounding distribution shift. Forward KL = mode-covering (student reproduces the teacher's full
move distribution). Cost = one frozen forward per state (~the self-play-opponent cost). **AWR is the
variant that uses the teacher's chosen ACTION + a value/advantage weight** (what search-teacher does) —
richer when the teacher only gives one action; full-distribution KL is simpler/better when the teacher is
a queryable policy (an exploiter is). **Do NOT distill the value head** — the teacher's value is on its
own PopArt/opponent scale and just fights your critic; policy-only KL, value stays your own PPO.

**We mostly have it:** `--opd-coef` already implements this KL-to-frozen-teacher aux loss; today the
teacher is a search beam. The MVP change = swap the teacher source to "frozen exploiter forward on this
state" (CHEAPER than the beam), gated on the team being piloted. *(Confirm the exact loss wiring in code
before building — machinery is there.)*

### The concrete PSRO loop for us (weakness-finding + covering sets)

**PSRO (Policy-Space Response Oracles, Lanctot 2017)** generalizes self-play/fictitious-play/double-
oracle: keep a POPULATION, build an empirical payoff matrix, solve it for a META-STRATEGY σ (which
opponents to face — self-play=newest, fictitious=uniform, **PFSP**=oversample-who-you-lose-to, which we
have via `--pfsp-scale`), train a BEST-RESPONSE (oracle) vs σ, add it, repeat. Convergence = a fresh
best-response can't gain (exploitability → 0). Our pool + PFSP + `--exploiter` is already a crude PSRO.

**Our two twists:** (1) DISTILL best-responses into one generalist instead of keeping a mixture (single-
agent goal); (2) best-respond over a TEAM axis too, not just policy (the exploiter's team is part of the
best-response — currently hand-picked; principled = search/retrieve the team that beats G).

**Finding what it's worst at — two granularities:**
- *Which teams* — a bridge TOURNAMENT SWEEP: frozen G vs the pool, per-team WR, rank ascending, aggregate
  by archetype (`gen3_team_archetypes.json`). Scale the equal-pilot mirror / `run_local_battles`. Cheap.
- *Which skill + is it coachable* — the PROBER (`triage`/`scan`/`falsify-scan`): decompose the worst
  cluster's losses into levers, and FILTER OUT team-draw (uncoachable) so an exploiter isn't wasted.

**Finding the N nearest teams to generalize with — team-similarity metrics:**
1. *Composition* (cheap, now): bag-of-species / averaged `pokemon_encoder` role-tokens → k-NN, or the
   `gen3_team_archetypes.json` cluster. Pragmatic default.
2. *Matchup* (most relevant): cluster teams by WHICH generalist weakness they punish (prober per-loss
   lever) — so distilling fixes the gap broadly.
3. *Behavioral/transfer* (the true target): train on one team, test on candidates, KEEP the ones the
   exploiter transfers to — its own generalization radius defines the neighborhood.
Start from the archetype cluster, tighten/widen by transfer.

**A round, concretely:** (1) freeze G_r; (2) sweep + prober-triage → worst coachable cluster C; (3)
covering set = C (or k-NN); (4) train exploiter E_r vs frozen G_r pinned to C, curriculum
specific→general; (5) distill E_r into G_{r+1} JOINTLY with prior teachers, per-archetype-normalized;
(6) fresh exploiter vs G_{r+1} on C — can't gain ⇒ C covered, move to next-worst; no cluster gains ⇒
approximate Nash ⇒ generalist ≈ specialists. Meta-strategy = PFSP extended to team CLUSTERS.

Honest caveats: distillation ≠ exact best-response (loses edge, risks forgetting → joint distill +
snapshot pool; exploitability-shrinking is the signal, not provable Nash); team best-response is
RETRIEVAL over the existing pool, not open-ended teambuilding (won't invent an unseen team); each round
costs ~an exploiter run + distill + sweep, so aim it with the sweep+triage.

### Do we even need MoE? (superposition says no)

**No — MoE is insurance, not the cure.** Superposition/polysemanticity means a DENSE shared trunk can
hold many archetype "programs" in overlapping directions (more features than dims when they're sparse;
archetype is sparse per-trajectory). Capacity is NOT the wall — the ~3–5-dim rank collapse shows we
don't even use what we have. MoE only earns its keep if a dense trunk hits an *empirical* interference
wall AFTER the objective is fixed.

The load-bearing gap in the LLM-superposition analogy: **representing an archetype ≠ playing it well.**
LLM superposition emerges under a dense next-token loss that rewards distinguishing every feature; our
RL loss is sparse, bootstrapped, and doesn't reward archetype-distinct *action*. Superposition is a
capability the objective must *summon*. And the trunk PROBABLY ALREADY represents archetype (team is in
obs, probes decode roles) — the gap is the map from representation to specialized ACTION, a
credit/exploration problem, not a capacity one.

**Is "per-archetype value/advantage" sufficient? Necessary-ish, not sufficient.** π(a|s) already has
the team in s, and GAE already gives per-STATE (hence per-team) advantages — so the pressure nominally
exists per state. It still fails because: (1) the shared critic is *wrong* for the archetypes it pilots
badly — A = Q − V, and a tail-blind V makes balance's good patient moves score neutral/negative, so the
gradient is present but POINTED WRONG (a corrupted signal, not a missing one). Fix = accurate-shared-V
(tail-weighted loss + per-archetype *critic-loss* normalization); NO separate value head needed, V(s)
already conditions on team. (2) Exploration — self-play may never SAMPLE the 25-turn specialist line to
reinforce it; advantage reweighting can't fix a trajectory you never visit → that's the teacher's job.
Per-archetype advantage *normalization* fixes only the narrower scale-drowning problem. So the no-MoE
recipe is: accurate per-archetype critic + a teacher for exploration + let the dense trunk superpose;
MoE only if that empirically interferes.

**Honest ceiling.** The exploiter's ~0.78 was vs a *weaker* _14 and includes exploiting *that
opponent's* mistakes — a **moving target** that shrinks as the generalist absorbs the specialists
(the team itself is a 0.375 underdog, so no team merit props up the 0.78). So the real target is
"**pilot each team near its skill-ceiling**," and the exploiter-headroom collapsing round-over-round
is the measurable signal — not matching the transient 0.78.

**RETRACTED — the "~⅔ of grind losses are unwinnable team-draw" claim.** Earlier notes
(`project_exploiter_league_tooling`, `project_positional_grind_decomposition`) cited this as a hard
ceiling. It is NOT credible: (1) circular — "matchup-lost/behind-from-turn-1" was defined by the
model's OWN tail-blind critic, so it mostly measures the model's self-inflicted pessimism, not the
position; (2) our own P(win)-threshold recalibration already flipped ~⅓ of that bucket to coachable
throws (grind 35%→27%); (3) gen3 OU has NO team preview at all, and the existence of a real skill/ELO
spread among strong players (who sustain high win rates vs the field) shows skill dominates matchup
luck — a ~⅔-coinflip game couldn't support that spread. Team matchup adds *some* uncoachable variance,
magnitude unknown but far below ⅔. Do not use the ⅔ figure.

## Synthesis

Nothing is broken in the wiring: the team is in the obs, the trunk attends over it, and probes show
it reads the board. What's "wrong" is structural, not a bug — a single self-play policy under an
averaged objective, with a tail-blind critic and no pressure to specialize, converges to a coarse
team-agnostic policy. The exploiter beats it by being a dedicated per-team solution, not a better team
(the mirror proved the team is a 0.375 underdog). Closing the gap needs a critic that can credit
long-horizon plans, architecture that forces per-archetype routing, a curriculum that punishes being
average, or distilling the specialists back in — not a fix to how the model "sees" its team.

## See also

- `designs/learning/on_policy_self_distillation.md` — folding specialist policies back into the trunk
- `designs/learning/pbs_value_functions_and_search.md` — the critic / value-function credit-assignment half
- `src/agents/model/CLAUDE.md` — the feature-extractor phase contract, `ValueDistHead`, projection heads
- `src/agents/observation/CLAUDE.md` — the "Our team (6 × 110)" obs block layout
- Memory: `project_exploiter_no_team_advantage` (equal-pilot mirror), `project_archetype_competence_gradient`
  (switch-flat / move-availability), `project_plateau_research_2026_06_25` (self-play flat fixed point),
  `project_search_teacher`, `project_positional_grind_decomposition` (tail-blind critic on the long game)
