# ai_v11 — TODO

The **human-ladder-replay** chapter. Where ai_v9 is the entity graph inside one battle and ai_v10 is
what transfers between teams, **ai_v11 is what we can learn from an external (non-self-play)
action distribution — and what survives the fact that those replays are partial information.**

Plan of record: [`design_human_replay_objectives.md`](design_human_replay_objectives.md).
**Nothing is built.** Every rung's gate is pre-registered there and unrun.

---

## Phase 0 — free, CPU-only, runs beside a live training run

- [x] **P0.0** — corpus + faithfulness census (`tmp/replay_faithfulness_census.py`;
      `tmp/census_all.json` seed 1, `tmp/census_1500.json` seed 2). Headline: 263,159 logs /
      2.8 GB / 2026-05-18→2026-08-02; parse failure 3.15% at ≥1500; tier-A (6/6 bench, 4/4 moves)
      **16.70%** of 30,146 decisions; own item known on **3.93%** of own mons; α-label pairing
      **92.04%**; human switch share **28.96%**.
- [ ] **P0.1** — widen the census to ~20k sides (tight CIs on the tier/coverage shares).
- [ ] **P0.2** — 🚩 **the rung-1 offline probe (gate G1)** — the cheapest kill in the document, and
      the one that decides candidacy. Must run from the checkpoint's own `git_hash` worktree
      (`log_reader` encodes with HEAD's encoder).
- [ ] **P0.3** — `alpha_mask_rate` on human data: is the human's move inside the belief's top-K
      seats? (rides P0.2)
- [ ] **P0.4** — 🚩 **per-DECISION mask-hazard rate** (§5.6 is per-GAME). Decides whether rung 3 is
      feasible at all — pre-registered floor is 50k admitted decisions at ≥1500.
- [ ] **P0.5** — fix `src/main/human_agreement.py`'s silent per-side `except Exception: continue`
      (it hides the measured 3–4% parse failures from its own fidelity block). 3 lines.
- [ ] **P0.6** — read the Metamon paper for **how it handled acting-side partial information**
      (§8.3 — if their team generator was callable, the precedent does not transfer).
- [ ] **P0.7** — re-measure the 2026-06-12 human-agreement headline (35% match; 16% vs 30% switch)
      at the current architecture. It cannot be reproduced at HEAD without a pinned worktree.
- [ ] update `designs/research_state/README.md`'s human-ladder-data frontier row: the corpus is
      **263,159** logs, not "~102k", and collection appears stopped at 2026-08-02.

## Owner decisions (§6) — none ruled

- [ ] **D1** domain-indicator obs feature — doc recommends **NO** (use the label-weight seam, per the
      `opp_class` precedent).
- [ ] **D2** human α labels as a sanctioned source — doc says **YES** by the standing rule; the
      obligations (committed artifact provenance; spread from `gen3_spread_priors.json`) need ruling.
- [ ] **D3** rating floor — ≥1500 for rungs 1–2, ≥1600 for rung 3, or a continuous rating weight.
- [ ] **D4** restart `collect_replays.py`?
- [ ] **D5** one α head with a HUMAN class weight, or a separate human-α evaluator head.

## The ladder (each rung starts only when the one below passes or is killed)

- [ ] **Rung 1** — α/β supervised on the human OPPONENT's actions. Offline; no generation slot.
      Gate **G1**, kill **K1**. Needs a new offline batch path into the intent loss (the `opp_class`
      rollout-buffer plumbing does NOT transfer) and an offline analogue of
      `align_labels_to_predictions`.
- [ ] **Rung 2** — outcome/value supervision on human states (the `WinProbLabelCallback` MC pattern,
      offline). `read_only` side readout only; LEVEL claims only. Gate **G2**, kill **K2**.
      ⚠️ outcome-balanced faithfulness weights are **mandatory and pre-registered** (the faithful
      stratum is loss-enriched 1.29×).
- [ ] **Rung 3** — BC-regularization on the faithful, mask-audited subset. **gen-17 candidate.**
      Gate **G3** (switch-rate move AND ELO non-regression; switch-rate-only is a KILL), kill **K3**.
- [ ] **Rung 4** — offline RL with team-completed acting sides. Gates **G4a** (downstream
      damage-row distribution overlap, not accuracy) / **G4b** (completion beats filtering at equal
      budget), kill **K4**.

## Team completion (§4) — the OOD-closer for rungs 3–4

- [ ] Treat as **BUILT, UNTRAINED** — `models/` carries no `team_prediction/` run (checked
      2026-08-18).
- [ ] Re-verify or **drop** the frozen-backbone embedding lift (it loads
      `features_extractor.{species,move,item}_embedding.weight` by name; unverified at v96, and 79
      of 79 archived runs cannot reload at HEAD).
- [ ] Re-point it at **our own** side (it was designed for the opponent's team / MCTS world
      sampling).
- [ ] **Spread completion is not learnable from replays** — no replay states a spread. Sample from
      `gen3_spread_priors.json` conditioned on species (+ moveset). Never from the team pool.
