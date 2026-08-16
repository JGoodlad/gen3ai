# RUNBOOK — gen-12 end-of-run audit battery → the gen-13 config

**THE BATTERY IS AUTOMATED** (2026-08-16, `src/main/endofrun.py`): one command applies §§1–3's
pre-registered rules mechanically and writes the verdict JSON + Markdown into
`designs/research_state/measurements/`:

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.endofrun models/ai_v9_14_gen12_h_entitypool_shaping_0816 \
    --ref models/ai_v9_13_gen11_labelonly_winprob_0815
```

(Gen-12 recorded v80 and the signature is unchanged, so the model-loading audits run under HEAD
directly — no pinned worktree needed; an arch-drifted run gets the pinned-tree instructions in
the report instead of an error.) This file remains the REGISTRATION OF RECORD — the runner cites
these rules; §4's per-family reads and any judgment calls stay human.

**Pre-registered 2026-08-16, BEFORE gen-12 launches** — decision rules written before the
numbers exist (the concat-deletion precedent). Assumes gen-12 = gen-11's config + the `h` edge
family + (if the gen-11 audit permitted) `--value-entity-pool`, on whichever belief-grad default
the gen-11 §1 verdict selected. Adjust the arm list to what actually launched; every rule below
names its arm explicitly so an absent arm just skips. Run from the RUN'S OWN pinned worktree;
write reports into `designs/research_state/measurements/` with provenance.

## 1. Non-inferiority (the generation gate)

Dense offline anchored ladder (`python -m main.elo <run>`) vs gen-11's dense number — the §5
margin convention (within −15, CI excluding −40). Sparse in-run `eval/elo` is orientation only.

## 2. The `h` family — did compiled pair-history come ALIVE?

- `edge_ablation_audit` per-family arm on `h`: masked KL / flips / |dV| vs the family-off
  forward. Zero-init family ⇒ the arm reads exactly 0 on an untrained head, so ANY nonzero is
  learned use; compare against the median live family (the d/s families), not against zero.
- The intended consumers: `opp_intent/alpha_acc_pool` and `beta` top-k vs gen-11 — the
  tendency inputs exist for the first time, so intent accuracy is the mechanism check
  (read the `_pool` suffix; the bare keys are a moving mix).
- Decision: `h` at ≥ half the median live family's |dV| OR an intent-accuracy gain ⇒ KEEP
  (it graduates into the standing family string). Both null ⇒ `h` reverts to opt-in and the
  H-A2 obs block becomes a deletion candidate for the H-B-enable generation (H-A1 last-action
  stays — it rides the slots, near-free).

## 3. `value_entity_pool` (if launched) — the Stage-3 adoption call

- `critic_route_audit` with the FULL arm set (`seed` / `threat` / `hidden_opp_*` /
  `entity_pool` / `nmr` / `all_off`). The successor earns the deletions: if `entity_pool`'s
  |dV| ≥ the sum of the routes it replaces (seed + threat) while those routes' single arms
  fall under the gen-11 §2 deletion bar, gen-13 DELETES seed+threat (pure code removal, the
  cleanup-journey playbook) and the pool stays ON.
- **The successor is now the FULL pool** (v82 `--value-entity-pool-full`): +the refined
  global token and +the hidden-opp belief queries as row sources — so a condemnation of the
  hidden-opp VF half or the `nmr` concat also has its replacement ready (enable `full` in the
  same config change; gen-12's v80-shape pool keeps loading regardless).
- The `nmr` arm is new evidence, not yet a deletion license: `non_matchup_rest`'s content also
  rides the global token, so a small `nmr` KL+|dV| ⇒ the direct concat is redundant ⇒ Phase-3
  item 2 (the re-home + deletion) is GO for gen-13; large ⇒ the global-token route is not
  carrying it — investigate before deleting.

## 4. H-A verdict → the H-B enable decision (gen-13's headline arm)

The compiled tier (v79) vs the event tier (v81) is the design's §6 sequencing:

- If `h` came alive (§2) OR the pair-history saliency probe reads nonzero, compiled history
  carries signal ⇒ **gen-13 enables `--history-events`** (the seats are built, fuzz-gated —
  73k checks / 0 failures — and OFF-byte-identical until then), and per the design's row-2
  discipline the SAME generation deletes NOTHING yet (one behavioral change per generation:
  the 7×159 TurnDelta deletion waits for H-B's own verdict in gen-14).
- Pre-enable gates (BOTH, already named in the CHANGELOG): the event-window obs benchmark on
  an idle box (the +608-dim fold cost must stay inside the observation leaf's regression
  criteria), and a `--history-events` bridge smoke.
- If compiled history nulls entirely, H-B enablement still proceeds on the design's own
  argument (the event residue is what compiled state CANNOT carry) but as the ONLY new arm.

## 5. Instrument re-reads (no new machinery)

- `query awareness <run>` — cap-aware@5 + coverage80 vs the gen-10 baselines (0.50 / 0.44)
  and gen-11's numbers; the label_only calibration story should show here if real.
- The belief `mask_rate` family — coverage drift across the h/entity arms.

## The gen-13 draft (to be edited by these verdicts)

```
gen-12's config
  + --history-events                     # §4, the headline arm (v81, fuzz-gated)
  + --edge-bias-families ...,r           # OPTIONAL rider: the H-C reference edges
                                         # (gen3_event_ref_edges_v1) — zero-init and
                                         # structural, so it can ride the SAME arm as the
                                         # seats (jointly ablatable via the family arm) or
                                         # wait for gen-14 with the frame deletion
  ± seed/threat DELETED + entity pool ON # §3, only if the audit condemns + the successor carries
  ± non_matchup_rest re-home/delete      # §3 nmr arm, only on a small reading
  + --item-belief                        # OPTIONAL rider (v83, gen3_item_belief_v1): the hidden
                                         # ITEM as a belief; cold-start == the Smogon prior the
                                         # op's static SPECIES_CB_PRIOR already used (within
                                         # 0.6%), so it is ~behavior-preserving at init and the
                                         # CE (belief/item_*) attributes it — a low-risk rider
                                         # in the same class as the H-C edges. Smoke-verified
                                         # e2e (acc 0.93 @ 4k debug steps — mostly the prior).
  + --intent-threshold                   # OPTIONAL riders (v84 + v85: gen3_intent_threshold_v1
  + --intent-conditional                 # + gen3_intent_conditional_v1), GATED ON gen-12's
                                         # per-arm audit of intent_move_cell (= the G3 verdict):
                                         # if c2-through-the-move-cell came alive, the ten
                                         # mechanic cells ride the same proven channel; if it
                                         # stayed at zero, enable ONLY --intent-threshold for
                                         # the p_KO critic half (ledger H1) or nothing. The
                                         # did-it-work readout is the G2 usage baseline
                                         # (measurements/gen12_mechanic_usage_baseline.json:
                                         # Endure 0.0%, Sub 0.9%, Counter 5.6% @ 9.2% prob —
                                         # re-measure post-retrain with
                                         # `python -m agents.model.mechanic_usage_baseline`).
  (nothing else behavioral — the attribution discipline)
```
