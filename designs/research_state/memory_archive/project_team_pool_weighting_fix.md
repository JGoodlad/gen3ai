---
name: project_team_pool_weighting_fix
description: "yak_attack per-mon manifest bug inflated team draws to 66%; fixed (collapse+dedupe), must land at gen3_outgoing_ko_v1 run boundary"
metadata: 
  node_type: memory
  type: project
  originSessionId: 61a09f90-51c9-405a-8204-992ae8e8ddc3
---

> **Archived 2026-09-08** — data fix landed; the 719-team pool count is stated in root CLAUDE.md. Preserved verbatim; nothing below is current.

Fixed (2026-06-10, branch `claude/gallant-newton-4ccef5`, NOT shipped) the team-pool draw-weighting
bug: `data/teams/others/yak_attack/teams.json` was indexed PER POKÉMON — 1122 rows over 185 distinct
.txt files (174 valid), so each yak team got ~6–12× a normal team's weight. `team_loader._load_teams`
appended file text once per ENTRY → pool was **1601 over 719 unique**; uniform draws gave yak_attack
**66% of ALL training + eval episodes** (both `train_rl_agent.py` and `eval_worker.py` use
`TeamLoader().get_all_teams()`).

**Root cause:** the source PokePaste dump (`tools/others_team_downloader/sync.py`) names a team once
per mon as `"<Mon>/<Team Name>"`, same 6-mon text under 6 headers (same sha256 id → same file). The
generator never deduped, so all per-mon rows survived. Every OTHER source is already one-entry-per-team.

**Fix (3 layers + guards):**
1. Generator: `collapse_duplicate_teams()` in sync.py — collapse rows by id before writing (name =
   strip `"<Mon>/"`, valid = AND, errors = union, idempotent). A re-run can't reproduce it.
2. Regenerated yak manifest in place (one-off collapse of the existing manifest via the same fn — no
   network needed): 1122 → 185 entries.
3. Loader defense-in-depth: `_load_teams` dedupes by resolved file path + prints a LOUD warning if a
   manifest references a file >1×. Both layers independently yield 719.
4. Guards: `team_manifest_test.py` (data-contract: one-entry-per-file over all manifests + collapse-fn
   units), extended `loader_test.py` (synthetic per-mon dedupe + real-count pin 32/687/719).

**Result:** pool 1601→**719** (samples=32, others=687). Draw share: yak 66%→24% (~2.7× less repetition),
every other contributor +2.23× (giraffe 23%→51%). Golden parity (`gen3_data_obs_parity`) UNAFFECTED —
`golden_obs_capture` uses sample-only (32, order-preserved).

**CRITICAL landing constraint:** this is a **data-distribution change** (training + eval). It will
confound any in-flight A/B — it must NOT land mid-A/B. Intended landing = ALONGSIDE the
[[project_outgoing_damage_design]] (`gen3_outgoing_ko_v1`) retrain boundary, as a clean reset.
Stale "~39 curated teams" claims remain in explicit-only design docs
(`designs/ai_v6/impl_step4_team_completion_enrichment.md`, `designs/ai_v5/design_incoming_damage_obs.md`)
— NOT auto-updated per the designs/ explicit-only rule; update via /gen3ai-update-design-docs if wanted.
