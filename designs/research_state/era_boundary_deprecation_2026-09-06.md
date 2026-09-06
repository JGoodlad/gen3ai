# The ERA-BOUNDARY deprecation census — 2026-09-06

**A census and a runbook. NOTHING IS DELETED IN THIS PASS.** It prices one hypothetical: *the moment
the win-probability critic becomes the DEFAULT* — the v110 flip — and with it a `MIGRATION_FLOOR`
raised to 110, a migration collapse of the v97–v109 branches, and a third dead-flag purge. The v51
pointer-native boundary is the precedent: *"the cross-era break rides the `ARCH_SIGNATURE` bump, so
every pre-v51 checkpoint fails loud (owner decision 2026-08-03: **no resume/warm-fork across the
boundary; pools/opponents re-seed from the new lineage**)."*

**State when this was written.** `MODEL_CONFIG_VERSION` **109**, `ARCH_SIGNATURE`
**`gen3_critic_route_wave_v1`**, `MIGRATION_FLOOR` **96** (read from
`src/agents/model/model_version/constants.py` and `.../migrations.py`, not from prose). The mode
exists and defaults to `shaped` (`gen3_winprob_critic_mode_v1`, `cbcb0bfb`); the flip has not
happened. **Arm 1 — `ai_v12_01_winprob_critic`, fresh weights, pinned `e798c13a` — launched at
~13:12 today**, mid-census, and is the only run in the archive recording `critic: "winprob"`.

**The design of record already owns the WHAT.**
[`../ai_v12/design_winprob_only_critic.md`](../ai_v12/design_winprob_only_critic.md) §5.3 is the
deletion list and its §6 **A2** is the consumer census for the `value_dist` family (done 2026-09-06,
11 areas, read-not-grepped). This document does not restate them. It adds the three things they do
not carry: **the archive evidence** (which runs actually had each flag ON), **the cost in
loadability** (which runs stop loading, and what still depends on them), and **the ordered runbook**
with a gate per commit.

**Every count below is executed.** The commands are pasted, and the scripts are in
`designs/research_state/measurements/era_boundary_2026-09-06/` — **committed beside their
outputs**, because the ledger already records what happens otherwise (probe H5's per-team artifacts
lived only in a job tmp dir and were nearly lost).

---

## 0. The headline

| question | answer |
|---|---|
| registry flags that become deletable **from the floor raise alone** | **0** |
| registry flags that become deletable **from the DEFAULT FLIP** | **6 flag families / 11 CLI flags** (the critic-side set §5.3 already names) |
| runs under `models/` that **stop loading** | **113** of 218 with a config (the other 105 are already refused) — **and one of them is `ai_v12_01_winprob_critic` itself**, see §7 |
| ... of those, named in a **checkpoint-LOADING script** | **24** |
| ... of those, with a **hard-coded module-level default** pointing at them | **1** — and it is the untaught meter's fixed opponent |
| pre-pointer residues still deletable today | **0 code paths.** Every hit is a comment, a migration/sanitizer entry that must survive, or a live flag |
| migration lines the collapse removes | **123** (`_migrate_config`'s v97–v109 branches) **+ ~50** version-independent sanitizer lines |

🚨 **THE FINDING THAT MATTERS, AND IT INVERTS THE PREMISE.** The task's framing was *"delete every
flag the purge KEPT only because a gen-9+ research run once had it ON"*. **That set is EMPTY.**
Measured over all 49 registry flags against 126 gen-9+ run configs: **every flag that any gen-9+ run
ever turned ON is ACTIVE in the production config today.** Not one flag is simultaneously
*(ON in a gen-9+ run)* and *(OFF in production)*. So the v108 purge's binding constraint was never
condition 1 (archive usage) — it was **condition 2 (production-ON)**, and a floor raise does not
touch condition 2.

What the flip *does* dissolve is condition 2, for exactly one family: the **critic-side flags the
`winprob` mode refuses**. Those are production-ACTIVE today and dead the moment the default moves.
**The floor raise is what then makes them safe to delete** — their `state_dict` keys live only in
pre-flip checkpoints, and a raised floor is what stops those checkpoints loading.

**So the two moves are not independent, and neither is sufficient alone.** The flip makes the flags
dead; the floor makes them *deletable*. That is the whole shape of this census.

---

## 1. Method and provenance

```bash
export PYTHONPATH=$PYTHONPATH:src
python designs/research_state/measurements/era_boundary_2026-09-06/inspect_registry.py       # the registry surface
python designs/research_state/measurements/era_boundary_2026-09-06/surface_counts.py         # both surfaces, recounted
python designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.py    # per-flag ON/OFF over every models/*/model_config.json
python designs/research_state/measurements/era_boundary_2026-09-06/run_loadability.py        # which runs load today, which stop at floor 110
python designs/research_state/measurements/era_boundary_2026-09-06/reference_scan.py         # who names each of those runs
python designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.py       # ... and which of those namers LOAD a checkpoint
python designs/research_state/measurements/era_boundary_2026-09-06/flag_footprint.py         # per-flag source/ledger footprint + the residue greps
python designs/research_state/measurements/era_boundary_2026-09-06/deletion_loc.py           # line-mention counts + whole-module sizes
```

* **Registry surface** — `agents.model.flag_registry.REGISTRY`, read from the module. **49 entries.**
* **Argparse surface** — `main.train_rl_agent.build_parser()._actions`. **261 actions, 260 excluding
  `-h`.**
* **Archive** — every `models/*/model_config.json` in the MAIN checkout, read-only. **217 runs carry
  a config; 126 are gen-9+** (`config_version >= 69`, the first version in the current signature
  lineage), spanning **69..107**. *(The §2 flag tables are all computed on this 12:15 snapshot;
  `ai_v12_01_winprob_critic` landed at 13:12 and appears in §3's second column and in §7.1. It
  changes no §2 row — it enables no flag that was previously OFF-everywhere.)*
* **Production values** — `designs/ARCHITECTURE.md` §6 (GENERATED from
  `designs/production_config.json` resolved against HEAD; `arch_tables --check` green).

⚠️ **Recount before quoting.** The flag census banked 2026-09-06 says 49 registry / **256** argparse
/ **214** runs / **123** gen-9+. Same day, hours later: 49 / **261** / **217** / **126**. Three runs
and five parser actions arrived in between. The counts here have a timestamp for the same reason.

⚠️ **`is_enabled` is VALUE-based, not default-based.** A flag counts as ON when its recorded value is
not one of `False / 0 / 0.0 / None / "off" / "none" / ""`. Default-based would have mis-scored
`attend_unrevealed_opponents`, whose default IS `True`.

---

## 2. (a) What becomes deletable

### 2.0 The empty set, stated first

```bash
python designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.py    # the ON/OFF column per flag
```

Result over 126 gen-9+ configs — **only three registry flags were never ON in any of them**:

| flag | ON | OFF | why the v108 purge kept it |
|---|---|---|---|
| `damage_candidate_k` | **0** | 126 | OFF in production, builds nothing — but it is an armed lever, not dead code |
| `pair_value_route` | **0** | 117 | OFF in production and **owes the C4 offline gate** — the re-entry condition is written verbatim into the registry, the CLI and the docs (condition 4) |
| `q_winprob_mode` | **0** | 63 | landed **at v107** and is the design doc's named **arm 2** (condition 4) |

All 46 others are ON in gen-9+ runs **and ACTIVE in production**. There is no flag in the
*"ON in the archive, OFF in production"* cell. **The set the task hypothesised does not exist.**

### 2.1 The set that DOES become deletable — the critic-side family

Deletable **only under the flip**, and safe to delete **only under the raised floor**. §5.3 of the
design doc is the authority on scope; this table adds the archive column and the footprint.

| flag family | CLI flags | builds params? | gen-9+ runs ON (newest) | production | ledger verdict | files | line mentions (src / test) |
|---|---|---|---|---|---|---|---|
| **distributional critic** | `--value-dist-mode`, `--value-dist-bins`, `--value-dist-vmin`, `--value-dist-vmax`, `--value-dist-coef` | **YES** — `ValueDistHead` (`aux_value_heads.py:50-102`, 53 lines) is a `Linear(d, bins)` + a non-persistent `atoms` buffer | **122 of 126** ON, newest `ai_v9_197_G5PLAINC_0906` (v107, `"shaping"` / 51 bins / ±12.0) | **ACTIVE** | **"an INSTRUMENT, not a lever here"** (sub-Gaussian residuals, no tail to re-weight). §3.8 adds the decisive COUNTING argument: under a terminal-only objective the return takes **two values**, a categorical over a two-point support **is a Bernoulli**, so 51 atoms are the same parameterization with 50 redundant DoF plus a discretization error | 11 areas — the A2 census table is the list; `aux_value_heads.py`, `value_terms.py:89-141`, `ppo.py`, `policy.py:184-201`, the whole prober awareness stack, 9 version-machinery modules | **260 / 121** |
| **critic selector** | `--value-from-dist`, `--allow-value-from-dist-change` | no (a branch in `_critic_value`) | 122 ON, newest `G5PLAINC` (v107) | **ACTIVE** | ledger **M2**: it ORPHANED the entire critic delivery chain for gen-9..12 — five routes bit-exact zero after 25M steps — **fixed at v89**, and the flag now names a choice between two critics where the flip leaves one | `policy.py`, `set_value_from_dist`, `check_value_from_dist`, `combination_checks` | **61 / 33** |
| **win-prob PBRS** | `--win-prob-pbrs-coef`, `--win-prob-pbrs-source` (+ `--win-prob-pbrs-frozen`, already HELD) | no | **0** ON (v104/v105 fields; no run has used them) | absent from §6 | §3.7 — with `V ≡ φ` it adds the advantage to the reward and then takes the advantage of *that*; the Ng shield is weakest exactly where it would now run. **Owner amendment 2026-09-06: the SELF path is deleted at the flip; the FROZEN path stays REFUSED for one generation** | **`agents/training/winprob_pbrs.py` — 395 lines, deleted outright** + the `collect_rollouts` hook + one `combination_checks` entry | **44 / 26** |
| **the second coefficient** | `--win-prob-coef` | no | 121 ON, newest `G5PLAINC` (v107, `0.05`) | **ACTIVE `0.05`** | one critic, one coefficient — two on one loss is the ambiguity that forced `_ce_w = vf_coef if value_from_dist else value_dist_coef` | the separate coefficient path | **42 / 11** |
| **tail weighting** | `--value-tail-weight` | no | ON in production at `0.3` | **ACTIVE (already INERT under Phase B, §1.4)** | §3.6 — the banned shape: at the decision boundary relevance and label NOISE arrive together, so "care more" must be *more samples*, **never larger per-sample weights**. A per-sample weight on a Bernoulli likelihood is that antipattern exactly | `_value_loss_from_se`'s weighting | **36 / 23** |
| **search leaf fall-back** | `--score auto` | no | n/a (a search-dividend knob, not a run config) | n/a | §3.10 — `search.py:270`'s `if mode == "value" or (wp is None…)` must become a REFUSAL: under one critic a fall-back is either a no-op or the losing arm | `main/search_dividend/search.py` | **47 / 19** |

**Whole modules that go:** `agents/training/winprob_pbrs.py` (**395**), `ValueDistHead` (**53** of
`aux_value_heads.py`'s 263), `_value_dist_loss` + the HL-Gauss target + PIT (part of
`instrumented_ppo/value_terms.py`'s **155**).

⚠️ **A "line mention" is a grep hit, not a deletable line.** A loop over a flag name is one mention
and zero deletable lines; a module deleted for one flag is hundreds. **Honest total estimate:
roughly 600–900 source lines and 200–250 test lines**, dominated by `winprob_pbrs.py` (395, gone
whole) and by the `value_dist` family's spread across 11 areas — most of the latter being deletions
of a few lines each in many files, which is the expensive kind.

### 2.2 EXPLICITLY NOT deletable at the flip — and the design doc says so

`--td-aux-coef` (**KEEP, and it gets *more* useful** — the residual is now in probability units and
the `/ popart.sigma` divisor disappears) · `--cf-winprob-coef` and family (**KEEP** — once `win_head`
is the critic, the counterfactual term becomes the label factory feeding the critic directly) ·
`cf_twin_heads` (**KEEP** — twins *of the critic*) · `cf_shadow_critic` (**REPURPOSE, do not
delete** — one line changes, the mirror direction is newly wanted) · `cf_evidential` (**KEEP, OFF**)
· `q_winprob_mode` (**KEEP, OFF for arm 1 — the natural arm 2**) · `use_popart` (**KEEP the flag and
the module, set false** — the two-currency machinery is correct and wanted again the moment the
objective stops being terminal-only) · `--draw-penalty` (**a REFUSAL, not a deletion** — the flag
stays meaningful in raw-terminal mode) · `--win-prob-mode` (**KEEP, drop `none` from its legal set**).

### 2.3 The migration collapse

```bash
awk 'NR>=313 && NR<=435' src/agents/model/model_version/migrations.py | wc -l   # -> 123
wc -l src/agents/model/model_version/migrations.py                              # -> 435
```

* **`_migrate_config`'s POST-FLOOR branches v97 → v109: 123 lines**, all unreachable at floor 110.
  Their narratives are preserved verbatim in the `PRE-FLOOR MIGRATION HISTORY` comment, which is how
  the v77–v95 run was collapsed when the floor rose to 96. **This is documentation, not code, and it
  is DELIBERATE** — the module docstring says so.
* **The VERSION-INDEPENDENT SANITIZERS (~50 lines, `migrations.py:261-312`)** also become
  unreachable: every key they pop left the config at or below v108 (`value_active_readout`,
  `damage_matrices_outgoing_all`, `pubval_mode`, `pubval_coef` at v88; `intent_value_reduce`,
  `value_clock`, `value_intent` at v96; `threat_prob_outspeed` at v108), so no v110+ config can
  carry one, and the floor check at `migrations.py:51` runs **first**.
* 🚨 **`snapshot._DEAD_FEK_INERT` / `_DEAD_FEK_JUDGED` DO NOT GO WITH THEM.** They sanitize the
  **PICKLED** `features_extractor_kwargs` out of the `.zip`, which carry **no version number and are
  therefore not floored at all** (`sanitize_dead_extractor_kwargs`, `snapshot.py:1003`). The module's
  own comment says it: *"the config JSON drives the version GATE, while these kwargs are what SB3
  splats into the extractor constructor"*. Deleting the JSON half and the ZIP half together is the
  single most likely mistake in this whole runbook.

### 2.4 The pre-pointer residues — **already clean, nothing to remove**

```bash
for t in action_net zarch film_ lut_ pubval damage_reattend opp_belief_cls_k \
         damage_refine_rounds threat_refine move_belief_prefuse; do
  echo "### $t"; grep -rn --include=*.py -F "$t" src/ | grep -v "_test.py"; done
```

| token | non-test py files | non-test line mentions | what they actually are |
|---|---|---|---|
| `action_net` | 8 | 14 | **A LIVE RAISING STUB.** `policy.py:120` sets `self.action_net = _NoFlatActionNet()`, whose `__call__` raises naming `gen3_pointer_native_v1`. It occupies SB3's slot deliberately so any residual upstream path fails loud instead of silently running a deleted flat head. **KEEP.** The rest are comments saying no flat head exists |
| `opp_belief_cls_k` | 13 | 27 | **A LIVE PRODUCTION FLAG** at `6`, ACTIVE, 122 of 126 gen-9+ runs ON. Not a residue at all |
| `zarch` | 2 | 9 | comments in `migrations.py` (the pre-floor history) + `snapshot.py`'s `_DEAD_FEK_INERT` / `_DEAD_FEK_JUDGED` entries. **The ZIP-side entries must survive** (§2.3) |
| `lut_` | 2 | 5 | same — `zarch_lut*` names inside those two lists |
| `film_` | 3 | 3 | **all three are comments** (`policy.py:164`, `extractor_api.py:83`, `migrations.py:131`) naming `film_pi`/`film_vf` in lists of zero-init projections. Zero code |
| `pubval` | 8 | 15 | the v88 sanitizer pair (`migrations.py:271,280`, `snapshot.py:929-940`), the `fields.py` history comments, and **four prose mentions** using `--pubval-*` as the canonical example of a deleted flag killing a resume (`checkargs.py:4,163`, `parser/__init__.py:57`, `prober/model.py:129`, `session/counterfactual.py:191`). No code path |
| `damage_reattend` | 3 | 4 | one `_DEAD_FEK_JUDGED` entry + two comments |
| `damage_refine_rounds` / `threat_refine` | 2 | 3 each | `_DEAD_FEK_INERT` entries + pre-floor history comments |
| `move_belief_prefuse` | 5 | 7 | one `_DEAD_FEK_JUDGED` entry + comments, incl. `extractor_build.py:478` explaining why a recorded `False` is REFUSED |

**Verdict: the strict rule licenses ZERO deletions here.** The three earlier purges (v75 / v78 / v88)
took these already; what is left is the machinery that makes their deletion *safe*, plus prose.
Removing any of it would re-open a silent-load path or delete a worked example that four different
modules point at.

---

## 3. (b) The cost — 113 runs stop loading

```bash
python designs/research_state/measurements/era_boundary_2026-09-06/run_loadability.py
```

A run loads today iff `config_version >= 96` **and** `arch_signature == "gen3_critic_route_wave_v1"`.

| | at 12:15 | re-run at 13:20 |
|---|---|---|
| runs with a `model_config.json` | 217 | **218** |
| ... carrying the live `ARCH_SIGNATURE` | 112 | **113** |
| ... carrying it but below today's floor | 0 | **0** |
| gen-9+ (`config_version >= 69`) | 126 | **127** |
| **LOADABLE TODAY → refused at floor 110** | 112 | **113** |

By `config_version`: `{97: 8, 98: 1, 100: 2, 101: 13, 102: 1, 103: 23, 104: 1, 107: 63, 109: 1}`.
The `109: 1` is **`ai_v12_01_winprob_critic`**, which did not exist an hour earlier — both columns are
shown because a census that silently absorbs a change it lived through teaches nothing.

**The other 105 runs are already unloadable** and cost nothing — the boundary was paid at v96.

### 3.1 The 113, by campaign

<details><summary>full list (config_version · run)</summary>

**v107 — 63 runs.** `.aborted_R4DOSE12_nometa_1401` · `.aborted_R4DOSE12_poolless_1355` ·
`.dryrun_K6A_1788581936` · `ai_v9_72_R3SELF_0828` · `ai_v9_73_R4S3a_0829` · `ai_v9_74_R4S3b_0829` ·
`ai_v9_75_R4S3c_0829` · `ai_v9_76_R4ACTION_0830` · `ai_v9_77_G1LEAN_0830` · `ai_v9_79_REVIVE1a_0830` ·
`ai_v9_80_REVIVE1b_0830` · `ai_v9_81_REVIVE1c_0830` · `ai_v9_82_REFOLD1_0830` ·
`ai_v9_91_COMPFOLD_0831` · `ai_v9_92_R5F00_0831` … `ai_v9_111_R5F19_0831` (the 20 R5F exploiters) ·
`ai_v9_120_R5FUND00_0901` … `ai_v9_134_R5FUND14_0901` (8 funded) · `ai_v9_140_B2_0901` ·
`ai_v9_141_C1_0901` · `ai_v9_142_N1_0901` · `ai_v9_143_N2_0901` · `ai_v9_150_R4DOSE12_0901` ·
`ai_v9_151_R4DOSE6_0901` · `ai_v9_152_R4DOSE3_0901` · `ai_v9_160_TCFUNDA_0903` ·
`ai_v9_161_TCFUNDB_0903` · `ai_v9_162_TCUNFA_0903` · `ai_v9_163_TCUNFB_0903` ·
`ai_v9_170_TCUNFK6A_0904` · `ai_v9_171_TCUNFK6B_0904` · `ai_v9_172_G1SHORT_0905` ·
`ai_v9_195_G5PLAINA_0906` · `ai_v9_196_G5PLAINB_0906` · `ai_v9_197_G5PLAINC_0906` ·
`run_20260830_180409` · `run_20260830_183828` · `run_20260830_184043` · `run_20260906_083317`

**v104 — 1.** `ai_v9_71_R3ACTIONHI_0828`

**v103 — 23.** `ai_v9_48_G1_action_0826` · `ai_v9_49_G2_advgate_0826` · `ai_v9_50_fdF_p1c_0826` ·
`ai_v9_51_fdF_p2c_0826` · `ai_v9_52_G1p_matched_0826` · `ai_v9_53_R2F5a_0826` …
`ai_v9_57_R2F5e_0826` · `ai_v9_58_R2CTRL_0827` · `ai_v9_59_R2ACTION_0827` · `ai_v9_60_R2TOPK_0827` ·
`ai_v9_61_R2KL_0827` · `ai_v9_62_R2PLAIN_0827` · `ai_v9_63_R3F6a_0828` … `ai_v9_68_R3F6f_0828` ·
`ai_v9_69_R3F6CURR_0828` · `ai_v9_70_R3ACTION_0828`

**v102 — 1.** `ai_v9_45_fdF_p1_0826`

**v101 — 13.** `ai_v9_29_rev1_0823` · `ai_v9_30_rev1_exploit_0824` · `ai_v9_31_tock1_k4_0824` ·
`ai_v9_32_tock1b_rain_0824` · `ai_v9_34_tick1_0824` · `ai_v9_35_tick1_exploit_0824` ·
`ai_v9_36_tock1c_q6_0824` · `ai_v9_37_tick1_dosext_0825` · `ai_v9_38_fdA_coef03_0825` ·
`ai_v9_39_fdB_lossonly_0825` · `ai_v9_40_fdC_ecology_0825` · `ai_v9_42_fdE_single_0825` ·
`ai_v9_44_tock2_v8shape_0825`

**v100 — 2.** `ai_v9_26_baitent_probe_0823` · `ai_v9_27_extremedial_probe_0823`

**v98 — 1.** `ai_v9_25_E4_baitbot_0822`

**v97 — 8.** `ai_v9_19_gen16_mechanics_0819` · `ai_v9_20_tdaux_rung2_lam{00,10,30}_0820` ·
**`ai_v9_21_gen17_pfspoff_0820`** (the last full pre-ai_v12 generation, and what
`production_config.json` mirrors) · `ai_v9_22_E1_substrate_on_0821` · `ai_v9_23_E2_substrate_on_0822`
· `ai_v9_24_E3_substrate_on_0822`

</details>

**In plain terms, the flip retires the entire gen-16/gen-17 lineage plus every ai_v10-era
distillation, exploiter and continuation-control arm** — the R5F fleet, the funded/unfunded 2×2, the
K=6 dose cell, the TC family, G1, G5, and rev-1/rev-3 themselves.

### 3.2 Which of them are still REFERENCED — and by what

```bash
python designs/research_state/measurements/era_boundary_2026-09-06/reference_scan.py     # three surfaces, literal name match
python designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.py   # ... split by whether the namer LOADS a checkpoint
```

| surface | runs named |
|---|---|
| anywhere (`measurements/` ∪ `src/main` ∪ ledger tail) | **111 of 113** — the two that are named nowhere are `run_20260906_083317` and `ai_v12_01_winprob_critic` (the live run; it is named in `designs/CLAUDE.md` and in the design doc §5.4, neither of which is a scanned surface) |
| a **checkpoint-LOADING script** (imports `load_model_snapshot` / `MaskablePPO.load` / `resolve_model_ref` / `SnapshotPool` / `load_snapshot`) | **24** |
| any script at all (`measurements/*.py` or `src/**.py`) | **64** |
| **committed artifacts only** (`*.json` / `*.md`) | **47** — these are RECORDS and cost nothing |
| the ledger's last 1000 lines | **8** |

**The 24 loading-script dependencies**, grouped by the script that would break:

| script | runs it names |
|---|---|
| `measurements/folding_history_probe.py` | `ai_v9_21_gen17_pfspoff_0820`, `_29_rev1`, `_34_tick1`, `_37_tick1_dosext`, `_58_R2CTRL`, `_59_R2ACTION`, `_62_R2PLAIN`, `_63_R3F6a`, `_68_R3F6f`, `_70_R3ACTION`, `_72_R3SELF`, `_76_R4ACTION` |
| `measurements/per_team_gradient_geometry_2026-08-28/probeF_grads.py` (+`probeF_acid.py`) | `_29_rev1`, `_53_R2F5a` … `_57_R2F5e`, `_59_R2ACTION` |
| `measurements/lr_licensing_probe.py` | `_29_rev1`, `_59_R2ACTION`, `_73_R4S3a`, `_74_R4S3b`, `_75_R4S3c`, `_76_R4ACTION` |
| `measurements/representational_richness_transfer_forward.py` | `_29_rev1`, `_58_R2CTRL`, `_62_R2PLAIN`, `_70_R3ACTION`, `_76_R4ACTION`, `_91_COMPFOLD` |
| `measurements/distillability_index_probe.py` | `_29_rev1`, `_53_R2F5a`, `_54_R2F5b` |
| `measurements/teacher_sharpness_probe.py` | `_29_rev1`, `_31_tock1_k4`, `_36_tock1c_q6` |
| `src/agents/training/cf_producer.py` | `_29_rev1` (prose only) |
| `src/agents/training/untaught_meter.py` | `_29_rev1` — **NOT prose. See below.** |
| `src/main/checkargs.py` | `ai_v9_162_TCUNFA_0903` (prose only) |

🚨 **ONE HARD DEPENDENCY, and it is the instrument every fold verdict rests on.**

```python
# src/agents/training/untaught_meter.py:89,94
DEFAULT_OPPONENT = "ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip"
DEFAULT_CONFIG   = "ai_v9_29_rev1_0823/snapshots/model_config.json"
```

These are **module-level defaults**, not docstrings. `ai_v9_29_rev1_0823` is `config_version` **101**
(its `snapshots/model_config.json` likewise), so at floor 110 the untaught meter's fixed opponent
**stops loading and `python -m main.untaught_meter` fails by default** — the meter behind the frozen
floor, the fold deltas, the G1/G5 readings and the whole rev-2/rev-3/rev-4 column. Nothing else in
`src/` has this shape: **every other `src/` mention of a soon-unloadable run is a comment, a docstring
example, or a test fixture string** — verified by reading each of the 13 files that name one.

**That is the cost to accept, in one sentence:** the flip does not merely retire old checkpoints, it
retires the *measuring apparatus calibrated against them*, and either the meter needs a new
era-native opponent (a re-measurement, not a rename — the levels are not comparable across
opponents) or it needs to run from an era checkout forever.

**The other 87 runs** are named only in committed artifacts, prose, or `measurements/` scripts whose
job is already historical. Those are records, and a record does not need its subject to load.

---

## 4. (c) What an era checkout still buys

**Everything model-free keeps working on all 217 runs, unchanged**: `main.lineage`,
`main.sidecar_audit`, `main.dose`, `main.elo`, `main.exploitability`, `main.scaffolding_gauge`
(model-FREE by construction — it reads recorded `values` + `win_probs`), and the prober's `scan` /
`triage` / `turns` / `falsify` / `falsify-scan` / `calibration`. A raised floor changes none of them;
they read JSON, `.npz` and `.jsonl`, never a `.zip`.

**Everything that LOADS a checkpoint** — `main.untaught_meter`, the prober's `analyze` / `lookahead`
/ `better-line` / `replay-counterfactual`, every `measurements/*.py` probe, an exploiter fork, a
distillation teacher — needs the run's own commit.

### The recipe (the `/tmp/v8rep_era` pattern, already proven twice)

```bash
# 1. the commit is recorded PER RUN and PER CHECKPOINT — read it, never guess it
python -m main.sidecar_audit models/<run> -v        # the run pin + every pin_history span
python -m main.lineage models/<run>                 # and who it forked from

# 2. a READ-ONLY worktree at that commit, outside the repo so no gate ever sees it
git worktree add --detach /tmp/<era>_era <the recorded git_hash>
for n in dist node_modules; do
  [ -e "/tmp/<era>_era/deps/pokemon-showdown/$n" ] || \
    ln -s "/home/goodlad/dev/gen3ai/deps/pokemon-showdown/$n" "/tmp/<era>_era/deps/pokemon-showdown/$n"
done

# 3. run the probe with the ERA's src on the path, and NEVER let it write bytecode into the checkout
PYTHONPATH=/tmp/<era>_era/src PYTHONDONTWRITEBYTECODE=1 ERA_ROOT=/tmp/<era>_era \
  nice -n 10 python <the probe>
```

`PYTHONPATH` entries land in `sys.path` **before** site-packages while an editable install's `.pth`
lands after, which is exactly why this beats the install and why the launcher's own pinning works the
same way. **`models/` is shared** — the era checkout has no `models/`, so probes reach it through
`utils.paths.main_models_dir()` (or `$GEN3AI_MODELS_DIR`), which is why the pattern needs no copying.

**What it does NOT buy, and both were paid for.** A cross-era comparison runs in **two processes**,
one per era, because the two trees cannot be imported together (`agents -> /tmp/v8rep_era/...`, obs
dim 2992 vs 2501) — `measurements/arch_transfer_2026-09-05/cross_era_head_to_head/side.py` is the
worked example, one `--role` per side. And a run whose recorded `git_hash` is **misattributed** gives
you the wrong tree: the 2026-09-05 census found **1,820 of 3,975 sidecars in 120 of 202 runs** naming
a commit differing from their run's final pin, which is why `pin_history` and `main.sidecar_audit`
exist and why step 1 is not optional.

---

## 5. (d) The runbook — ordered commits, one gate each

**Precondition, non-negotiable: arm 1 has produced a verdict.** The design doc's own words — *"the
default flip and the `ARCH_SIGNATURE` bump that forces fresh weights land in a later one, **after an
arm has run**"*. `ai_v12_01_winprob_critic` started 2026-09-06 ~13:12 at `--steps 75000000`. Until
its 5M pre-test reads crater / crawl / keep-pace, none of the commits below is licensed.

| # | commit | contents | GATE |
|---|---|---|---|
| **0** | **the §7.1 DECISION** — no code | choose (A) accept orphaning `ai_v12_01_winprob_critic`, (B) floor 109 and let the signature carry the break, or (C) restart arm 1 at v110. **Commit 1's floor value is this decision's output** | banked in the ledger BEFORE commit 1, with the reasoning — it is not recoverable from the diff afterwards |
| **1** | **the DEFAULT FLIP + the signature bump** | `CRITIC_DEFAULT = "winprob"` (`agents/model/critic_mode.py:40`, currently `CRITIC_SHAPED`); new `ARCH_SIGNATURE` (e.g. `gen3_winprob_critic_v1`); `MODEL_CONFIG_VERSION` → **110**; `MIGRATION_FLOOR` → **110 or 109 per §7.1**; the paired `SIGNATURE_FIRST_VERSION` entry. **The signature bump is FORCED** — a critic trained to predict a shaped return cannot be warm-started into predicting a probability | `pytest src/ -m "not slow and not e2e"` · `agents.model.model_version` self-consistency (`MIGRATION_FLOOR == SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE]`) · a NEW test asserting a v109 config is REFUSED with the era diagnosis · `python -m main.checkargs` over a flagless argv |
| **2** | **the MIGRATION COLLAPSE** | delete `_migrate_config`'s v97–v109 branches (**123 lines**) and the version-independent sanitizer block (**~50**), moving every narrative verbatim into the `PRE-FLOOR MIGRATION HISTORY` comment. **DO NOT touch `snapshot._DEAD_FEK_*`** (§2.3) | `dead_kwargs_sanitize_test.py` (the JSON/ZIP agreement pin) · `ctor_kwarg_snapshot_test.py` · every `*_test.py` asserting a pre-floor refusal (there are ~20; each already tests the shape) · `model_version_hub_contract_test.py` |
| **3** | **PURGE v3 — the critic-side deletion** | §5.3's list: `ValueDistHead` + the five `--value-dist-*` + `check_value_dist` + `_value_dist_loss`; `--value-from-dist` + `--allow-value-from-dist-change`; `agents/training/winprob_pbrs.py` whole + its hook + its check; `--win-prob-coef`; `--value-tail-weight`; `--score auto`. Each deleted **resume-immutable** field ships its refusal in the same commit (the v75 rule). **Work §6-A2's list (c) — the ~15 sites that gate on the MODE STRING — as the checklist**; the head cannot be removed by not building it | `mypy_gate_test` · `ruff_gate_test` · `file_size_gate_test` · `flag_registry --check` · `arch_tables --check` · `delivery_graph --check` · `build_arch_viewer --check` · **a new `deleted_toggles_v3_test.py`** asserting every deleted name is absent from `build_parser()._actions`, from `REGISTRY`, and from `ModelVersion`'s fields — and that a config recording one is REFUSED, not popped |
| **4** | **`production_config.json` re-mirror** | it currently mirrors **v97** and still carries `threat_prob_outspeed`, deleted at v108. Re-mirror from `ai_v12_01_winprob_critic`'s own `model_config.json` once it exists | `arch_tables_test.py` (the bump-window detector) · `arch_tables --check` |
| **5** | **CHANGELOG + ARCHITECTURE §6** | append the v110 entry to `CHANGELOG.md` (**append-only — never edit an existing entry**); regenerate `ARCHITECTURE.md` §6's flag and coefficient tables; state the new truth and **delete the old prose** rather than narrating the change | `arch_tables --check` · `flag_registry --check` |
| **6** | **leaf docs + the always-current set** | root `CLAUDE.md` (the `--critic` section, the compile/flag tables), `designs/CLAUDE.md` (both rows), `src/agents/model/CLAUDE.md` (the versioning playbook + the resume-immutable list), `src/agents/training/CLAUDE.md`, `src/main/prober/CLAUDE.md` (the awareness stack loses `knew_by_turn` / `blind_loss` / `coverage80` / `pit_mean`) | the doc gates above, plus a read-through: no surviving doc may describe a deleted flag as live |
| **7** | **the meter's opponent** | give `untaught_meter.DEFAULT_OPPONENT` / `DEFAULT_CONFIG` an era-native successor, **or** pin the meter to an era checkout and say so in its docstring. ⚠️ **A new opponent is a RE-MEASUREMENT, not a rename** — levels are not comparable across opponents, so every banked untaught number keeps its old opponent in its provenance | `untaught_meter_reproducibility_integration_test.py` (two `--workers 2` runs byte-identical) · `main.untaught_meter --check` resolving every ref without playing |

**Ordering that is load-bearing:** **0 before 1** (§7.1 — the floor value IS the decision) · **1 before 2** (the floor must be raised before its branches are
dead) · **2 before 3** (a deletion whose refusal branch has already been collapsed cannot be written)
· **A2 before any of 3** (know what breaks before breaking it) · **3 before 5** (the generated tables
are generated *from* the code) · **7 whenever, but before the next fold verdict**.

**Do NOT bundle.** Each of 1–3 is independently revertable and each has a distinct failure mode; a
single commit carrying all three cannot be bisected when a probe starts failing three weeks later.

---

## 6. (e) What must NOT be deleted, even then

1. **`snapshot._DEAD_FEK_INERT` / `_DEAD_FEK_JUDGED` and `sanitize_dead_extractor_kwargs`.** The
   ZIP-side twin, with **no version to floor against** (§2.3). Deleting these with the JSON
   sanitizers is this runbook's likeliest single mistake.
2. **`check_leaf` in `search_dividend`.** It becomes vacuous, and §3.10 says keep it anyway: *"deleting
   a guard because it currently has nothing to catch is how the choice-reject allowlist entry outlived
   its own fix."* Same for `_NoFlatActionNet` — a raising stub in a deleted head's slot.
3. **The whole bridge / eval / websocket transport tier.** `--use-bridge {off,node,rust}` keeps all
   three values: `node` because the parity harness and `gen_sim_bridge_diff.js` need it, **`off`
   because `src/main/play.py` is the only entry point that talks to a Showdown server as a CLIENT and
   is therefore the exact code path a rated ladder game uses.** The ladder is a live objective
   (`ladder_readiness.md`), and `ladder_drift_scan.py` is its pre-flight gate.
4. **Every model-free prober view and offline CLI** — `scan`, `triage`, `turns`, `falsify`,
   `falsify-scan`, `calibration`, `main.lineage`, `main.sidecar_audit`, `main.dose`, `main.elo`,
   `main.exploitability`, `main.scaffolding_gauge`. These are the *only* things that will still read
   the 217 archived runs after the flip, and their value goes UP, not down. `main.scaffolding_gauge`
   loses its two DIVERGENCE gauges (with one readout there is nothing to diverge from — gap A3) but
   **keeps the reliability half**, which is strictly more informative for a binary outcome.
5. **The era-checkout tooling** — `pin_history`, `main.sidecar_audit`, `utils.paths.main_models_dir()`
   / `$GEN3AI_MODELS_DIR`, the launcher's `--pin-commit` / `resolve_pin`, `main.checkargs`'s
   pinned-parser path (`gen3_pinned_argv_parser_v1`), and `main.lineage`'s
   `resolution_rung` / `resolution_rule` recording. **After the flip these stop being provenance
   niceties and become the only route to 113 runs.**
6. **`use_popart` and the PopArt module** — flipped to false, kept whole, per §3.5/§5.3.
7. **`--draw-penalty`** — refused under the new critic, still meaningful in raw-terminal mode.
8. **The reward-shaping family** (`drop_redundant_bias`, `drop_switch_bias`, `stall_pbrs`,
   `bias_redesign`, `switch_bias_weight`, `mat_alive_weight`, `bias_additivity`, `self_ko_hp_penalty`,
   `all_shaping_pbrs`, `no_progress_penalty`). The v108 census's own ruling: these are
   `check_reward_config` resume-immutable **value** hparams, not extractor toggles, and deleting one
   *silently narrows the reward-immutability check* rather than removing dead code. `--no-hand-shaping`
   turns them all off at once; it does not make them dead. **`--arm-no-progress-tax` in particular is
   the ONLY anti-stall lever a `[0,1]` critic leaves**, since the `−35 < −30` timeout ordering is
   unrepresentable there.
9. **`designs/CHANGELOG.md`'s existing entries and `migrations.py`'s pre-floor history comment.**
   Append-only; their job is to record what was believed at the time.

---

## 7. Three hazards this census found on its way past

### 7.1 🚨 THE FLIP AS DESIGNED WOULD ORPHAN THE ARM THAT LICENSES IT — and it does not have to

`ai_v12_01_winprob_critic` is `config_version` **109** with the CURRENT `ARCH_SIGNATURE`, so a floor
raised to 110 **refuses its own checkpoints**. At 13:20 it already holds
`checkpoints/checkpoint_3200000_steps.zip`, and it is running to 75M.

**But look at what its config actually records:**

```
critic = winprob     value_dist_mode = "none"    value_dist_bins  = 0
                     value_from_dist = false     use_popart       = false
                     value_tail_weight = 0.0     win_prob_pbrs_coef = 0.0
```

**With `value_dist_mode = "none"` the head was never built, so arm 1's `state_dict` contains no
`value_dist_head.*` keys at all** — which is precisely the reason the design doc gives for the forced
signature bump (*"`value_dist_head`'s parameters leave the `state_dict`; the critic route changes"*).
That reason **does not apply to this run.** Post-deletion code could load arm 1's weights; a floor of
110 would refuse it on the VERSION NUMBER alone.

**This is a decision the runbook cannot make for the owner, and it must be made before commit 1:**

* **(A) Accept it.** Arm 1 is the *evidence-gathering* arm; the generation that follows the flip is
  fresh weights anyway, exactly as v51 and v96 were. Cost: 75M steps of GPU become read-only history
  (its traces, `eval_results.jsonl`, ELO ladder and scaffolding-gauge readings all survive — they are
  model-free), and any post-flip probe of arm 1 needs an era checkout at `e798c13a`.
* **(B) Set the floor to 110 but let the SIGNATURE carry the break.** The signature bump alone already
  refuses every pre-flip `shaped` checkpoint with a diagnosis. A floor of **109** would then keep
  arm 1's *version* legal while the signature decides compatibility — but it also keeps the v97–v108
  migration branches alive, so commit 2 shrinks from 123 lines to ~110 and the sanitizer block cannot
  be collapsed. **This is the cheaper option only if arm 1's weights are actually wanted.**
* **(C) Restart arm 1 at v110 after the flip.** Cleanest semantically, most expensive in GPU.

**Do not discover this at commit 1.** A floor raise is irreversible in the sense that matters: once
the branch is deleted, restoring the ability to load a v109 checkpoint means reverting two commits
and re-deriving what the branches said.

## 7.2 The other two

* **`designs/production_config.json` is at config v97 and carries `threat_prob_outspeed`, a key
  deleted at v108.** `arch_tables --check` is green because it resolves the file against HEAD's
  registry and ignores the stale key — so the drift gate does not catch a *stale extra* field, only a
  changed generated table. Harmless today; commit 4 fixes it.
* **The archive moves under a census.** `ai_v12_01_winprob_critic` appeared between this document's
  loadability scan (12:15) and its reference scan (13:14), and the earlier draft of
  `designs/CLAUDE.md`'s *Active training run* row — written from a `ps` that returned nothing and a
  GPU at 0% — was **wrong within the hour**. Re-read `ps` and `models/` immediately before banking
  any claim about what is running.

---

### Appendix — reproducing the two headline tables

```bash
export PYTHONPATH=$PYTHONPATH:src
# every registry flag's ON/OFF over the archive, newest ON run named
python designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.py | head -60
# the 113 runs and the by-version histogram
python designs/research_state/measurements/era_boundary_2026-09-06/run_loadability.py
# who still names them, split by whether the namer loads a checkpoint
python designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.py
```

Each script's docstring states its definitions (what "enabled" means, what "loads a checkpoint"
means) so a re-run is comparable; **re-derive the counts, do not inherit them from this file.** Both
reference scans `--exclude-dir` their own artifact directory: `loadability.json` names every run by
construction, so without that a scan would score itself as a dependency for all 113.
