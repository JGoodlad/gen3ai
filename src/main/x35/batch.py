"""X35 BATCH 1's GATES: the orchestrator's rulings on the 2026-10-19 adversarial review (``X35_BATCH1_REVIEW.md``),
made BEFORE any X35 launch. Pure functions over files; nothing here launches, trains or plays. ``main.x35.kit``
exposes each as a subcommand, and ``launch_x35.sh`` calls them as pre-checks (a refusal is a HOLD, exit 7).

* **AM-1, ONE PIN PER SIDE (amended by AM-8, orchestrator 2026-10-20).** :data:`PIN_FILE` (``x35_pin_batch1_<side>.txt``
  in the stage directory, one per side: ``production`` and ``toy``) is written ONCE, by the side's first GO: the pin's
  full sha, the date and the side's T62 token (:data:`T62_BY_SIDE`: production ``FP32_ALT``, the toy ``NA_TOY``, it has no
  tau solve). Every later launch OF THAT SIDE must name the same ``--pin``, and every existing batch-1 run OF THAT SIDE
  must have RECORDED that pin (``metadata.json``'s ``git_hash`` and every ``pin_history`` span, the record
  ``main.sidecar_audit`` reads); a run of that side whose pin cannot be read is a HOLD too (it is not evidence of
  agreement). The other side's runs are listed and judged at THEIR side's launches. The reason for two pins: each side's
  control (C0 / toy C0) shares its own arms' pin, which is all AM-1 protects, and the production pin must carry the
  FP32_ALT build (:func:`required_check`) that the toy never needs.
* **AM-8, THE SIDE'S OWN REQUIRED COMMITS** (:func:`required_check`). ``x35_required_commits_<side>.txt`` lists commits
  only that side's pin must carry. Production's MUST exist and name ``FP32_ALT``; while its sha is a placeholder (not
  hex) every production fork is PENDING (:func:`stage_status`) and the launch HOLDs (``kit required-check``).
* **POOL, ONE SEEDED SET.** Every fork of a side seeds its pool from the start node's run (T54: the parent's
  ``snapshot_*.zip`` at or below the fork step). The first launch of a side records that SET (names + sha256 and its
  digest) in :data:`POOL_FILE`; every later launch recomputes it and HOLDs on any difference, and the post-spawn
  check compares the fork's ``pool_seed.json`` names with the record. The pool's METADATA (``summary.json``, pruned
  by T54 but carrying the parent's LATEST scalars, F31) is recorded descriptively, never compared: X26 has ended, so
  production's cannot move, and X34 STOPPED (it plateaued, 2026-10-19), so the toy's cannot move either; each side's record is
  still written only by that side's first GO (a read-only ``pool-check`` without ``--write`` previews the digest).
* **AM-2, THE VARIANCE GATE G1** (after row 6: A1 and C0 at s3501-s3503). σ̂ = the POOLED WITHIN-ARM SD of the six
  finals' population scores S (pp; 4 df). **PASS iff σ̂ ≤ 1.2 pp**; σ̂ > 1.2, or within :data:`RULE8_EPS` of 1.2
  (rule 8: counted as above), is **HOLD**: D1, A2 and the toy calibration wait for the orchestrator's re-plan; C0
  s3504-s3505 run next and σ̂ is re-estimated on 5 C0 + 3 A1 (6 df, reported). It reads VARIANCE only, never an
  arm's Δ̂. Written once to :data:`G1_FILE`.
* **AM-9 (the order, amended 2026-10-20 from AM-7) and the conditional stages** (:data:`STAGES`): the TOY block runs
  FIRST (T1 toy C0 + X47 s3601-s3603 interleaved, then T1b = X47 and toy C0 s3604-s3605 only if X47's look 1 is
  CONTINUE, AM-4), then production once FP32_ALT has landed: P1 the first six forks -> G1 -> P2 C0 s3504-s3505 -> P3
  A1 s3504-s3505 (only if A1's look 1 is CONTINUE) -> P4 D1 (only if G1 PASSes); the toy calibration T2 stays last
  and conditional (AM-3: toy A1 only after production A1 has a FINAL class GAIN, NULL or LOSS, and G1 PASSed).
  **AM-RP-c (orchestrator, 2026-10-20): A2 and toy A2 are DROPPED** (:data:`DROPPED`): they left ``STAGES`` and
  :func:`stage_status` REFUSES them; they stay in :data:`BATCH1_ARMS` so a run of theirs would still be judged. :func:`stage_status` says, from the saved read records, whether a fork is ALLOWED, PENDING (its record does
  not exist yet) or REFUSED; the ORDER is the Training session's launch sequence (:data:`STAGES`), not a gate.
* **AM-6, K9(b) AT ARM END.** :func:`k9b_read` reads ``behaviour/max_abs_dlogp_judged`` and
  ``behaviour/p99_abs_dlogp_current`` over the fork's OWN rows (steps > its fork step: the inherited rows are
  X26's, ``gen3_tb_inherit_v1``). VOID iff any judged max ≥ 1e-3, any p99 ≥ 1e-4, any non-finite value, or the child
  log's NON-FINITE K9(b) line; a value within 1e-9 RELATIVE of a ceiling is recorded and does not fire (rule 8).
  No post-fork row, or an unknown fork step, is INCONCLUSIVE (never a PASS). A VOID arm enters no read and is re-run
  at the same pin and seed after its cause is fixed. **An OVERLAP arm (X14, ``--overlap-collect``)** is judged on
  ``behaviour_snapshot/*``, its REQUIRED family (:func:`overlap_evidence`: the run's recorded ``overlap_collect``, or
  an ``overlap/*`` or ``behaviour_snapshot/*`` row of its own): the learner's own probe has no current row on an
  overlapped update, so ``behaviour/*`` holds only each process's first, serial update and a PASS on it alone would be
  near-vacuous. Both families' rows are judged against the ceilings (a fire in either VOIDs); only the required
  family's ABSENCE is INCONCLUSIVE, the other's is "nothing judged" (``gen3_k9b_no_row_no_tag_v1``: the probe writes
  no current-row tag on an update with no current row, and a real NaN still VOIDs). The last update's rows share a
  step with the one before (the final dump writes them at the same step, ``loop._final_dump``): two real updates,
  both counted, never deduplicated.

THE READ RECORDS the stage rules read are ``<stage>/reads/<arm>_look<k>.json``: the saved JSON of
``python -m main.x35.readout decide`` (A1 / A2 and the toy pair), ``decide-d1`` (D1) or ``python -m main.x35.x47
decide`` (X47). The record names its arm and look; a record for another arm or look is a refusal.

* **BATCH 2 (F-AS-8, STAGED_PLAN §4.3)**: ``sprt`` / ``toy_sprt`` and ``lrfloor`` / ``toy_lrfloor``
  (:data:`BATCH2_ARMS`) launch at their SIDE'S written pin and share its C0. They are SIDE-PINNED
  (:data:`SIDE_PINNED_ARMS` = batch 1 + batch 2): :func:`pin_check` judges their recorded pins, :func:`k9b_batch`
  pre-checks their K9(b) and ``kit.disk_check`` counts their forks. :func:`stage_status` runs them through the AM-8
  side gate (production PENDING until FP32_ALT is recorded) and then :data:`BATCH2_STAGES`' registered conditions
  (:func:`_batch2_status`); the closed :data:`STAGES` does not list them. Their VOID guards are
  ``main.x35.batch2_void``.
* **R0, THE EXPLOITABILITY CURVE** (``main.x35.r0``; launch sheet §0 AM-R0, 2026-10-20): ``toy_r0`` / ``r0``
  (:data:`R0_ARMS`) are SIDE-PINNED like batch 2 (:func:`pin_check` judges their recorded pins, :func:`k9b_batch`
  pre-checks their K9(b)); :data:`R0_STAGES` holds them, never the closed :data:`STAGES`. :func:`stage_status` gives
  ``r0`` the AM-8 side gate (PENDING until FP32_ALT is named) and nothing else; ``toy_r0`` has no condition. Their order
  (AM-R0: toy R0 right after the GPU battery, production R0 right after P1's C0 seeds) is the Training session's, not a
  gate. ``kit.disk_check`` does not count them (a best responder's REQUIRED GiB is below an arm's, F-LS-22's rule).
* **X52, THE PRACTICE CEILING** (``main.x35.x52``; launch sheet §0 AM-X52, 2026-10-22): ``x52`` (:data:`X52_ARMS`) is
  SIDE-PINNED at the production pin like R0, held by :data:`X52_STAGES`, never :data:`STAGES`; :func:`stage_status`
  gives it the AM-8 side gate and nothing else. Its slot (after G1, before D1) is the Training session's order, not a
  gate. ``kit.disk_check`` does not count it (a +5M fork, a quarter of an arm's checkpoints).
* **AM-D1-FIX, D1'S OWN CORPUS-ONLY PIN** (launch sheet §0, 2026-10-22; the ONE exception to AM-8's one pin per side).
  D1 s3501 crash-looped on a corpus team the env's label path cannot encode (``sfv2_5051``, level 5); the corpus fix is
  code, so D1 launches at :data:`D1_PIN` = the production pin + ONE commit confined to :data:`D1_PIN_PATHS`
  (:func:`d1_pin_check`: the parent IS the side's written pin, every changed path is under the corpus package, and the
  package tree is :data:`D1_CORPORA_TREE`, the guard commit's). :func:`pin_check` with ``arm="d1"`` requires that pin
  instead of the side's; every other arm still requires the side's. D1's runs are judged against {``D1_PIN``, the side
  pin}: the side pin is only ever the VOID pre-fix attempt, since a D1 launch there is refused by this check and by
  ``kit corpus-check``. :func:`pin_write_if_absent` never writes for D1.
* **AM-D3, D3'S OWN CORPUS-ONLY PIN** (``x41_d3_setmut_2026-10-25/PREREG.md``, 2026-10-25): ``d3`` (:data:`D3_ARMS`,
  X41, set mutation) names a corpus (``d3_setmut_v1``) the production pin does not hold, so it launches at
  :data:`D3_PIN` = the production pin + ONE commit confined to ``src/utils/team_corpora/`` (D1's pattern, checked by the
  same function, :func:`arm_pin_check`, against :data:`D3_CORPORA_TREE`). Its runs are judged against ``D3_PIN`` only
  (no pre-fix attempt exists). :data:`D3_STAGES` holds it, never :data:`STAGES`; :func:`stage_status` gives it the
  AM-8 side gate (its look 1 is the orchestrator's GO, no gate) and its look 2 (s3504-s3505) only on its own look-1
  CONTINUE, as D1's. Its read record is ``reads/d3_look<k>.json``, arm label ``X41`` (``decide-d1``: D1's rule).
* **AM-N1P, N1-P'S OWN CORPUS-ONLY PIN** (``x61p_n1_production_2026-10-26/PREREG.md``, 2026-10-26; the X61 review's
  recommended design): ``n1p`` (:data:`N1P_ARMS`, X61P, Procgen-scale procedural breadth at the plateau) names
  ``n1_proc_v1``, which the production pin does not hold, so it launches at :data:`N1P_PIN` = the production pin + ONE
  commit confined to ``src/utils/team_corpora/`` that adds ONLY the corpus's three committed files and its registry
  line (not the builder, not D3's corpus, not the 10-26 GIGO fixes C0 ran without), checked by :func:`arm_pin_check`
  against :data:`N1P_CORPORA_TREE`. Its runs are judged against ``N1P_PIN`` only. :data:`N1P_STAGES` holds it, never
  :data:`STAGES`; :func:`stage_status` treats it as D3 (the AM-8 side gate; look 2 only on its own look-1 CONTINUE).
  Its read record is ``reads/n1p_look<k>.json``, arm label ``X61P`` (``decide-d1``: D1's rule on ``forum72`` alone).
* **X48, PER-TOKEN-KIND FFN EXPERTS** (``x48_kffn_production_2026-10-27/PREREG.md``, 2026-10-27): ``x48``
  (:data:`X48_ARMS`) is SIDE-PINNED at the production pin (B-KFFN is in it), held by :data:`X48_STAGES`, never
  :data:`STAGES`. It is read by N4's ONE-look screen rule at s3501-s3503, so there is no look 2 and no look-2
  licence: :func:`stage_status` gives it the AM-8 side gate and nothing else. Its read record is
  ``reads/x48_look1.json``, arm label ``X48``.
* **X63-C, THE X63 CONTROL REPLICATE** (``reviews/x63_review_2026-10-27.md`` B1 fix 3; PREREG
  ``x63c_control_replicate_2026-10-28/PREREG.md``, :data:`X63C_PREREG`): ``x63c`` (:data:`X63C_ARMS`) is X26c's
  recorded argv with only the seed (6302) and run name changed, a +100M continuation from X26 @380M (not P*), with no
  lever. It is SIDE-PINNED at the production pin (X26c's own pin ``f4b25238``), held by :data:`X63C_STAGES`, never
  :data:`STAGES`; :func:`stage_status` gives it the AM-8 side gate and its PREREG (PENDING until the file exists). Its
  pool is checked against X26c's recorded seed (``kit.reference_pool_names``), never this module's :data:`POOL_FILE`.
  Its run name is not ``rb_x35*``, so :func:`_side_pinned_dirs` lists it by name. Arm label ``X63C``.
* **AM-X63U, X63-U'S OWN CORPUS-ONLY PIN** (``design_x63_long_breadth_continuation.md`` §2; PREREG
  ``x63_2026-10-29/PREREG.md``, :data:`X63U_PREREG`): ``x63u`` (:data:`X63U_ARMS`) is X63-C's arm (X26c's recorded
  argv, X26 @380M, X26c's pool reference, +100M) + N1-P's pattern: the ONE lever ``--team-mix x63_union_v1@ALL:0.8
  --team-mix-sides both``, at :data:`X63U_PIN` = the production pin + ONE commit confined to ``src/utils/team_corpora/``
  adding only the union corpus and its ``CORPORA`` line, checked by :func:`arm_pin_check` against
  :data:`X63U_CORPORA_TREE` (filled at the registration; a None in either makes every pin check of ``x63u`` REFUSE
  and a run recorded for it unjudgeable). :data:`X63U_STAGES` holds Stage A (s2601 only),
  never :data:`STAGES`; :func:`stage_status` gives it the AM-8 side gate and its PREREG, as X63-C's. Its run name is
  ``rb_x63u_*`` (listed by name, as X63-C's). Arm label ``X63U``.
* **THE G-E READERS** (``main.x35.ge``; launch sheet §0 "G-E readers"): ``ge_a1`` / ``ge_a2`` / ``ge_d1`` / ``ge_c0``
  are not batch-1 ARMS (not in :data:`STAGES`, the disk count or AM-6's K9(b) pre-check) but they run at the SIDE'S
  pin: :func:`pin_check` judges their recorded pins with the side's batch-1 runs, and :func:`stage_status` gives a
  reader the ``ge.trigger`` condition (ALLOWED only when its arm's primary reads GAIN; ``ge_c0`` when any does).

DECLARED HERE (choices the rulings leave open; each a FINDING of this build):
* AM-1's "every existing ``rb_x35*`` run" is read as every SIDE-PINNED run OF THE LAUNCH'S SIDE
  (:func:`side_pinned_run_names`: batch 1, batch 2 and the G-E readers, a moved-aside ``<run>.<SUFFIX>`` included).
  Any other ``rb_x35*`` run (the unregistered A3 / A6 kits) is listed as not judged here.
* The pin file is ``key: value`` text with a ``side:`` line; its T62 token must be one of the side's
  :data:`T62_BY_SIDE` (the battery DECIDED FP32_ALT, 2026-10-20: production no longer launches under ``PENDING`` /
  ``ADOPT``, C1 (a) is superseded; the toy's ``NA_TOY`` says the toy makes no tau-solve call).
* G1's input is the six finals' S; a missing, duplicated or non-finite S is a refusal, never a verdict.
* D1 / A2 look 2 (3504-3505) and the toy pair's look 2 are not in the staged order; the kit allows them only on their own
  look-1 record reading CONTINUE (and AM-3 / G1 for the toy pair), and :data:`BATCH1` does not schedule them.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

RULE8_EPS = 1e-9
#: AM-1 amended by AM-8 (orchestrator 2026-10-20): ONE pin file PER SIDE, written once by the side's first GO.
PIN_FILE = "x35_pin_batch1_{side}.txt"
SIDES = ("production", "toy")
#: AM-8: commits only ONE side's pin must carry (the common list is ``x35_required_commits.txt``, read by the launch
#: script). Production's file MUST exist and name every entry of :data:`SIDE_REQUIRED_NAMES`.
SIDE_REQUIRED_FILE = "x35_required_commits_{side}.txt"
SIDE_REQUIRED_NAMES = {"production": ("FP32_ALT",), "toy": ()}
POOL_FILE = "x35_pool_batch1_{side}.json"
POOL_SCHEMA = "x35_pool_batch1_v1"
G1_FILE = "x35_gate_g1.json"
G1_SCHEMA = "x35_gate_g1_v1"
READS_DIR = "reads"
K9B_DIR = "k9b"
K9B_SCHEMA = "x35_k9b_arm_end_v1"
#: The T62 token each side's pin file may carry (orchestrator, 2026-10-20: the GPU battery DECIDED FP32_ALT, because the
#: float64 tau solve doubles the production learner update; the toy makes ZERO tau-solve calls, so its token is NA_TOY).
#: A production pin therefore carries the FP32_ALT build (``required_check``); C1 (a) (launch under T62 pending) is
#: superseded.
T62_BY_SIDE = {"production": ("FP32_ALT",), "toy": ("NA_TOY",)}
T62_VERDICTS = tuple(sorted({v for vs in T62_BY_SIDE.values() for v in vs}))   # the CLI's choices
#: AM-2's bar on σ̂, pp.
G1_SIGMA_BAR_PP = 1.2
G1_SEEDS = ("3501", "3502", "3503")
#: AM-6's ceilings (X26's Amendment 3; X34's for the toy are the same numbers, ``kit.K9B_TOY_CEILINGS``).
K9B_TAGS = {"judged_max": "behaviour/max_abs_dlogp_judged", "p99": "behaviour/p99_abs_dlogp_current"}
#: The overlap arm's REQUIRED family: the same two statistics from the behaviour-snapshot pass of the same probe
#: (``consistency.behaviour_probe(prefix="behaviour_snapshot/")``). Each key's ceiling is its :data:`K9B_CEILINGS` twin.
K9B_SNAPSHOT_TAGS = {"snapshot_judged_max": "behaviour_snapshot/max_abs_dlogp_judged",
                     "snapshot_p99": "behaviour_snapshot/p99_abs_dlogp_current"}
K9B_CEILING_KEY = {"judged_max": "judged_max", "p99": "p99",
                   "snapshot_judged_max": "judged_max", "snapshot_p99": "p99"}
#: A row of this tag (written once per overlapped update) says the arm OVERLAPPED, whatever the record says.
K9B_OVERLAP_TAG = "overlap/join_wait_ms"
K9B_CEILINGS = {"judged_max": 1e-3, "p99": 1e-4}
K9B_NONFINITE_LINE = "FATAL: a NON-FINITE |d log pi|"
ALLOWED, PENDING, REFUSED = "ALLOWED", "PENDING", "REFUSED"
PASS, HOLD, VOID, INCONCLUSIVE = "PASS", "HOLD", "VOID", "INCONCLUSIVE"
FINAL_CLASSES = ("GAIN", "NULL", "LOSS")

#: AM-9's order (orchestrator, 2026-10-20; it replaces AM-7's production-first order), stage by stage: the TOY block
#: first (T1, then T1b on X47's look-1 CONTINUE), production P1-P4 once FP32_ALT has landed, the toy calibration T2
#: last. ``BATCH1`` (kit) is this list flattened.
STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("T1", (("toy_c0", 3601), ("toy_x47", 3601), ("toy_x47", 3602), ("toy_c0", 3602), ("toy_c0", 3603),
            ("toy_x47", 3603))),
    ("T1b", (("toy_x47", 3604), ("toy_c0", 3604), ("toy_c0", 3605), ("toy_x47", 3605))),
    ("P1", (("a1", 3501), ("c0", 3501), ("c0", 3502), ("a1", 3502), ("a1", 3503), ("c0", 3503))),
    ("P2", (("c0", 3504), ("c0", 3505))),
    ("P3", (("a1", 3504), ("a1", 3505))),
    ("P4", (("d1", 3501), ("d1", 3502), ("d1", 3503))),
    ("T2", (("toy_a1", 3601), ("toy_a1", 3602), ("toy_a1", 3603))),
)
#: AM-RP-c (orchestrator, 2026-10-20, BATCH1_LAUNCH_SHEET.md §0, before any production arm): arms DROPPED from batch 1.
#: :func:`stage_status` REFUSES every seed of them; their registration text stays in the sheet as history.
DROPPED = {a: ("AM-RP-c (2026-10-20): A2 (PFSP) is DROPPED from batch 1 -- its lever tilts 1.2 % of games, R1 cannot "
               "reach NULL, its promotion count is confounded by its sentinel count (TM-1) and its PFSP state is "
               "biased (F31)") for a in ("a2", "toy_a2")}
#: The arms of batch 1 (the AM-1 pin and the POOL record cover their runs).
BATCH1_ARMS = ("c0", "a1", "a2", "d1", "toy_c0", "toy_a1", "toy_a2", "toy_x47")
#: Batch 2 (STAGED_PLAN §4.1-§4.3, F-AS-8): the SPRT promotion rule and the LR floor, each at its side's pin.
BATCH2_ARMS = ("sprt", "lrfloor", "toy_sprt", "toy_lrfloor")
#: Every arm whose runs MUST record their side's written pin (AM-1), whose finished runs AM-6 pre-checks, and whose
#: forks the disk pre-check counts. The G-E readers share the side's pin too (:func:`side_pinned_run_names`).
#: R0, the exploitability curve (``main.x35.r0``, AM-R0): one best responder per side, side-pinned.
R0_ARMS = ("toy_r0", "r0")
#: X52, the practice ceiling (``main.x35.x52``, AM-X52, 2026-10-22): one production arm, side-pinned.
X52_ARMS = ("x52",)
#: X41 / D3, set mutation (AM-D3, 2026-10-25): one production arm at its OWN corpus-only pin (:data:`D3_PIN`).
D3_ARMS = ("d3",)
#: X61P / N1-P, procedural breadth at the plateau (AM-N1P, 2026-10-26): one production arm at its OWN corpus-only pin.
N1P_ARMS = ("n1p",)
#: X48, per-token-kind FFN experts (2026-10-27): one production arm at the side pin, read by N4's one-look screen.
X48_ARMS = ("x48",)
#: X63-C, the X63 control replicate (2026-10-28): one production arm at the side pin (X26c's), forking X26 @380M.
X63C_ARMS = ("x63c",)
#: X63-C's registration: its stage-check is PENDING until this file exists (repo-relative, read in the kit's checkout).
X63C_PREREG = "designs/research_state/measurements/x63c_control_replicate_2026-10-28/PREREG.md"
#: X63-U, the X63 treatment (2026-10-29): one production arm at its OWN corpus-only pin (:data:`X63U_PIN`), forking
#: X26 @380M with X26c's argv + the union team mix.
X63U_ARMS = ("x63u",)
#: X63-U's registration: its stage-check is PENDING until this file exists (repo-relative, read in the kit's checkout).
X63U_PREREG = "designs/research_state/measurements/x63_2026-10-29/PREREG.md"
#: The arms whose stage-check is the side gate + their PREREG present (the orchestrator's GO): :func:`arm_prereg`.
PREREG_GATED_ARMS = X63C_ARMS + X63U_ARMS
SIDE_PINNED_ARMS = (BATCH1_ARMS + BATCH2_ARMS + R0_ARMS + X52_ARMS + D3_ARMS + N1P_ARMS + X48_ARMS + X63C_ARMS
                    + X63U_ARMS)
#: AM-D1-FIX (launch sheet §0, 2026-10-22): D1's pin. The production pin ``f4b25238`` + the corpus label-guard commit
#: (``fix(team corpora): refuse teams the env's label path cannot encode ...``), rebuilt onto it with only
#: ``src/utils/team_corpora/`` replaced. Kept reachable by the local branch ``x35-d1-pin``.
D1_PIN = "440b61c320efdd4e172f54a192a2c84630a4c796"
#: The production pin D1_PIN must sit directly on (the side's written ``x35_pin_batch1_production.txt``).
D1_PIN_BASE = "f4b25238def57900c0305bb2a3a5a8f9edaef285"
#: The only paths D1_PIN may change relative to D1_PIN_BASE.
D1_PIN_PATHS = ("src/utils/team_corpora/",)
#: The git tree of ``src/utils/team_corpora`` at D1_PIN: the guard commit's own (content-addressed, so it survives a
#: rebase; a patch-id would not cover the binary ``teams.jsonl.gz``).
D1_CORPORA_TREE = "e73d5a6fac6230cd15c74202be2c3f151485b281"
#: AM-D3 (2026-10-25): D3's pin. The production pin ``f4b25238`` + the D3 lever commit's ``src/utils/team_corpora/`` (the
#: ``d3_setmut_v1`` corpus, its builder and its label-path soak), built onto it with only that path replaced. Tagged on
#: origin as ``x35-d3-pin-<sha8>`` and kept reachable by the local branch ``x35-d3-pin``.
D3_PIN = "9e289ba44eb0015f6fa9c7815fc2e94f296d8f4f"
#: The production pin D3_PIN sits directly on (the same side pin as D1's).
D3_PIN_BASE = D1_PIN_BASE
#: The only paths D3_PIN may change relative to D3_PIN_BASE.
D3_PIN_PATHS = ("src/utils/team_corpora/",)
#: The git tree of ``src/utils/team_corpora`` at D3_PIN: the lever commit's own (content-addressed).
D3_CORPORA_TREE = "2156dd46c78e749feb1d3bd3f6d42b8a16beb4d8"
#: AM-N1P (2026-10-26): N1-P's pin. The production pin ``f4b25238`` + ONE commit adding only ``n1_proc_v1``'s three
#: committed files (byte for byte main's ``de269480``) and its line in ``CORPORA``. Tagged on origin as
#: ``x61p-n1p-pin-<sha8>`` and kept reachable by the local branch ``x35-n1p-pin``.
N1P_PIN = "5f8f6cfa37de8e283ba553c0d415680cf51bff18"
#: The production pin N1P_PIN sits directly on (the same side pin as D1's and D3's).
N1P_PIN_BASE = D1_PIN_BASE
#: The only paths N1P_PIN may change relative to N1P_PIN_BASE.
N1P_PIN_PATHS = ("src/utils/team_corpora/",)
#: The git tree of ``src/utils/team_corpora`` at N1P_PIN (content-addressed). No main commit holds it: it is the side
#: pin's package + the corpus subtree main's ``de269480`` landed (``batch_test`` checks that composition).
N1P_CORPORA_TREE = "b273cfe1ff1117e855bbf5f83f25f15fd755d61c"
#: AM-X63U (2026-10-29): X63-U's pin. The production pin ``f4b25238`` + ONE commit adding only ``x63_union_v1``'s
#: committed files (``manifest.json``, ``soak_selfcheck.json``, ``teams.jsonl.gz``) and its line in ``CORPORA``.
#: Tagged on origin as ``x63u-pin-39eaa326`` and kept reachable by the local branch ``x63u-pin`` (filled 2026-10-29 at
#: the registration; a None here would make every :func:`x63u_pin_check` REFUSE).
X63U_PIN: Optional[str] = "39eaa326736376a5ec3dde41af4afbe8a76a9015"
#: The production pin X63U_PIN sits directly on (X26c's own pin; the same side pin as D1's, D3's and N1-P's).
X63U_PIN_BASE = D1_PIN_BASE
#: The only paths X63U_PIN may change relative to X63U_PIN_BASE.
X63U_PIN_PATHS = ("src/utils/team_corpora/",)
#: The git tree of ``src/utils/team_corpora`` at X63U_PIN (content-addressed): the side pin's package + the union's
#: subtree + the one CORPORA line (no main commit holds it, as N1P_CORPORA_TREE).
X63U_CORPORA_TREE: Optional[str] = "ca0845cfcb22f70e7a808871f57cff480ad621db"
#: The arms with their own pin (AM-D1-FIX, AM-D3, AM-N1P, AM-X63U): ``{arm: pin}``; None = a registered placeholder
#: (every check of that arm refuses until it is filled).
ARM_PINS: Dict[str, Optional[str]] = {"d1": D1_PIN, "d3": D3_PIN, "n1p": N1P_PIN, "x63u": X63U_PIN}
#: The amendment that registered each own-pin arm (named in every line its checks print).
ARM_PIN_AMENDMENT = {"d1": "AM-D1-FIX", "d3": "AM-D3", "n1p": "AM-N1P", "x63u": "AM-X63U"}
#: The own-pin arms whose run MAY also sit at the side's pin: D1's VOID pre-fix attempt (AM-D1-FIX-f) only.
ARM_PIN_PREFIX_OK = ("d1",)
#: Batch 2's stages (F-AS-8), in AM-RP-e's slots: (6) production SPRT, (8) production LR-F, (11) the toy pair; each
#: look 2 (``b``) only on its own look-1 CONTINUE. The ORDER is the Training session's (AM-RP-e), not a gate; the
#: conditions are :func:`_batch2_status`'s. Never merged into the closed :data:`STAGES`.
BATCH2_STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("B2-SPRT", (("sprt", 3501), ("sprt", 3502), ("sprt", 3503))),
    ("B2-LRF", (("lrfloor", 3501), ("lrfloor", 3502), ("lrfloor", 3503))),
    ("B2-T", (("toy_lrfloor", 3601), ("toy_lrfloor", 3602), ("toy_lrfloor", 3603),
              ("toy_sprt", 3601), ("toy_sprt", 3602), ("toy_sprt", 3603))),
    ("B2-SPRTb", (("sprt", 3504), ("sprt", 3505))),
    ("B2-LRFb", (("lrfloor", 3504), ("lrfloor", 3505))),
    ("B2-Tb", (("toy_lrfloor", 3604), ("toy_lrfloor", 3605), ("toy_sprt", 3604), ("toy_sprt", 3605))),
)
#: AM-R0's stages (2026-10-20): toy R0 at the toy pin, production R0 at the production pin. Never merged into STAGES.
R0_STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("R0-T", (("toy_r0", 3631),)),
    ("R0-P", (("r0", 3531),)),
)
#: AM-X52's stage (2026-10-22): both seeds back to back in AM-RP-e's slot (5), after G1 and before D1. The order is the
#: Training session's, not a gate (F-RP-6). Never merged into the closed STAGES.
X52_STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("X52", (("x52", 3501), ("x52", 3502))),
)
#: AM-D3's stages (2026-10-25): look 1 (s3501-s3503) on the orchestrator's GO, look 2 (s3504-s3505) only on D3's own
#: look-1 CONTINUE. C0 s3501-s3505 (shared, already run at the side pin) is the control. Never merged into STAGES.
D3_STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("D3", (("d3", 3501), ("d3", 3502), ("d3", 3503))),
    ("D3b", (("d3", 3504), ("d3", 3505))),
)
#: AM-N1P's stages (2026-10-26): as D3's. C0 s3501-s3505 (shared, already run at the side pin) is the control.
N1P_STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("N1P", (("n1p", 3501), ("n1p", 3502), ("n1p", 3503))),
    ("N1Pb", (("n1p", 3504), ("n1p", 3505))),
)
#: X48's stage (2026-10-27, PREREG x48_kffn_production_2026-10-27 §3): one look, three seeds, on the orchestrator's GO.
#: C0 s3501-s3503 (already run at the side pin) is the control. Never merged into STAGES.
X48_STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("X48", (("x48", 3501), ("x48", 3502), ("x48", 3503))),
)
#: X63-C's stage (2026-10-28, review B1 fix 3): one seed, on the orchestrator's GO once its PREREG exists. Its comparators
#: are X26c (s2601) and X63's own continuation, not C0. Never merged into STAGES.
X63C_STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("X63C", (("x63c", 6302),)),
)
#: X63-U's Stage A (2026-10-29): seed 2601 only, on the orchestrator's GO once its PREREG exists. Its control is X26c
#: (banked, the same seed); Stage B / C seeds are later amendments. Never merged into STAGES.
X63U_STAGES: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = (
    ("X63U-A", (("x63u", 2601),)),
)
#: The arm label each read record carries (``readout`` A1 / A2 / SPRT / LRF, ``d1`` X38, ``d3`` X41, ``n1p`` X61P,
#: ``x47`` X47, ``x48`` X48, ``x63c`` X63C, ``x63u`` X63U).
READ_LABEL = {"a1": "A1", "a2": "A2", "d1": "X38", "d3": "X41", "n1p": "X61P", "toy_a1": "A1", "toy_a2": "A2", "toy_x47": "X47",
              "sprt": "SPRT", "toy_sprt": "SPRT", "lrfloor": "LRF", "toy_lrfloor": "LRF", "x48": "X48", "x63c": "X63C",
              "x63u": "X63U"}
#: The toy arms whose look-1 CONTINUE licenses toy C0's look-2 seeds s3604-s3605 (their shared control).
TOY_LOOK2_LICENSORS = ("toy_x47", "toy_a1", "toy_a2", "toy_lrfloor", "toy_sprt")
#: Where a read record keeps its PRIMARY when it is not R1 (``d1.D1Verdict``: H).
PRIMARY_KEY = {"d1": "h", "d3": "h", "n1p": "h"}
CONTROLS = ("c0", "toy_c0")
LOOK2_SEEDS = {"production": (3504, 3505), "toy": (3604, 3605)}


class BatchError(ValueError):
    """A gate's input is unreadable or invalid (named; the launch script HOLDs on it)."""


def _near(x: float, b: float) -> bool:
    return abs(x - b) <= RULE8_EPS


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def side_pinned_run_names(*, readers: bool = False) -> Dict[str, Tuple[str, int]]:
    """``{run name: (arm, seed)}`` of every side-pinned arm (:data:`SIDE_PINNED_ARMS`: batch 1 + batch 2) at every
    registered seed; with ``readers`` the G-E readers' runs too (they share the side's pin, ``main.x35.ge``)."""
    from main.x35.ge import READER_ARMS
    from main.x35.kit import ARMS, run_name

    arms = SIDE_PINNED_ARMS + (tuple(READER_ARMS) if readers else ())
    return {run_name(a, s): (a, s) for a in arms for s in ARMS[a].seeds}


def side_of_run(base: str) -> str:
    """The side (``production`` / ``toy``) of a side-pinned (or G-E reader) run name."""
    from main.x35.kit import ARMS

    return ARMS[side_pinned_run_names(readers=True)[base][0]].side


def _side_pinned_dirs(models_dir: str, *, readers: bool = False) -> Tuple[List[Tuple[str, str]], List[str]]:
    """``([(dir name, run name)], [other rb_x35* dirs])``; ``<run>.<SUFFIX>`` (moved aside) counts as its run. A
    side-pinned run not named ``rb_x35*`` (X63-C's ``rb_x63c_*``, X63-U's ``rb_x63u_*``) is found by its registered
    name."""
    names = side_pinned_run_names(readers=readers)
    ours: List[Tuple[str, str]] = []
    other: List[str] = []
    if not os.path.isdir(models_dir):
        return ours, other
    for d in sorted(os.listdir(models_dir)):
        if not (d.startswith("rb_x35") or d.split(".", 1)[0] in names) or not os.path.isdir(os.path.join(models_dir, d)):
            continue
        base = d.split(".", 1)[0]
        if base in names:
            ours.append((d, base))
        else:
            other.append(d)
    return ours, other


# ----------------------------------------------------------------------------- AM-1 (amended): one pin PER SIDE
def _side(side: str) -> str:
    if side not in SIDES:
        raise BatchError(f"side {side!r} is not one of {SIDES}")
    return side


def pin_path(stage: str, side: str) -> str:
    """The side's pin file (``x35_pin_batch1_<side>.txt`` in the stage directory)."""
    return os.path.join(stage, PIN_FILE.format(side=_side(side)))


def pin_label(side: str) -> str:
    """How messages name a side's pin."""
    return f"X35_PIN_1_{_side(side)}"


def read_pin_file(path: str, side: str) -> Optional[Dict[str, str]]:
    """The side's pin file's ``key: value`` lines, or None when it does not exist. A file without a 40-hex ``sha``, one
    that names another side, or one whose T62 token is not the side's (:data:`T62_BY_SIDE`) is a refusal (a corrupt
    record never reads as "no pin yet")."""
    _side(side)
    if not os.path.exists(path):
        return None
    out: Dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", out.get("sha", "")):
        raise BatchError(f"{path}: no 40-hex 'sha:' line ({out.get('sha')!r})")
    if out.get("side") != side:
        raise BatchError(f"{path}: the record says side {out.get('side')!r}, it is the {side} side's file")
    if out.get("t62_verdict") not in T62_BY_SIDE[side]:
        raise BatchError(f"{path}: T62 token {out.get('t62_verdict')!r} is not a {side} token {T62_BY_SIDE[side]}")
    return out


def write_pin_file(path: str, sha: str, t62: str, side: str, now: Optional[str] = None) -> Dict[str, str]:
    """Write the side's pin ONCE (refuses an existing file and a T62 token that is not the side's)."""
    _side(side)
    if os.path.exists(path):
        raise BatchError(f"{path} exists: the {side} pin is written ONCE")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise BatchError(f"the pin {sha!r} is not a full 40-hex sha")
    if t62 not in T62_BY_SIDE[side]:
        raise BatchError(f"T62 token {t62!r} is not a {side} token {T62_BY_SIDE[side]}")
    doc = {"sha": sha, "side": side, "written_at": now or _now(), "t62_verdict": t62,
           "rule": f"AM-1 as amended by AM-8 (orchestrator 2026-10-20): every batch-1 {side} arm, look 2 included, "
                   f"launches at this pin"}
    Path(path).write_text(f"# X35 batch 1's {side.upper()} pin ({pin_label(side)}), written once by the side's first "
                          "launch's GO.\n" + "".join(f"{k}: {v}\n" for k, v in doc.items()))
    return doc


def recorded_pins(run_dir: str) -> List[str]:
    """Every git hash a run RECORDED as its code: ``metadata.json``'s ``git_hash`` and each ``pin_history`` span
    (``main.sidecar_audit.audit_run``'s fields). Empty when the run has written no metadata yet."""
    p = os.path.join(run_dir, "metadata.json")
    if not os.path.isfile(p):
        return []
    try:
        meta = json.loads(Path(p).read_text())
    except (OSError, ValueError) as exc:
        raise BatchError(f"{p}: unreadable ({exc})") from exc
    hs = [str(meta["git_hash"])] if meta.get("git_hash") else []
    hs += [str(e.get("git_hash")) for e in meta.get("pin_history") or [] if isinstance(e, dict) and e.get("git_hash")]
    return hs


def _same_sha(a: str, b: str) -> bool:
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


def own_pin_check(arm: str, pin: str, side_pin: str, repo: str, *, want_pin: Optional[str], base: str,
                  paths: Tuple[str, ...], tree_want: Optional[str], tree_of: str) -> Tuple[bool, List[str]]:
    """An OWN-PIN arm's check (AM-D1-FIX, AM-D3): ``pin`` is ``want_pin``, sits directly on the side's written pin
    (``base``), changes only paths under ``paths``, and carries the registered corpus tree ``tree_want`` (``tree_of``
    names whose tree it is). A ``want_pin`` or ``tree_want`` of None is a registered PLACEHOLDER: a refusal."""
    am, tag = ARM_PIN_AMENDMENT.get(arm, arm), arm.upper()
    unset = [n for n, v in ((f"{tag}_PIN", want_pin), (f"{tag}_CORPORA_TREE", tree_want)) if v is None]
    if unset:
        return False, [f"{am}: {' and '.join(unset)} {'is a PLACEHOLDER' if len(unset) == 1 else 'are PLACEHOLDERS'} "
                       f"(unset): {arm} launches nowhere until its own pin is registered"]
    assert want_pin is not None and tree_want is not None

    def git(*args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)

    if pin != want_pin:
        return False, [f"{am}: {arm} launches only at {tag}_PIN {want_pin[:12]}, not {pin[:12]}"]
    if side_pin != base:
        return False, [f"{am}: {tag}_PIN sits on {base[:12]}, but the side's written pin is {side_pin[:12]}"]
    parents = git("rev-list", "--parents", "-n", "1", pin)
    if parents.returncode != 0:
        return False, [f"{am}: {tag}_PIN {pin[:12]} is not a commit in {repo}: {parents.stderr.strip()[-200:]}"]
    if parents.stdout.split()[1:] != [side_pin]:
        return False, [f"{am}: {tag}_PIN's parent(s) {[h[:12] for h in parents.stdout.split()[1:]]} != the side's "
                       f"pin {side_pin[:12]} (one commit on top of it, no more)"]
    names = [n for n in git("diff", "--name-only", side_pin, pin).stdout.splitlines() if n]
    outside = [n for n in names if not n.startswith(paths)]
    if not names or outside:
        return False, [f"{am}: {tag}_PIN must change only {list(paths)} relative to {side_pin[:12]}; it "
                       f"changes {len(names)} file(s), outside: {outside[:10]}"]
    tree = git("rev-parse", f"{pin}:src/utils/team_corpora").stdout.strip()
    if tree != tree_want:
        return False, [f"{am}: {tag}_PIN's src/utils/team_corpora tree {tree[:12]} != the {tree_of} "
                       f"{tree_want[:12]}"]
    return True, [f"{am}: {tag}'s pin {pin[:12]} = the side's pin {side_pin[:12]} + one commit, {len(names)} file(s), "
                  f"all under {list(paths)}, corpus tree {tree[:12]} (the {tree_of})"]


def d1_pin_check(pin: str, side_pin: str, repo: str) -> Tuple[bool, List[str]]:
    """AM-D1-FIX: ``pin`` is :data:`D1_PIN`, sits directly on the side's written pin (:data:`D1_PIN_BASE`), changes only
    paths under :data:`D1_PIN_PATHS`, and carries the guard commit's corpus tree (:data:`D1_CORPORA_TREE`)."""
    return own_pin_check("d1", pin, side_pin, repo, want_pin=D1_PIN, base=D1_PIN_BASE, paths=D1_PIN_PATHS,
                         tree_want=D1_CORPORA_TREE, tree_of="guard commit's")


def d3_pin_check(pin: str, side_pin: str, repo: str) -> Tuple[bool, List[str]]:
    """AM-D3: ``pin`` is :data:`D3_PIN`, sits directly on the side's written pin (:data:`D3_PIN_BASE`), changes only
    paths under :data:`D3_PIN_PATHS`, and carries the lever commit's corpus tree (:data:`D3_CORPORA_TREE`)."""
    return own_pin_check("d3", pin, side_pin, repo, want_pin=D3_PIN, base=D3_PIN_BASE, paths=D3_PIN_PATHS,
                         tree_want=D3_CORPORA_TREE, tree_of="lever commit's")


def n1p_pin_check(pin: str, side_pin: str, repo: str) -> Tuple[bool, List[str]]:
    """AM-N1P: ``pin`` is :data:`N1P_PIN`, sits directly on the side's written pin (:data:`N1P_PIN_BASE`), changes
    only paths under :data:`N1P_PIN_PATHS`, and carries the registered corpus tree (:data:`N1P_CORPORA_TREE`)."""
    return own_pin_check("n1p", pin, side_pin, repo, want_pin=N1P_PIN, base=N1P_PIN_BASE, paths=N1P_PIN_PATHS,
                         tree_want=N1P_CORPORA_TREE, tree_of="registered N1-P tree")


def x63u_pin_check(pin: str, side_pin: str, repo: str) -> Tuple[bool, List[str]]:
    """AM-X63U: ``pin`` is :data:`X63U_PIN`, sits directly on the side's written pin (:data:`X63U_PIN_BASE`), changes
    only paths under :data:`X63U_PIN_PATHS`, and carries the registered union tree (:data:`X63U_CORPORA_TREE`). REFUSES
    while either constant is its None placeholder."""
    return own_pin_check("x63u", pin, side_pin, repo, want_pin=X63U_PIN, base=X63U_PIN_BASE, paths=X63U_PIN_PATHS,
                         tree_want=X63U_CORPORA_TREE, tree_of="registered X63-U union tree")


def arm_pin_check(arm: str, pin: str, side_pin: str, repo: str) -> Tuple[bool, List[str]]:
    """The own-pin check of ``arm`` (:data:`ARM_PINS`)."""
    return {"d1": d1_pin_check, "d3": d3_pin_check, "n1p": n1p_pin_check,
            "x63u": x63u_pin_check}[arm](pin, side_pin, repo)


def pin_check(pin: str, stage: str, models_dir: str, side: str, t62: Optional[str] = None,
              arm: Optional[str] = None, repo: Optional[str] = None) -> Tuple[bool, List[str]]:
    """AM-1's pre-check for one SIDE: ``(ok, lines)``. ``pin`` is the full sha the launch would use; ``t62`` (when typed)
    must be the side's token. Only this side's side-pinned runs (batch 1, batch 2, the G-E readers) are judged against
    the side's pin; the other side's are listed (their own side's launches judge them). ``arm`` (the launching arm) with
    its own pin (:data:`ARM_PINS`, AM-D1-FIX) must launch at THAT pin, verified against the side's by
    :func:`d1_pin_check` in ``repo`` (default: this checkout)."""
    lines: List[str] = []
    ok = True
    label, fname = pin_label(side), os.path.basename(pin_path(stage, side))
    if t62 is not None and t62 not in T62_BY_SIDE[side]:
        ok = False
        lines.append(f"--t62 {t62!r} is not a {side} token {T62_BY_SIDE[side]}")
    rec = read_pin_file(pin_path(stage, side), side)
    if arm in ARM_PINS:
        if rec is None:
            ok = False
            lines.append(f"{ARM_PIN_AMENDMENT.get(arm, arm)}: {arm}'s own pin rests on {label}, and no {fname} is "
                         "written yet")
        else:
            from utils.paths import repo_root

            ok_arm, arm_lines = arm_pin_check(arm, pin, rec["sha"], repo or str(repo_root()))
            ok = ok and ok_arm
            lines += arm_lines
    elif rec is None:
        lines.append(f"no {fname} yet: the first {side} GO writes {pin[:12]} as {label}")
    elif rec["sha"] != pin:
        ok = False
        lines.append(f"the pin {pin[:12]} != {label} {rec['sha'][:12]} ({fname}, written {rec.get('written_at')})")
    else:
        lines.append(f"the pin equals {label} {pin[:12]} ({fname}, T62 {rec.get('t62_verdict')})")
    want = rec["sha"] if rec is not None else pin
    names = side_pinned_run_names(readers=True)
    ours, other = _side_pinned_dirs(models_dir, readers=True)
    mine = [(d, base) for d, base in ours if side_of_run(base) == side]
    for d, base in mine:
        hs = recorded_pins(os.path.join(models_dir, d))
        if not hs:
            ok = False
            lines.append(f"{d}: no recorded pin (metadata.json absent or without git_hash): cannot show it is {label}")
            continue
        run_arm = names[base][0]
        own = ARM_PINS.get(run_arm)
        if run_arm in ARM_PINS and own is None:              # a registered placeholder: no pin to judge the run by
            ok = False
            lines.append(f"{d}: {ARM_PIN_AMENDMENT.get(run_arm, run_arm)}: {run_arm}'s own pin is a PLACEHOLDER "
                         f"(unset), so its recorded pin(s) {[h[:12] for h in hs]} cannot be judged")
            continue
        allowed = ((want, own) if run_arm in ARM_PIN_PREFIX_OK else (own,)) if own is not None else (want,)
        bad = sorted({h for h in hs if not any(_same_sha(h, a) for a in allowed)})
        if bad:
            ok = False
            lines.append(f"{d}: recorded pin(s) {[h[:12] for h in bad]} != {' / '.join(a[:12] for a in allowed)}")
        elif run_arm in ARM_PINS and any(_same_sha(h, want) for h in hs):
            lines.append(f"{d}: a PRE-FIX {run_arm} attempt at {label} {want[:12]} (AM-D1-FIX-f: VOID, never read)")
    if mine:
        lines.append(f"{len(mine)} side-pinned {side} run(s) checked against {want[:12]}")
    elsewhere = [d for d, base in ours if side_of_run(base) != side]
    if elsewhere:
        lines.append(f"the other side's run(s), judged at its own launches: {elsewhere}")
    if other:
        lines.append(f"not a side-pinned arm (batch 1 / 2 / G-E reader), not judged here: {other}")
    return ok, lines


def pin_write_if_absent(pin: str, stage: str, side: str, t62: Optional[str], arm: Optional[str] = None) -> str:
    """At GO, after every pre-check: write the side's pin file if it does not exist (needs ``t62``), else confirm it. An
    arm with its own pin (AM-D1-FIX) never writes: it confirms its pin and the side's record it rests on."""
    path = pin_path(stage, side)
    label, fname = pin_label(side), os.path.basename(path)
    rec = read_pin_file(path, side)
    if arm in ARM_PINS:
        own = ARM_PINS[arm]
        if own is None:
            raise BatchError(f"{ARM_PIN_AMENDMENT.get(arm, arm)}: {arm}'s own pin is a PLACEHOLDER (unset): nothing "
                             f"launches or is recorded for it until the pin is registered")
        if pin != own or rec is None:
            raise BatchError(f"{ARM_PIN_AMENDMENT.get(arm, arm)}: {arm} launches at {own[:12]} on a written {label} "
                             f"(got {pin[:12]}, {fname} {'absent' if rec is None else rec['sha'][:12]})")
        return f"{arm}'s own pin {pin[:12]} ({ARM_PIN_AMENDMENT.get(arm, arm)}) on {label} {rec['sha'][:12]}; {fname} unchanged"
    if rec is not None:
        if rec["sha"] != pin:
            raise BatchError(f"the pin {pin[:12]} != {label} {rec['sha'][:12]}")
        return f"{label} {pin[:12]} already recorded"
    if t62 is None:
        raise BatchError(f"the first {side} GO writes {fname} and needs --t62 <{'|'.join(T62_BY_SIDE[side])}>")
    write_pin_file(path, pin, t62, side)
    return f"{label} {pin[:12]} WRITTEN to {fname} (T62 {t62})"


# ------------------------------------------------------------- AM-8: the side's own required commits (FP32_ALT)
def read_side_required(stage: str, side: str) -> List[Tuple[str, str]]:
    """``[(sha-or-placeholder, name)]`` of ``x35_required_commits_<side>.txt`` (``<sha> <name> [note]`` lines, ``#``
    comments). A side that must name commits (:data:`SIDE_REQUIRED_NAMES`) and has no file, or whose file lacks one of
    the names, is a refusal: deleting the file is never a way past the check."""
    path = os.path.join(stage, SIDE_REQUIRED_FILE.format(side=_side(side)))
    need = SIDE_REQUIRED_NAMES[side]
    if not os.path.exists(path):
        if need:
            raise BatchError(f"{path} is missing: the {side} side must name its {list(need)} commit (AM-8)")
        return []
    rows: List[Tuple[str, str]] = []
    for line in Path(path).read_text().splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        parts = t.split(None, 2)
        rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    lacking = [n for n in need if n not in {name for _, name in rows}]
    if lacking:
        raise BatchError(f"{path} names no {lacking} line (AM-8: the {side} side's required commit)")
    return rows


def _is_commit_sha(sha: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{7,40}", sha) is not None


def side_gate(stage: str, side: str) -> Tuple[bool, str]:
    """``(filled, why)``: every commit the side must carry has a real sha written (no placeholder). Reads the file only."""
    try:
        rows = read_side_required(stage, side)
    except BatchError as exc:
        return False, str(exc)
    unfilled = [name for sha, name in rows if not _is_commit_sha(sha)]
    if unfilled:
        return False, (f"AM-8: the {side} side's {unfilled} commit is not recorded yet (x35_required_commits_{side}.txt "
                       "holds a placeholder)")
    return True, f"the {side} side's required commit(s) are recorded ({[name for _, name in rows] or 'none'})"


def _git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def required_check(stage: str, side: str, pin: str, repo: str) -> Tuple[bool, List[str]]:
    """AM-8's pre-check: every commit of the side's own list is a real commit of ``repo`` AND an ancestor of ``pin``
    (the common list ``x35_required_commits.txt`` stays the launch script's own loop). A placeholder sha, a missing
    side file for production, or a pin lacking the commit is a HOLD."""
    rows = read_side_required(stage, side)
    ok, lines = True, []
    for sha, name in rows:
        if not _is_commit_sha(sha):
            ok = False
            lines.append(f"{name}: no commit recorded yet ({sha!r}): the {side} GO is refused until "
                         f"x35_required_commits_{side}.txt names it")
            continue
        full = _git(repo, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
        if full.returncode != 0:
            ok = False
            lines.append(f"{name} {sha}: does not name a commit of {repo}")
        elif _git(repo, "merge-base", "--is-ancestor", full.stdout.strip(), pin).returncode != 0:
            ok = False
            lines.append(f"{name} {sha[:12]}: the pin {pin[:12]} lacks it (not an ancestor)")
        else:
            lines.append(f"{name} {sha[:12]} is in the pin {pin[:12]}")
    return ok, lines or [f"no {side}-only required commits"]


# --------------------------------------------------------------------------------------- POOL: one seeded set
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected_pool(parent_run_dir: str, fork_step: int) -> Dict[str, Any]:
    """The set T54's ``seed_pool`` copies from the parent (its own filter: every ``snapshot_*.zip`` whose step is
    ``<= fork_step`` or unparseable), with each file's sha256 and the set's digest."""
    from agents.training.pool_seed import POOL_METADATA_FILES, snapshot_step, snapshot_zips

    src = os.path.join(parent_run_dir, "snapshots")
    snaps = []
    for z in snapshot_zips(src):
        st = snapshot_step(z)
        if st is not None and st > fork_step:
            continue
        snaps.append({"name": os.path.basename(z), "sha256": sha256_of(z)})
    if not snaps:
        raise BatchError(f"{src}: no snapshot at or below {fork_step} (T54 would refuse the fork)")
    digest = hashlib.sha256("".join(f"{s['name']} {s['sha256']}\n" for s in snaps).encode()).hexdigest()
    meta = {n: sha256_of(os.path.join(src, n)) for n in POOL_METADATA_FILES if os.path.isfile(os.path.join(src, n))}
    return {"parent_run_dir": os.path.abspath(parent_run_dir), "fork_step": int(fork_step), "snapshots": snaps,
            "n_snapshots": len(snaps), "digest": digest, "metadata_sha256_descriptive": meta}


def pool_check(stage: str, side: str, call: Mapping[str, Any], parent_run_dir: str, *,
               write: bool = False) -> Tuple[bool, str]:
    """The POOL pre-check of one launch: recompute the seeded set and compare it with the side's record (or, on the
    first GO of a side, write the record). ``(ok, line)``."""
    path = os.path.join(stage, POOL_FILE.format(side=side))
    p = call["pstar"]
    now = expected_pool(parent_run_dir, int(p["step"]))
    if not os.path.exists(path):
        if not write:
            return True, (f"no {os.path.basename(path)} yet: the first {side} GO records {now['n_snapshots']} "
                          f"snapshot(s), digest {now['digest'][:12]}")
        doc = dict(now, schema=POOL_SCHEMA, side=side, pstar_sha256=p["sha256"], written_at=_now())
        Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return True, f"pool RECORDED: {now['n_snapshots']} snapshot(s), digest {now['digest'][:12]}"
    rec = json.loads(Path(path).read_text())
    probs = []
    if rec.get("schema") != POOL_SCHEMA:
        probs.append(f"schema {rec.get('schema')!r}")
    if rec.get("pstar_sha256") != p["sha256"]:
        probs.append(f"the record is for start node {str(rec.get('pstar_sha256'))[:12]}, not {str(p['sha256'])[:12]}")
    if rec.get("digest") != now["digest"]:
        was = {s["name"]: s["sha256"] for s in rec.get("snapshots") or []}
        is_ = {s["name"]: s["sha256"] for s in now["snapshots"]}
        probs.append(f"the seeded set moved: digest {str(rec.get('digest'))[:12]} -> {now['digest'][:12]}; "
                     f"gone {sorted(set(was) - set(is_))}, new {sorted(set(is_) - set(was))}, "
                     f"changed {sorted(n for n in set(was) & set(is_) if was[n] != is_[n])}")
    if probs:
        return False, "; ".join(probs)
    return True, f"pool set == the {side} record ({now['n_snapshots']} snapshot(s), digest {now['digest'][:12]})"


def recorded_pool_names(stage: str, side: str) -> Optional[List[str]]:
    """The recorded seeded set's names (the post-spawn comparison), or None when no record exists."""
    path = os.path.join(stage, POOL_FILE.format(side=side))
    if not os.path.exists(path):
        return None
    return [s["name"] for s in json.loads(Path(path).read_text())["snapshots"]]


# ------------------------------------------------------------------------------------- AM-2: the variance gate
@dataclass(frozen=True)
class G1:
    verdict: str
    sigma_pp: float
    df: int
    near: bool
    reestimate: Optional[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {"schema": G1_SCHEMA, "verdict": self.verdict, "sigma_hat_pp": self.sigma_pp, "df": self.df,
                "bar_pp": G1_SIGMA_BAR_PP, "near_bar": self.near, "reestimate": self.reestimate,
                "rule": "AM-2: PASS iff the pooled within-arm SD of S over A1 / C0 s3501-s3503 <= 1.2 pp "
                        "(within 1e-9 of 1.2 counts as above); HOLD holds D1, A2 and the toy calibration"}


def pooled_sd(groups: Sequence[Sequence[float]]) -> Tuple[float, int]:
    """The pooled within-group SD and its df (Σ (n_g − 1))."""
    ss = math.fsum(math.fsum((x - math.fsum(g) / len(g)) ** 2 for x in g) for g in groups)
    df = sum(len(g) - 1 for g in groups)
    if df <= 0:
        raise BatchError("no within-group degrees of freedom")
    return math.sqrt(ss / df), df


def _scores(d: Mapping[str, Any], arm: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise BatchError(f"{arm} seed {k}: S {v!r} is not a finite number")
        out[str(k)] = float(v)
    return out


def gate_g1(doc: Mapping[str, Any]) -> G1:
    """AM-2 from ``{"a1": {seed: S_pp}, "c0": {seed: S_pp}}`` (C0 may carry 3504 / 3505: then the 6-df re-estimate
    on 5 C0 + 3 A1 is reported; the verdict reads s3501-s3503 only)."""
    a1, c0 = _scores(doc.get("a1") or {}, "A1"), _scores(doc.get("c0") or {}, "C0")
    for arm, d in (("A1", a1), ("C0", c0)):
        miss = [s for s in G1_SEEDS if s not in d]
        if miss:
            raise BatchError(f"{arm}: no S for seed(s) {miss} (G1 reads the six finals of rows 1-6)")
    extra = sorted(set(a1) - set(G1_SEEDS))
    if extra:
        raise BatchError(f"A1 seeds {extra}: G1 reads A1 s3501-s3503 only (A1's look 2 is not an input)")
    sd, df = pooled_sd([[a1[s] for s in G1_SEEDS], [c0[s] for s in G1_SEEDS]])
    near = _near(sd, G1_SIGMA_BAR_PP)
    verdict = PASS if (sd < G1_SIGMA_BAR_PP and not near) else HOLD
    re = None
    if set(c0) - set(G1_SEEDS):
        sd2, df2 = pooled_sd([[a1[s] for s in G1_SEEDS], [c0[s] for s in sorted(c0)]])
        re = {"sigma_hat_pp": sd2, "df": df2, "c0_seeds": sorted(c0), "a1_seeds": list(G1_SEEDS),
              "reported_only": True}
    return G1(verdict, sd, df, near, re)


def write_g1(stage: str, g: G1, doc: Mapping[str, Any]) -> str:
    path = os.path.join(stage, G1_FILE)
    if os.path.exists(path):
        raise BatchError(f"{path} exists: G1 is written ONCE")
    Path(path).write_text(json.dumps(dict(g.as_dict(), input=doc, written_at=_now()), indent=1, sort_keys=True)
                          + "\n")
    return path


def read_g1(stage: str) -> Optional[str]:
    """G1's recorded verdict, or None when it is not read yet."""
    path = os.path.join(stage, G1_FILE)
    if not os.path.exists(path):
        return None
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != G1_SCHEMA or doc.get("verdict") not in (PASS, HOLD):
        raise BatchError(f"{path}: not a G1 record")
    return str(doc["verdict"])


# ---------------------------------------------------------- AM-9 / AM-3 / AM-4 / AM-8: which fork may launch now
def read_record(stage: str, arm: str, look: int) -> Optional[Dict[str, Any]]:
    """The saved look-``look`` read of ``arm`` (``reads/<arm>_look<k>.json``), checked to be that arm's and look's."""
    path = os.path.join(stage, READS_DIR, f"{arm}_look{look}.json")
    if not os.path.exists(path):
        return None
    doc = json.loads(Path(path).read_text())
    want = READ_LABEL.get(arm)
    if str(doc.get("arm")) != want or int(doc.get("look", -1)) != look:
        raise BatchError(f"{path}: the record is arm {doc.get('arm')!r} look {doc.get('look')!r}, not {want} look "
                         f"{look}")
    return dict(doc)


def final_class(stage: str, arm: str) -> Tuple[Optional[str], str]:
    """``(class, why)``: the production arm's FINAL R1 class (AM-3), from its latest saved look. ``None`` = none yet."""
    for look in (2, 1):
        rec = read_record(stage, arm, look)
        if rec is None:
            continue
        r1 = rec.get(PRIMARY_KEY.get(arm, "r1")) or {}
        if not r1.get("final"):
            return None, f"{arm} look {look}: R1 is not final (outcome {rec.get('verdict')})"
        return str(r1.get("class")), f"{arm} look {look}: R1 FINAL class {r1.get('class')}"
    return None, f"{arm}: no saved read"


def _look2_licence(stage: str, arm: str) -> Tuple[str, str]:
    rec = read_record(stage, arm, 1)
    if rec is None:
        return PENDING, f"{arm} look 1 is not read yet (look 2 runs only on CONTINUE)"
    if rec.get("verdict") == "CONTINUE":
        return ALLOWED, f"{arm} look 1 is CONTINUE"
    return REFUSED, f"{arm} look 1 is {rec.get('verdict')}, not CONTINUE: no look 2"


def stage_of(arm: str, seed: int) -> Optional[str]:
    for name, forks in (STAGES + BATCH2_STAGES + R0_STAGES + X52_STAGES + D3_STAGES + N1P_STAGES + X48_STAGES
                        + X63C_STAGES + X63U_STAGES):
        if (arm, seed) in forks:
            return name
    return None


def arm_prereg(arm: str) -> Tuple[str, str]:
    """``(PREREG path, tag)`` of a PREREG-gated arm (:data:`PREREG_GATED_ARMS`), read from the constants now."""
    return {"x63c": (X63C_PREREG, "X63-C"), "x63u": (X63U_PREREG, "X63-U")}[arm]


def stage_status(stage: str, arm: str, seed: int) -> Tuple[str, str]:
    """ALLOWED / PENDING / REFUSED for ``arm`` × ``seed`` now (module docs), with the reason. Production forks are
    PENDING until the side's FP32_ALT commit is named (AM-8); the toy side has no such condition."""
    from main.x35.ge import READER_ARMS, trigger
    from main.x35.kit import ARMS

    if arm in READER_ARMS:                   # a G-E reader: only on its arm's GAIN (main.x35.ge), after AM-8
        return trigger(stage, arm)
    if arm in BATCH2_ARMS:                   # F-AS-8: the AM-8 side gate, then batch 2's registered conditions
        return _batch2_status(stage, arm, seed)
    if arm in R0_ARMS + X52_ARMS + X48_ARMS:  # AM-R0 / AM-X52 / X48: the side gate only (production: FP32_ALT, AM-8)
        if ARMS[arm].side == "production":
            filled, why = side_gate(stage, "production")
            if not filled:
                return PENDING, why
        tag = ("AM-X52" if arm in X52_ARMS else "X48 PREREG: one look, the orchestrator's GO" if arm in X48_ARMS
               else "AM-R0")
        return ALLOWED, f"{arm} s{seed}: no unmet condition (stage {stage_of(arm, seed)}, {tag})"
    if arm in PREREG_GATED_ARMS:             # X63-C / X63-U: the side gate, then its PREREG (the orchestrator's GO)
        filled, why = side_gate(stage, "production")
        if not filled:
            return PENDING, why
        from utils.paths import repo_path

        prereg, tag = arm_prereg(arm)
        if not repo_path(prereg).is_file():
            return PENDING, f"{arm} s{seed}: its PREREG {prereg} is not in this checkout yet"
        return ALLOWED, f"{arm} s{seed}: no unmet condition (stage {stage_of(arm, seed)}, {tag} PREREG present)"
    if arm in D3_ARMS + N1P_ARMS:            # AM-D3 / AM-N1P: the side gate; look 2 only on its own look-1 CONTINUE
        filled, why = side_gate(stage, "production")
        if not filled:
            return PENDING, why
        if seed in LOOK2_SEEDS["production"]:
            return _look2_licence(stage, arm)
        return ALLOWED, f"{arm} s{seed}: no unmet condition (stage {stage_of(arm, seed)}, " \
                        f"{ARM_PIN_AMENDMENT[arm]}: the orchestrator's GO)"
    if arm not in BATCH1_ARMS:
        return ALLOWED, f"{arm} is not a staged arm (no AM-9 or batch-2 condition)"
    if arm in DROPPED:
        return REFUSED, DROPPED[arm]
    side = ARMS[arm].side
    if side == "production":                 # AM-8: no production fork before its FP32_ALT commit is named
        filled, why = side_gate(stage, side)
        if not filled:
            return PENDING, why
    look2 = seed in LOOK2_SEEDS[side]
    if arm == "c0" and look2:
        g = read_g1(stage)
        return (ALLOWED, f"G1 is read ({g})") if g else (PENDING, "C0's look-2 seeds run after G1 (AM-2 / AM-9)")
    if arm == "toy_c0" and look2:
        why = []
        for b in TOY_LOOK2_LICENSORS:
            st, w = _look2_licence(stage, b)
            if st == ALLOWED:
                return ALLOWED, w
            why.append(w)
        return PENDING if any("not read yet" in w for w in why) else REFUSED, "; ".join(why)
    if arm in ("d1", "a2", "toy_a1", "toy_a2"):
        g = read_g1(stage)
        if g is None:
            return PENDING, f"{arm} waits for G1 (AM-2)"
        if g != PASS:
            return REFUSED, f"G1 is {g}: {arm} is HELD for the orchestrator's re-plan (AM-2)"
    if arm in ("toy_a1", "toy_a2"):
        prod = arm[len("toy_"):]
        cls, why = final_class(stage, prod)
        if cls is None:           # not read yet, or look 1 said CONTINUE: look 2 may still give a final class
            return PENDING, f"AM-3: {why}"
        if cls not in FINAL_CLASSES:
            return REFUSED, f"AM-3: production {prod}'s final class is {cls}: the calibration is NOT SHOWN, no toy fork"
    if look2 and arm not in CONTROLS:
        return _look2_licence(stage, arm)
    return ALLOWED, f"{arm} s{seed}: no unmet condition (stage {stage_of(arm, seed)})"


def _batch2_status(stage: str, arm: str, seed: int) -> Tuple[str, str]:
    """A batch-2 fork's registered conditions (STAGED_PLAN §4.1.1, §4.1.2, §4.2): production is PENDING until its
    FP32_ALT commit is named (AM-8); production LR-F is REFUSED once toy LR-F's FINAL class is LOSS (§4.1's licence
    table, registered in §4.1.2; an unread toy LR-F is no condition, AM-RP-e puts production first); a look-2 seed
    needs its own look 1 to read CONTINUE."""
    from main.x35.kit import ARMS

    side = ARMS[arm].side
    if side == "production":
        filled, why = side_gate(stage, side)
        if not filled:
            return PENDING, why
    if arm == "lrfloor":
        cls, why = final_class(stage, "toy_lrfloor")
        if cls == "LOSS":
            return REFUSED, (f"STAGED_PLAN §4.1.2: toy LR-F's FINAL class is LOSS ({why}): the controller's cuts were "
                             "protective, production LR-F is not run on this evidence")
    if seed in LOOK2_SEEDS[side]:
        return _look2_licence(stage, arm)
    return ALLOWED, f"{arm} s{seed}: no unmet condition (stage {stage_of(arm, seed)})"


# --------------------------------------------------------------------------------- AM-6: K9(b) at each arm's end
@dataclass(frozen=True)
class K9b:
    run: str
    verdict: str
    fork_step: Optional[int]
    n_rows: Dict[str, int]
    max_value: Dict[str, Optional[float]]
    fires: Tuple[str, ...]
    near: Tuple[str, ...]
    reasons: Tuple[str, ...]
    #: the arm overlapped (X14): the snapshot family is the REQUIRED one; ``overlap_evidence`` says why we believe it
    overlap: bool = False
    overlap_evidence: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"schema": K9B_SCHEMA, "run": self.run, "verdict": self.verdict, "fork_step": self.fork_step,
                "rows_after_fork": self.n_rows, "max_after_fork": self.max_value, "ceilings": K9B_CEILINGS,
                "fires": list(self.fires), "near_ceiling": list(self.near), "reasons": list(self.reasons),
                "overlap": self.overlap, "required_family": "behaviour_snapshot" if self.overlap else "behaviour",
                "overlap_evidence": list(self.overlap_evidence)}


def judge_k9b(run: str, fork: Optional[int], series: Mapping[str, Sequence[Tuple[int, float]]],
              log_text: str = "", *, overlap: bool = False, overlap_evidence: Sequence[str] = ()) -> K9b:
    """The deterministic AM-6 verdict on the fork's own rows (``series`` = ``{key: [(step, value)]}`` for
    :data:`K9B_TAGS`' keys, plus :data:`K9B_SNAPSHOT_TAGS`' on an overlap arm, already restricted to steps > ``fork``).

    Every row of every family present is judged against the ceilings (a fire in either VOIDs). The REQUIRED family is
    ``behaviour`` on a serial arm and ``behaviour_snapshot`` when ``overlap`` (or when the series carries a snapshot row:
    that row is itself the evidence); only the required family's absence is INCONCLUSIVE. The other family's absence
    is "nothing judged" (a probe with no current row writes no tag, ``gen3_k9b_no_row_no_tag_v1``), never a reason
    and never a VOID; a NON-FINITE row is a fire in either. Rows are COUNTED, never deduplicated by step: the last
    update's rows are logged at the same step as the one before (``loop._final_dump``)."""
    fires: List[str] = []
    near: List[str] = []
    reasons: List[str] = []
    n_rows: Dict[str, int] = {}
    mx: Dict[str, Optional[float]] = {}
    overlap = bool(overlap) or any(series.get(k) for k in K9B_SNAPSHOT_TAGS)
    tags = dict(K9B_TAGS, **(K9B_SNAPSHOT_TAGS if overlap else {}))
    required = tuple(K9B_SNAPSHOT_TAGS) if overlap else tuple(K9B_TAGS)
    if fork is None:
        reasons.append("the fork step is unknown: the inherited rows cannot be excluded")
    for key, tag in tags.items():
        bar = K9B_CEILINGS[K9B_CEILING_KEY[key]]
        rows = [(s, v) for s, v in series.get(key) or () if fork is None or s > fork]
        n_rows[key] = len(rows)
        if not rows:
            mx[key] = None
            if key in required:
                reasons.append(f"{tag}: no row after the fork step (absence is not a pass)")
            continue
        finite = [v for _, v in rows if math.isfinite(v)]
        mx[key] = max(finite) if finite else None
        for s, v in rows:
            if not math.isfinite(v):
                fires.append(f"{key} non-finite {v!r} at step {s}")
            elif abs(v - bar) <= RULE8_EPS * bar:
                near.append(f"{key} {v!r} at step {s} within 1e-9 relative of {bar:g} (recorded, no fire)")
            elif v >= bar:
                fires.append(f"{key} {v!r} >= {bar:g} at step {s}")
    if K9B_NONFINITE_LINE in log_text:
        fires.append(f"the child log carries '{K9B_NONFINITE_LINE}'")
    verdict = VOID if fires else INCONCLUSIVE if reasons else PASS
    return K9b(run, verdict, fork, n_rows, mx, tuple(fires), tuple(near), tuple(reasons), overlap,
               tuple(overlap_evidence))


def recorded_overlap(run_dir: str) -> List[str]:
    """What the run's own ``metadata.json`` records about ``--overlap-collect`` (X14): ``env_core.overlap_collect`` (the
    env core the last process ran; written ON only) and ``cli_args.overlap_collect`` (the recipe row). Empty = the record
    says OFF, is absent (a run that predates the flag) or is unreadable; the TensorBoard rows are the second witness."""
    try:
        meta = json.loads(Path(run_dir, "metadata.json").read_text())
    except (OSError, ValueError):
        return []
    out: List[str] = []
    for block, name in (("env_core", "metadata.json env_core.overlap_collect"),
                        ("cli_args", "metadata.json cli_args.overlap_collect")):
        b = meta.get(block)
        if isinstance(b, dict) and b.get("overlap_collect") is True:
            out.append(name)
    return out


def k9b_read(run_dir: str) -> K9b:
    """AM-6 on a real run directory (read-only): its own TensorBoard rows after the fork step, and its child log.
    An arm that OVERLAPPED (:func:`recorded_overlap`, or an own ``overlap/*`` or ``behaviour_snapshot/*`` row) is judged
    with ``behaviour_snapshot/*`` as its required family (:func:`judge_k9b`)."""
    from main.x35.tbrows import fork_step, load_segments

    fork = fork_step(run_dir)
    tag_of = dict(K9B_TAGS, **K9B_SNAPSHOT_TAGS)
    want = list(tag_of.values()) + [K9B_OVERLAP_TAG]
    segs = load_segments(run_dir, want, after_step=fork) if fork is not None else []
    series: Dict[str, List[Tuple[int, float]]] = {k: [] for k in tag_of}
    n_overlap_rows = 0
    for seg in segs:
        for key, tag in tag_of.items():
            series[key] += seg.rows.get(tag, [])
        n_overlap_rows += len(seg.rows.get(K9B_OVERLAP_TAG, []))
    evidence = recorded_overlap(run_dir)
    if n_overlap_rows:
        evidence.append(f"{n_overlap_rows} own {K9B_OVERLAP_TAG} row(s)")
    n_snap = max(len(series[k]) for k in K9B_SNAPSHOT_TAGS)
    if n_snap:
        evidence.append(f"{n_snap} own behaviour_snapshot/* row(s)")
    log = os.path.join(run_dir, "launcher_child.full.log")
    text = Path(log).read_text(errors="replace") if os.path.isfile(log) else ""
    return judge_k9b(os.path.basename(os.path.normpath(run_dir)), fork, series, text, overlap=bool(evidence),
                     overlap_evidence=evidence)


#: Where a K9(b) record is written when the caller names no directory: OUTSIDE every checkout.
K9B_SCRATCH_ENV = "GEN3AI_X35_K9B_SCRATCH"


def k9b_scratch_dir() -> str:
    """The default output directory for a K9(b) record: ``$GEN3AI_X35_K9B_SCRATCH``, else
    ``~/.cache/gen3ai/x35_k9b`` (a record lands in the tree only by a shipped commit, never by a read)."""
    return os.path.abspath(os.environ.get(K9B_SCRATCH_ENV) or os.path.expanduser("~/.cache/gen3ai/x35_k9b"))


def _main_checkout_roots() -> List[str]:
    """The real path of the MAIN checkout (git's common dir parent from this checkout), plus this checkout when it
    is not a linked worktree. Fails closed: a linked worktree whose main git cannot name raises."""
    from utils.git import get_main_repo_root
    from utils.paths import is_linked_worktree, repo_root

    here = str(repo_root())
    roots: List[str] = [] if is_linked_worktree(Path(here)) else [os.path.realpath(here)]
    try:
        roots.append(os.path.realpath(get_main_repo_root(cwd=here)))
    except Exception as exc:
        if not roots:
            raise BatchError(f"cannot name the main checkout from {here} ({exc}): a K9(b) record is not written "
                             "where main cannot be ruled out") from exc
    return roots


def refuse_inside_main_checkout(path: str, *, roots: Optional[Sequence[str]] = None) -> None:
    """A K9(b) record is never written inside the MAIN checkout (main is never dirty; CB-11 / F1): a write there
    is a refusal. A linked worktree is fine (that is how a record lands: written there, then shipped)."""
    real = os.path.realpath(path)
    for root in (roots if roots is not None else _main_checkout_roots()):
        if real == root or real.startswith(root.rstrip(os.sep) + os.sep):
            raise BatchError(f"refusing to write a K9(b) record into the main checkout ({real} is inside {root}): "
                             "pass --out-dir (default a scratch dir outside every checkout) or a linked worktree's dir")


def k9b_record(stage: str, k: K9b) -> str:
    """Write (or overwrite: a re-read of a finished arm is the same read) ``k9b/<run>.json`` under ``stage``, which
    must not be inside the main checkout."""
    refuse_inside_main_checkout(stage)
    d = os.path.join(stage, K9B_DIR)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{k.run}.json")
    Path(path).write_text(json.dumps(dict(k.as_dict(), read_at=_now()), indent=1, sort_keys=True) + "\n")
    return path


def k9b_batch(stage: str, models_dir: str, *, out_dir: Optional[str] = None, reader: Any = None) -> Tuple[bool, List[str]]:
    """The launch pre-check: every FINISHED side-pinned run (batch 1 + 2, ``final_model.zip``) has a K9(b) arm-end
    record that PASSes. ``stage`` is only READ (a record already shipped there); a missing record is read now
    (``reader``, default :func:`k9b_read`) and written to ``out_dir`` (default :func:`k9b_scratch_dir`; refused inside
    the main checkout), which is also read for a record a previous check left. VOID / INCONCLUSIVE ⇒ HOLD."""
    read = reader or k9b_read
    out_dir = os.path.abspath(out_dir) if out_dir else k9b_scratch_dir()
    refuse_inside_main_checkout(out_dir)
    ok, lines = True, []
    ours, _ = _side_pinned_dirs(models_dir)
    for d, _base in ours:
        rd = os.path.join(models_dir, d)
        if d != _base or not os.path.isfile(os.path.join(rd, "final_model.zip")):
            continue
        have = [p for p in (os.path.join(root, K9B_DIR, f"{d}.json") for root in (stage, out_dir)) if os.path.exists(p)]
        if have:
            v = json.loads(Path(have[0]).read_text()).get("verdict")
        else:
            k = read(rd)
            k9b_record(out_dir, k)
            v = k.verdict
        if v != PASS:
            ok = False
        lines.append(f"{d}: K9(b) arm end {v}")
    return ok, lines or ["no finished batch-1/2 arm yet"]
