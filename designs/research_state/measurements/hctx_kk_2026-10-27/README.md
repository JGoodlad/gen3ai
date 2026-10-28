# H-CTX A-K: under a neutral pilot rc72b is slightly weaker, and coherence does not explain it (2026-10-27 / 28)

Registered in [`PREREG.md`](PREREG.md) (`fb84c260f`, before any game). The registration is
[`tsr_registration.json`](tsr_registration.json) (digest `d30f3d79…`). Record: [`record.json`](record.json), from
[`scripts/record.py`](scripts/record.py). Post-hoc sensitivity: [`sensitivity_alias.json`](sensitivity_alias.json).
Parent: H-CTX [`../hctx_rc72_2026-10-26/`](../hctx_rc72_2026-10-26/README.md) (SUPPORTED); this read answers its
review's M3 (A-K).

Abbreviations: pp = percentage points; Elo = 400·log10 odds; P\* = `rb_x26_s2601` @280,000,170; Kakuna = Metamon's
Kakuna, checkpoint 34; F = `forum72`; COHERENT = home72 ∪ F; S = Kakuna-on-rc72b's score against Kakuna-on-COHERENT;
A-K = Kakuna's own response to a panel.

## Verdict

**`hctx_kk_2026-10-27: rc72b UNDECIDED under Kakuna (S = 0.459 [0.428, 0.493])`**, i.e. −28 Elo [−51, −5].

- **DETECTED WEAKER**: the upper bound is below 0.50.
- **Not shown to be MATERIALLY weaker**: the lower bound is below the 0.450 (−35 Elo) bar, but so is not the upper
  bound, so the registered class is UNDECIDED.
- **The author's registered guess was S ≈ 0.47, UNDECIDED.**

| quantity | value |
|---|---|
| **S, rc72b against COHERENT** (54 units, 1,080 games) | **0.459 [0.428, 0.493]** |
| against home72's teams (544 games) | 0.458 [0.410, 0.508] |
| against F's teams (536 games) | 0.461 [0.415, 0.508] |
| home72 − F, as opponents | +0.005 [−0.066, +0.077] |
| S, teams resampled too (D2) | [0.400, 0.519] |
| post-hoc, all 60 units (FINDING 1) | 0.458 [0.429, 0.488], UNDECIDED, detected weaker |

All intervals are unit bootstraps, B = 10,000, seed 7274 (+1, +2, …).

## Reading

**Against the three accounts** (REVIEW M1: (a) memorised sixes, (b) P\*'s core knowledge, (c) a recombined six has no
plan):

- **(c)'s premise is neither confirmed nor excluded at the set level.** In Kakuna's hands rc72b is about 4 pp
  (28 Elo) weaker than human-built and pool sixes. That is detected, but under the 35-Elo materiality bar at the
  point estimate.
- **(c)'s mechanism is not seen inside the panel (D4).** S does not rise with coherence:
  - Smogon-lift terciles: 0.459 / 0.458 / 0.461;
  - pool species-pair PMI terciles: 0.473 / 0.439 / 0.465;
  - per-team correlations: +0.08 and +0.07.

  rc72b's most coherent third (lift +0.06 to +0.47, F's mean is +0.14) is as weak as its least coherent third. The
  panel is uniformly a little weaker for a reason the coherence metrics do not capture. Candidates the review lists
  as descriptive: the species mix (C5) and three times the sleep moves (MINOR 4).
- **A-K is small and points one way.** Kakuna does a little worse on rc72b than on coherent teams: −4 pp
  [−7, −1] in this asymmetric cell.
  - H-CTX's cells were symmetric (both seats pilot rc72b), so a panel-wide strength gap cancels to first order there.
  - If the gap is instead Kakuna PILOTING rc72b worse, it pushes P\*'s rc72b score UP. That direction is conservative
    for H-CTX's SUPPORTED.
  - Either way, the 10.1 pp P\* drop is not a Kakuna artefact of this size.
- **(a) against (b) is not separated, as registered.** The bridge row (D5) reads direction only:
  - P\*'s banked per-team edge on rc72b does not move with species-pair PMI or lift once each team's Kakuna strength
    is held fixed (partial r −0.001 and −0.016);
  - it does not fall on rc72b's weakest teams (P\*'s edge by Kakuna-strength tercile: 0.511 / 0.484 / 0.538);
  - (b) predicted a PMI gradient and (c) a fall on the weakest teams. Neither is seen; (a) predicted nothing.
  - The per-team edge's reliability is about 0.33 (REVIEW C4), so this leans toward (a) without measuring it.
  - The separator stays the rc72c core-preserving panel, whose predictions are registered in the PREREG §2.
- **H-CTX's verdict and its narrowed reading stand.** The sets keep at most about a fifth of the premium. What the six
  carries is still open, but "no plan" now has to explain an effect the coherence metrics do not track.
- **For X62** (recombined-team training, ≤ 2 rule): its corpus's sixes are about 28 Elo weaker as teams in a neutral
  pilot's hands. That is small next to procedural P72's −128 Elo (TSR), and REVIEW M1's rule still holds: only F reads
  an X62 gain.

## POST HOC: the strength-adjusted recovery (not registered; it crosses engines)

The question: if rc72b's 4 pp weakness under a neutral pilot went straight into P\*'s rc72b score, how much of the
home premium would rc72b recover?

- **The arithmetic** (H-CTX's R = rc72b − F, P = home72 − F, and δ = 0.5 − S from this read):
  - (R + δ) / P = (−2.6 + 4.2) / 7.5 = **0.21**.
  - With independent unit bootstraps of the two reads (B = 10,000, seed 7279): **[−0.88, +0.88]**.
  - The linear form R + δ − 0.25 P: −0.3 pp [−5.4, +4.9].
- **What that means:** under full transfer, rc72b would recover about a fifth of the premium. The interval spans both
  of H-CTX's bands, so the adjusted reading is UNDECIDED, where the registered one is SUPPORTED.
- **Why it is only an illustration.** The two reads cannot be combined into a registered number:
  - **Engines:** H-CTX played on `bdc5d2be…/95b45f10…` with Kakuna's legacy pack; this read on
    `e00a212a…/f2918a06…` with the new pack.
  - **The assumption:** that the weakness transfers in full. H-CTX's cells were symmetric (both seats pilot rc72b),
    so a panel-wide strength gap cancels to first order and the adjustment is then 0: the share stays −0.34
    [−2.05, +0.20]. Full transfer needs the weakness to compress P\*'s skill edge one-for-one, and nothing measured
    says it does.
- **Taken together:**
  - The premium is still mostly team-level under either assumption: point estimates −0.34 and +0.21, both below 0.25.
  - "No plan" (team weakness) can account for at most the 4 pp part of P\*'s rc72b − F gap, and only under full
    transfer.
  - The registered SUPPORTED holds only under the first-order cancellation H-CTX's symmetric design assumes.

## The run

- **Pin and engine.** Pin `3d11545b5`: the first main commit holding both the registration and the F1 fix
  (`our_team` from the peer's draw log). One engine `port=e00a212a102f5ce2/94;env=f2918a060efd2723/128`, peer code
  `9cc9caacde85f8ea`, new peer pack (`peer_pack_changed: true`), `ws_frontend@3d11545b5`. All 54 counted units carry
  that one stamp.
- **Lane.** `python -m main.team_strength run --lane kakuna`. Concurrency 1, on rule 5b's cores, on the heavy slots'
  short lane, with claimed 95xx ports. It ran 2026-10-27 15:49 → 2026-10-28 02:25 PT, alongside N1-P look 1.
- **Attempts.** 60 of 60 units ran on their first attempt (no short-lane overrun, no failure). Re-derived guards:
  60 / 60 clean (no forced end, no peer fault).
- **Counted: 54 of 60**, exactly the registered floor. The six refused units all drew `rc72b_05` on side A
  (FINDING 1).
- **Dispersion and games.** Unit SD 0.122 against 0.111 binomial. 2 ties. Mean game 58 turns (median 46).
- **Seat halves** (descriptive; from `games.jsonl`, where Metamon books a tie as a loss): 0.448 when side A
  challenges, 0.469 when it accepts.
- **Run-order halves:** 0.444 then 0.472.
- **Team attribution** comes from the captures' START lines, never from Metamon's CSV (REVIEW F1). Side B drew home72
  and F about equally (544 / 536).

## FINDINGS

1. **Team identity does not canonicalise Showdown move aliases.** `rc72b_05.txt` writes Gengar's move as `- wisp`; the
   server packs it as Will-O-Wisp; `main.team_strength.panel.team_id` keys the raw id, so the capture matched no
   rc72b file and every unit that drew it was refused (12 games of 120).
   - The games were valid: H-CTX played the same file with its pack-identity check passing.
   - The verdict uses the registered rule (54 units). The alias-mapped sensitivity agrees (60 units, S = 0.458).
   - The same alias sits in 1 pool file under `data/teams`. Any identity keyed on the raw move id (the meter, and
     `team_overlap`'s strict key) can miss it. It needs an owner: a canonicaliser through Showdown's aliases, with a
     test.
2. **D6's registered "seat halves" row is not in `record.py`.** It was computed afterwards from `games.jsonl` (above),
   with the CSV's tie-as-loss booking.
3. **`main.reads` has no Kakuna-vs-Kakuna group shape.** This read used the team-strength meter's lane, so `main.reads`'
   `plan` / `record` / VOID gate did not run. The same checks were done in the lane (unit counting, re-derived guards
   in count mode) and in `record.py` (one stamp, the floor).
4. **The superseded play tree** at `fb84c260f` was prepared, never played, and removed. The Metamon team sets
   `gen3ai_hctxk_ref143` and `gen3ai_hctxk_rc72b` stay installed in Metamon's cache, as the TSR's sets did.

## Files

- `PREREG.md`, `tsr_spec.json`, `tsr_registration.json`;
- `scripts/record.py` and `record.json` (the registered record);
- `scripts/sensitivity_alias.py` and `sensitivity_alias.json` (post hoc);
- **state:** `~/gen3ai_archive/hctx_kk_read/` (every unit's captures, peer logs, `games.jsonl`, `summary.json`; the
  lane journal `kakuna/units.jsonl`; `play.json`).
