# P\* against Metamon's HUMAN-ONLY models: the record (FR-1 and FR-3), 2026-10-25 / 26

The rules are in [`PREREG.md`](PREREG.md), committed at `57fd657b` before any counted game. Rows:
[`rows.jsonl.gz`](rows.jsonl.gz) (4,800 counted games, gzipped; `aggregate.py` writes it plain); per-cell stats and
every unit's counting verdict: [`cells.json`](cells.json) (both from [`scripts/aggregate.py`](scripts/aggregate.py));
raw units in `/home/goodlad/dev/gen3ai-reads/hr_reads_2026-10-25/` (not committed). Abbreviations as in the PREREG.

## The answer: P\* beats all three human-only models (OURS-BEATS ×3)

Greedy vs greedy, 1,600 games per opponent, **240 / 240 units counted**; every game twins X51's P\* vs Kakuna game of
the same panel, seed, half and index (400 / 400 twins matched in every cell). One regime stamp per cell.

| opponent (params; regime id) | U (H, F, P72 equal weight) [95 %] | verdict | home72 | Δ-K on U vs Kakuna's twins (X51 U 0.487) | Δ-K on home72 (X51 0.598) |
|---|---|---|---|---|---|
| `SmallRL` (14M, offline RL; `f0828cc22aa53b41`) | **0.666 [0.639, 0.692]** | **OURS-BEATS** | 0.785 [0.742, 0.822] | +0.179 [+0.140, +0.218] | +0.188 [+0.124, +0.249] |
| `LargeRL` (195M, offline RL; `1664a6b4a5c16de4`) | **0.631 [0.603, 0.658]** | **OURS-BEATS** | 0.700 [0.653, 0.743] | +0.144 [+0.105, +0.183] | +0.102 [+0.036, +0.167] |
| `LargeIL` (194M, imitation; `541710e5b509e8b7`) | **0.790 [0.766, 0.812]** | **OURS-BEATS** | 0.785 [0.742, 0.822] | +0.303 [+0.266, +0.339] | +0.188 [+0.124, +0.249] |

Every Δ-K class is **X WEAKER THAN KAKUNA**, on every panel. Per panel (P\*'s win rate, Wilson 95 %, all BETTER than 0.50):

| opponent | H | F | P72 | paired twins on U panels: Δ vs Kakuna (seed bootstrap) H / F / P72 |
|---|---|---|---|---|
| SmallRL | 0.678 [0.630, 0.721] | 0.663 [0.615, 0.707] | 0.658 [0.610, 0.702] | +0.180 [+0.110, +0.248] / +0.133 [+0.078, +0.188] / +0.225 [+0.155, +0.298] |
| LargeRL | 0.640 [0.592, 0.686] | 0.613 [0.564, 0.659] | 0.640 [0.592, 0.686] | +0.143 [+0.093, +0.193] / +0.083 [+0.035, +0.133] / +0.208 [+0.138, +0.280] |
| LargeIL | 0.810 [0.769, 0.845] | 0.770 [0.726, 0.809] | 0.790 [0.747, 0.827] | +0.313 [+0.260, +0.365] / +0.240 [+0.180, +0.298] / +0.358 [+0.288, +0.430] |

**RL vs IL on the same human data (FR-3's descriptive contrast):** P\* wins 0.159 [0.123, 0.195] more on U against
LargeIL than against LargeRL (home72 +0.085 [+0.024, +0.145]). The offline-RL recipe is clearly the stronger one;
LargeRL is the strongest of the three, ≈ 14 pp easier than Kakuna on U.

**What this answers.** "Does a model using only human data outperform ours?" **No**, at every human-only scale and
recipe Metamon publishes (14M RL, 200M RL, 200M IL), in our greedy regime, on familiar and unfamiliar teams alike.
**What it cannot say** (PREREG §6): these are upstream's recipes, which differ from P\* in architecture and scale as
well as data. It does not show that human data loses to self-play data inside OUR network; X59-S (the matched-student
contest) is that read. H is replay-derived and may be familiar to these models; their margin there is no smaller than
on F. H18 (Metamon misreads a Baton Pass) favours P\*.

**Against the predictions** (GUESSES, read by no rule): SmallRL U 0.72 → 0.666; LargeRL U 0.64 → 0.631 (close);
LargeIL U 0.68 → **0.790** (much weaker than guessed). The guessed panel order (F ≥ H > P72) held for none: the panels
are level within ≈ 4 pp for every opponent, so **P\*'s P72 deficit against Kakuna (0.433) is something Kakuna
exploits, not a general weakness of P\* on P72**.

Ties 8 and forfeit-limit games 31 of 4,800; median 29–36 turns. LargeRL's F cell is the one asymmetric one (0.570 when
we challenge vs 0.655 when the peer does), inside noise for 200-game halves.

## How the read ran

- **FR-1 first, then FR-3** (the orchestrator, 2026-10-25: the CPU is the bottleneck). FR-3 was HELD (`hold_fr3` in the
  run dir) after 20 LargeRL / 18 LargeIL units and resumed after FR-1's record (`4d8d9eef`) shipped.
- Two lanes, each unit a `mem_cap.sh --max-s 600` short-lane job on a flock-claimed 95xx port, `nice 19`, cores
  `5-7,13-15`, CPU only, the play tree `d32ffcee`, `HF_HUB_OFFLINE=1`. Wall ≈ 21 h for ≈ 17 lane-h of play: the heavy
  slots were shared with a live training run and up to 9 queued jobs (≈ 6–27 units / h). The orchestrator declined a
  BLOCKING mark (no GPU job waits on this read).

## FINDINGS

1. **Metamon's peer crashes after its last battle on some (seed, half) units.** Upstream's poke-env raises "Battle is
   already finished, call reset"; Metamon's `metamon_to_amago.py:323` handler retries by RECURSION until Python's
   recursion limit (985 "Force resetting due to long-tail error" lines, then `RecursionError`). The peer's own report
   says "the games stand, the process does not"; the counting rule drops the unit (`peer_clean = False`). It hit 3
   units: `sr_hpfix_10` (clean on its re-run), `lrl_p72_08` (clean on its re-run) and **`lrl_home72_02`, which failed
   TWICE at the identical log line** (deterministic for that seed and half on two tries) and was clean on its third,
   the last the PREREG allows. An upstream bug, left unpatched (patching the peer changes the anchor). The first tries
   are kept as `cells/<unit>.unclean*`. A harness-side fix (count a unit whose peer died AFTER its last game, with the
   games' records intact) is a separate decision.
2. The queue (`scripts/hrrq.py`) was restarted mid-read to add the FR-3 hold: both supervisors stopped by PID while
   their unit children ran on; the restarted lane skips any unit whose child is alive. No unit was played twice.
3. The PREREG's registration findings (the play tree `d32ffcee`; `HF_HUB_OFFLINE`, `main.anchors` not pinning a hub
   revision) stand.
