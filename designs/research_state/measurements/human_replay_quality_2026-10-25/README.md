# The human-replay DATA CARD of `hr_spec_v1` (2026-10-25; REVISED the same day after reviews A / B / C)

**Revision.** Three adversarial reviews sit beside this file; [`RESPONSE.md`](RESPONSE.md) maps every finding to its
disposition. What changed here: the extraction reads each snapshot's MANIFEST and re-hashes every log; `results.json`
is stamped with the snapshot ids; account names are salted hashes; the drift floor is account-clustered; §E adds the
account-disjoint re-read and a full-six contrast with its interval; the familiarity reading is "uninformative".

**What this is.** The data-quality analysis the owner's 2026-10-25 grant requires before human replays may train the
policy ([`../../../endstate/human_data_architecture.md`](../../../endstate/human_data_architecture.md) §5;
the plan: [`../../../endstate/design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md)). CPU only;
read-only on the corpus, `data/teams`, the registered panels (never the SEALED forum216b) and the X50 rung-1b archive.
No model was run: the HOM / P\* numbers re-cut R-b1's cached per-row forwards.

**Snapshots** (frozen here; registry `designs/research_state/human_replay_snapshots.json`): `hr_spec_v1` = the
spectator manifest's 376,410 logs, 3.49 GB, manifest sha256 `e7d491e7…f3b1`; `hr_arch_2026-10-24` = the 16,490 archive
logs its manifest lists (dates ≤ 2026-10-24), sha256 `ad940108…f7a2`. `results.json`'s `snapshots` block records each
id, its manifest digest, the extraction commit and `hrq.py`'s own hash. `hr_spec_v1` has a durable, re-verified copy at
`~/gen3ai_archive/human_snapshots/hr_spec_v1/logs.tar.zst` (297 MB, sha256 `315a9d1b…f09c`). Every number below is
`hr_spec_v1` unless it says archive.

**Registration.** The strata, the bot rule (a name pattern; ≥ 100 spectator games on one Los Angeles date; games
started in ≥ 20 distinct clock hours of one date), the machine-speed bar (< 2 s a turn over ≥ 10 turns), the
walk-over bar (a forfeit / timer loss by turn 3) and the train / held-out split were fixed in `hrq.py` before the full
analysis ran, by this card's own account: the code arrived in one commit with its results, so history cannot confirm it
(REVIEW A m7). A 3-day development run (07-01, 09-21, 10-01) exercised the code first; no threshold changed after it.
Everything here is DESCRIPTIVE: no verdict rule was registered for the numbers, and the per-stratum verdicts at the
end are judgements from them, named as such.

## A. Volume, format, parse, duplicates, bots, endings

- **Files:** 376,410 spectator logs (2026-05-18 → 09-23, 107 full days ≈ 3,497 games / day; holes 06-16..20 and
  08-03..17) plus 16,490 archive logs (06-16 → 10-24). **0 text-pass errors.** Format: **100 % `[Gen 3] OU` singles,
  one rule set** (the same 13 clauses in every log). Every log has a result line.
- **Parse (the production reader, earlier reads on the same files):** 0 refusals in 752,820 seat reads (X50 census);
  C-GAME completed **594,266 of 597,920 seats (99.39 %)** of the train dates; refused 0.014 % (rung-1b build).
- **Duplicates:** no id in two files; **4** spectator logs whose protocol body is byte-identical to another's.
- **Accounts:** 23,364; the top account holds 0.55 % of sides, the top 10 3.1 %, the top 100 14.9 % (median 4 games
  per account, p99 486). The spectator stream is not dominated by a few accounts (the archive is: 10 % for one account).
- **Suspected bots, by the rule: 61 accounts, ≈ 5–6 % of sides in every rated stratum** (1500–1599: 6.1 %). Accounts
  appear here only as salted ids (`hrq.acct`; the salt and the name map stay in `~/gen3ai_archive/human_snapshots/`).
  They include one account whose name declares it a bot (`acct_082084f2f100`, 883 games in one day), a fleet of
  generated-looking names active in all 24 clock hours (`acct_64c8a8570c27` and five more), one name resembling a
  Metamon public agent (`acct_9f2ef7808db6`, 94 games) and, by the volume bar alone, heavy human grinders the rule cannot
  tell from scripts (33 of the 61 are flagged by volume only). The name pattern's `^bot` also caught three human-looking
  names (30 games together; immaterial). **Machine-speed games: ≤ 0.02 %**, but that meter can only see bot-vs-bot
  games (a bot waits for its human opponent), so it is no evidence about bots (REVIEW A m7). No account of ours plays
  the ladder. A name / volume rule cannot see a Kakuna-family fleet at ordinary volume: the plan's §3.4 audit.

| stratum (the side's own pre-battle rating) | sides | share | turns p10 / p50 / p90 | forfeit | timer | KO | walk-over ≤ 3 turns | this side quit | suspected bot | full team seen | moves seen / mon | item seen / mon |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 1300 | 410,177 | 54.5 % | 6 / 21 / 41 | 57.6 % | 7.2 % | 35.1 % | 5.1 % | 33.0 % | 5.4 % | 49.8 % | 1.80 | 0.61 |
| 1300–1499 | 246,435 | 32.7 % | 11 / 25 / 45 | 52.2 % | 4.2 % | 43.3 % | 2.0 % | 28.0 % | 5.2 % | 63.3 % | 1.86 | 0.66 |
| **1500–1599** | 57,547 | 7.6 % | 13 / 26 / 48 | 49.9 % | 3.1 % | 46.7 % | 1.6 % | 24.9 % | 6.1 % | 67.9 % | 1.90 | 0.69 |
| **1600–1699** | 19,235 | 2.6 % | 14 / 28 / 51 | 49.2 % | 2.5 % | 47.9 % | 1.2 % | 22.4 % | 5.9 % | 70.7 % | 1.93 | 0.71 |
| **1700+** | 3,844 | 0.5 % | 15 / 29 / 56 | 50.8 % | 2.2 % | 46.7 % | 1.0 % | 20.5 % | 5.7 % | 69.0 % | 1.97 | 0.74 |
| unrated | 15,582 | 2.1 % | 8 / 25 / 47 | 28.1 % | 1.3 % | 70.3 % | 4.0 % | 14.7 % | 1.2 % | 63.7 % | 1.92 | 0.60 |

"Forfeit" counts every game that ended in a forfeit, at any turn: about half of all ladder games, at every rating.

## B. Labels and the fill (the X50 rung-1b extraction, train dates)

- **The chosen action is known for 15,180,478 of 16,808,248 completed decisions (90.3 %)**; 190,634 had one legal
  action (no information). The rest are hidden choices (sleep / paralysis / flinch / freeze `cant`), fainted-before-acting
  and the last decision. **They are missing NOT at random** (REVIEW A M3, a 6,000-log text pass): a `cant` or a faint
  before acting proves a move was chosen, and dropping them makes a sleeping mon's labelled switch rate 0.558 instead of
  0.260 (frozen 0.675 vs 0.288; all 0.250 vs 0.228). The plan keeps those rows with a set-valued label.
- **Informative rows (an own label, ≥ 2 legal actions) by the actor's stratum:** < 1300 7.70M; 1300–1499 5.14M;
  1500–1599 1.26M; 1600–1699 0.44M; 1700+ 0.09M (≥ 1500: **1.80M**, the HOM's 1.76M train + 38k validation; 19 % of
  battles were removed by the heldout72 team-memory rule first). Switch labels are 31–34 % of them.
- **How much of a row is invented (the fill).** At ≥ 1500 a mon shows ≈ 1.9 of its 4 moves by the end of the game
  (so ≈ 52 % of move slots are Smogon draws), its item 69–74 % of the time, its EVs / IVs / nature never; 68–71 % of
  sides show all six mons (the rest are drawn whole from the teammates prior); an acting mon with all four moves shown
  is 18 % of decisions (census tier A). **The chosen action itself is never drawn.** What the draws can corrupt is the
  CONTEXT: the alternatives' content, the stats, the bench. Measured cost where the truth is known (our own teams, R0):
  +0.012 to +0.019 nats on the intent head; on human teams it is **UNVERIFIED** (no full-information human source exists).
  Two more properties of the fill: a drawn move is never the label and a drawn bench mon is never a switch target (a
  structural signature, REVIEW B M6), and the Smogon prior (2025-05..2026-04) is 1–17 months older than the games, its
  species TVD to ≥ 1500 spectator usage growing 0.070 → 0.092 over the window (REVIEW A m4).

## C. Collection (Q-H1)

- **Spectator stream:** ≈ 3,497 games / day ≈ **17k informative ≥ 1500 rows / day** (1.80M over the 106 train-date
  spectator days). Its collector is deleted (poke-env retired); reviving one needs a build on the Rust live reader and
  the owner's word on joining public rooms (the proxy grant is "download only").
- **Archive pull (live, `gen3ai-replay-daily.timer`):** **365 uploaded games / day** (29 full days 09-25..10-23,
  160–714), 26 % of sides ≥ 1500 (spectator 11 %), so ≈ 4–6k ≥ 1500 rows / day, SKEWED. The proxy floor (1 request / s,
  ≈ 2 requests per replay) is not the bound: ≈ 43k replays / day would fit; the archive's upload volume is the bound.
  The archive's `search.json` reaches 2020-09, so a historical backfill (**ESTIMATE** ≈ 0.5M uploaded gen3ou games,
  ≈ 2 weeks at the floor) is the one large collection lever left; it is an older meta (drift) and the uploader's
  selection, and it is not authorised by this unit.
- **What volume buys:** the HOM's validation NLL on ≥ 1500 humans after 0.25 / 0.5 / 1 / 3 epochs of its 1.76M rows:
  1.260 / 1.228 / 1.197 / 1.156 (rung-1b build). That is a curve over epochs, not over distinct data; a data-scaling
  curve is unmeasured.

## D. Coverage: do human teams reach the unfamiliar region? (Q-H6)

On the 50,827 non-bot ≥ 1500 sides of the train dates with all six mons seen (4,810 distinct species sextets; all
rated sides: 41,238 distinct sextets, against our pool's 719 teams).

| yardstick | humans ≥ 1500 | heldout72_hpfix | forum72 | p72 | home72 |
|---|---:|---:|---:|---:|---:|
| species-usage TVD vs humans ≥ 1500 | — | 0.125 | 0.170 | 0.134 | 0.160 |
| species-usage TVD vs our pool | 0.161 | 0.170 | 0.184 | 0.212 | 0.104 |
| share sharing ≤ 3 species with their nearest pool team | 8.3 % | 4.2 % | 4.2 % | 13.9 % | 0 % |
| share with ≤ 2 mons fitting one pool team's sets (the panels' region) | **15.9 %** | 98.6 % | 77.8 % | 100 % | 0 % |
| share with ≥ 4 mons fitting one pool team's sets (pool-like) | 46.7 % | 0 % | 0 % | 0 % | 100 % |

A human side's fit counts are an UPPER bound (an unrevealed move fits any set), so 15.9 % is a LOWER bound on the
FULL-SIX human share in the panels' region (full-six sides are 68–71 % of sides, and longer games). Human mass on species absent from our pool: 1.7 %; on species in < 1 % of pool
teams or absent: 6.0 %.

**Per panel team: how many ≥ 1500 human sides fit ≥ 4 of its sets** (the team-memory rule; a species-only reveal fits,
so this over-counts):

| panel | median sides per team [p10, p90] | teams with ≥ 10 sides | teams with 0 |
|---|---|---:|---:|
| home72 (in-pool reference) | 218 [22, 822] | 97 % | 0 % |
| heldout72_hpfix | 105 [7, 473] | 85 % | 1.4 % |
| forum72 | 74 [3, 282] (CORE 263, NEAR 91, **FAR 11**, FAR zero 21 %) | 82 % | 4.2 % |
| p72 (procedural) | 6 [1, 62] | 43 % | 9.7 % |

So humans pilot cores close to most H and F teams hundreds of times; P72's procedural teams and forum72's FAR teams
are thinly covered. **The near-panel region must be EXCLUDED from any training use, at the panels' own radius:** a
side fitting ≥ 3 of one held-out team's sets sits in 84.4 % of ≥ 1500 TRAIN battles (70.7 % counting only mons with a
revealed move), against 36.4 % at the drafted ≥ 4 (REVIEW A B2, without `forum216b`). The plan registers ≥ 3
move-revealed fits on the REVEALED facts, before the fill (its §3.1).

## E. Learnability and the familiarity gradient (Q-H7; R-b1's cached forwards re-cut, post hoc)

Held-out dates 09-21..23, actor ≥ 1500, 69,171 rows; battle-clustered 95 % CIs (2,000 resamples, seed 50). The
acting side's familiarity is its overlap with the nearest pool team. 98.0 % of all rows' header rating matched the
joined seat; every mismatch is an unrated row (None vs 0) and none changes the ≥ 1500 stratum (REVIEW A m2: resolved).
The held-out dates have now been read by R1, R-b1 and this section: they are not "untouched".

| stratum (≥ 1500) | rows | battles | NLL P\* | NLL HOM | Δ (HOM − P\*) |
|---|---:|---:|---|---|---|
| **set fits ≤ 2 (the panels' region)** | 11,792 | 341 | **1.762 [1.706, 1.826]** | 1.144 | −0.619 [−0.662, −0.576] |
| set fits ≥ 4 (pool-like) | 31,978 | 872 | **1.738 [1.709, 1.767]** | 1.157 | −0.581 [−0.607, −0.554] |
| species overlap ≤ 3 | 4,300 | 217 | 1.708 [1.636, 1.776] | 1.140 | −0.568 [−0.620, −0.513] |
| species overlap 5–6 | 44,228 | 996 | 1.728 [1.701, 1.753] | 1.157 | −0.570 [−0.592, −0.547] |
| turn ≤ 5 / 6–20 / > 20 | 10,656 / 29,119 / 29,396 | | 1.765 / 1.766 / 1.682 | 1.187 / 1.172 / 1.110 | −0.58 / −0.59 / −0.57 |
| the game ends in a forfeit / timer | 26,899 | 684 | 1.796 | 1.185 | −0.611 [−0.641, −0.583] |
| the game is played out (KO) | 41,794 | 794 | 1.681 | 1.120 | −0.561 [−0.582, −0.540] |
| **acting account UNSEEN on the train dates** (revision) | 15,588 | 428 | 1.694 | 1.191 | **−0.503 [−0.542, −0.467]** |
| acting account seen on the train dates (revision) | 53,583 | 1,229 | 1.741 | 1.135 | −0.605 [−0.624, −0.586] |
| full-six side, set fits ≤ 2 (revision) | 8,319 | 174 | 1.803 | 1.160 | −0.643 [−0.700, −0.586] |
| full-six side, set fits ≥ 4 (revision) | 28,051 | 721 | 1.739 | 1.159 | −0.579 [−0.607, −0.552] |

Contrasts, one joint battle-cluster bootstrap each (revision): P\* NLL, full-six fits ≤ 2 minus fits ≥ 4,
**+0.064 [−0.016, +0.149]**; the original all-sides cut +0.024 [−0.043, +0.097]; HOM − P\* on unseen minus seen
accounts **+0.102 [+0.060, +0.143]**.

**Reading (revised).** Human choices are learnable in every cut, and **not only by account memory**: on accounts that
never played on the train dates the HOM still beats P\* by 0.503 nats; ≈ 17 % of the gain (0.102) is account memory.
The HOM's own validation split was by battle id, so its early stopping was in-account (REVIEW A M2). **The familiarity
proxy is UNINFORMATIVE** (REVIEW A M1): restricted to full-six sides, P\* is +0.064 [−0.016, +0.149] nats worse on the
panels' region, an interval that includes 0 and no bar; and the metric has no prediction under A-1 (P\* disagrees with
humans on pool-like teams too, for the opposite reason), so a flat or a sloped profile is expected either way. It
neither supports nor removes the case for A-1; the matched-student contest X59-S tests it.

## F. Drift (Q-H8)

Species-usage TVD of full-six rated sides by calendar month. The side-level split-half floor (two random halves of
July, 58,766 sides each) is 0.009, but it ignores REPEAT PLAYERS (REVIEW A m1): one account's sides share a team. The
revision's floor splits ACCOUNTS (below).

| month | full-six sides | mean side rating | ≥ 1500 share | TVD vs previous month | TVD vs pool / H / F / P72 |
|---|---:|---:|---:|---:|---|
| spec 2026-05 (from 05-18) | 56,849 | 1,258 | 8.7 % | — | 0.199 / 0.180 / 0.180 / 0.154 |
| spec 2026-06 | 107,082 | 1,266 | 10.0 % | 0.044 | 0.194 / 0.172 / 0.185 / 0.154 |
| spec 2026-07 | 117,533 | 1,287 | 12.3 % | 0.041 | 0.190 / 0.166 / 0.185 / 0.147 |
| spec 2026-08 (hole 03..17) | 59,033 | 1,273 | 11.0 % | 0.038 | 0.198 / 0.176 / 0.189 / 0.161 |
| spec 2026-09 (to 09-23) | 75,156 | 1,279 | 11.7 % | 0.049 | 0.199 / 0.167 / 0.200 / 0.164 |
| arch 2026-08 | 4,288 | 1,380 | 29.6 % | — (other population) | 0.166 / 0.144 / 0.180 / 0.152 |
| arch 2026-09 | 5,247 | 1,411 | 29.9 % | 0.232 | 0.251 / 0.241 / 0.256 / 0.291 |
| arch 2026-10 | 9,462 | 1,392 | 30.6 % | 0.099 | 0.212 / 0.190 / 0.209 / 0.237 |

**The account-clustered read (revision; 20 keyed account splits; `results.json` `drift.by_account`).**

| | 05 | 06 | 07 | 08 | 09 |
|---|---:|---:|---:|---:|---:|
| WITHIN: two account-disjoint halves of one month | 0.064 | 0.047 | 0.048 | 0.062 | 0.060 |
| CROSS: account-disjoint halves of consecutive months (m → m+1) | 0.064 | 0.058 | 0.060 | 0.071 | — |
| EXCESS = CROSS − mean WITHIN (share of splits > 0) | +0.009 (1.00) | +0.011 (1.00) | +0.005 (0.95) | +0.010 (1.00) | — |

**Reading (revised).** The month-to-month TVD of 0.038–0.049 is mostly which PLAYERS a sample holds: two
account-disjoint halves of one month already differ by 0.047–0.064. Beyond that floor, consecutive months differ by an
excess of 0.005–0.011, positive in 19–20 of 20 splits: slow drift is real but an order of magnitude smaller than first
stated. The distance to our pool and to each panel is flat across the five months (0.19–0.20 to the pool), and the
rating mix is stable (mean 1,258–1,287). The archive months swing 0.09–0.23: uploader concentration, not meta. The
drift alarm for a new snapshot (architecture doc §8): a new month's excess over the snapshot's last month, positive in
every split, with a mean above 0.016 (1.5 × the largest above), or a mean-rating move > 50 Elo within one population.

## G. The verdict per stratum (judgement from A–F; nothing registered decides it)

| stratum / slice | verdict | why |
|---|---|---|
| actor 1500–1599, 1600–1699, 1700+ | **USE-WITH-FILTER** | the policy levers' rows (1.80M); drop suspected bots, walk-overs, the 4 duplicates and every panel-near battle |
| actor 1300–1499 | **USE-WITH-FILTER, the human-only comparator only** | as learnable as ≥ 1500 (R-b1: −0.609), 2.9× the volume; weaker play, so never a policy lever's teacher in the first arm |
| actor < 1300 | **DROP** for every policy lever (HB0 may take it, walk-overs removed) | 58 % forfeits, 7 % timer losses, 5 % walk-overs, half the teams never fully seen |
| unrated | **DROP** | no rating to stratify; 70 % KO endings mark a different (challenge) population |
| suspected bots (61 accounts, ≈ 6 %) | **DROP** | not human; at least one fleet runs 24 h a day |
| walk-overs (forfeit / timer by turn 3) | **DROP** the battle | no decisions of interest; 1–5 % of sides |
| any battle with a side whose REVEALED facts fit ≥ 3 sets of one held-out panel team (mons with a revealed move) | **DROP** | evaluation hygiene at the panels' own radius (REVIEW A B1 / B2; the plan §3.1) |
| the archive population (`hr_arch_*`) | **DROP for training; USE for drift / collection reads** | SKEWED (fd7f4624); one account 10 % of a window |

## FINDINGS (standing rule 7)

1. **The familiarity proxy is uninformative** (§E, revised): neither support for nor evidence against H-BC's mechanism.
   Post hoc, descriptive; it does not test strength.
2. **The panel exclusion will cost the most useful data** (§D): at the panels' own radius (≥ 3 revealed fits) it
   removes ≈ 71 % of ≥ 1500 battles (without `forum216b`, sealed and not measured here).
3. **Resolved (REVIEW A m2):** the 2.0 % header-rating mismatches are all unrated rows; the join is right.
4. **The bot rule is a heuristic.** The volume bar cannot separate a multi-tab human grinder from a script; the
   per-player decision latency that would is not in a spectator log (`|t:|` stamps the turn, not the side). A hand
   audit (A-6) is the check.
5. **The spectator stream's own selection is unmeasured**: public rooms only, ≤ 20 at once, rooms > 1 h dropped, a
   reaper added 07-05; ≈ 50 % of ladder games (ESTIMATE, REVIEW A M7). The read that would measure it needs proxy pulls
   the owner has not approved.
6. **Process.** CPU only, rule 5b cores, every heavy step under `mem_cap.sh`; read-only on the corpus, `data/`,
   `models/` and the X50 archive. Cache, snapshots and logs: `~/gen3ai_archive/human_replay_quality/`,
   `~/gen3ai_archive/human_replay_snapshots/`.
7. **Privacy (REVIEW C F13):** committed artifacts carry salted account ids only, from this revision on. Older commits
   of this folder (`14ae4203`, `742decb2`) still contain the names; the owner decides on history. Two raw public replay
   fixtures under `src/main/x50_r1/testdata/` carry players' names as any replay does (no label attached).

## Reproduce

```
export PYTHONPATH=$PWD/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES=""
D=designs/research_state/measurements/human_replay_quality_2026-10-25 A=~/gen3ai_archive/human_replay_quality
C="nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh"
$C --max-s 600 6 $PY $D/hrq.py extract --snapshot hr_spec_v1 --cache $A/cache_v2           # ≈ 5 min; re-hashes every log
$C --max-s 300 4 $PY $D/hrq.py extract --snapshot hr_arch_2026-10-24 --cache $A/cache_v2
$C --max-s 600 16 $PY $D/hrq.py analyze --cache $A/cache_v2 --out $D/results               # ≈ 2 min; needs the salt
$C 4 $PY $D/hrq.py archive --snapshot hr_spec_v1                                           # the durable copy, verified
$PY -m pytest $D/hrq_test.py -q
nice -n 19 taskset -c 5-7,13-15 $PY -I $D/plan_power.py > $D/results/plan_power.json        # the revised plan's power
```

`results/results.json` holds every number above and names its snapshots; `results/plan_power.json` holds the plan's
power. A snapshot whose manifest, a log's hash or a log's header differs from the registry REFUSES extraction.

**Adversarial reviews of the plan built on this card:** [`REVIEW_A_data.md`](REVIEW_A_data.md) (data, bias and
leakage; its numbers are `review_a_checks.py`'s, `review_a_mnar.py`'s and `review_a_guard_sim.py`'s),
[`REVIEW_C_reasoning.md`](REVIEW_C_reasoning.md) (the architecture's reasoning) and
[`REVIEW_B_experiments.md`](REVIEW_B_experiments.md) (X59 / X60's design and causal inference; its power numbers are
`review_b_power.py`'s).
