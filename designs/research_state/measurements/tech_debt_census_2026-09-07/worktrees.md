# The worktree census — 2026-09-07

> **READ-ONLY.** Nothing here was removed, unlocked, pruned or signalled. Every command below
> is PRINTED for the owner to run; none of them was executed. This report answers the
> `TECH_DEBT_BACKLOG.md` §2 row *"129 git worktrees, 13 detached"* — which is now **77**
> (76 plus this census's own, listed as `~wt/census` and removed by `scripts/land.sh`).

## How every number here was produced

```bash
cd /home/goodlad/dev/gen3ai
git worktree list --porcelain                        # 77 entries
git branch --merged main                             # merged set
git -C <path> status --short                         # dirty, per worktree
git rev-list --count main..<HEAD>                    # commits not on main
git merge-base --is-ancestor <pin-sha> main          # is the pin commit safe from gc
ps -eo pid,etimes,args | grep -E 'train_rl_agent|main.launcher'
readlink /proc/<pid>/cwd                             # for every pid in /proc
du -sm <path>                                        # size
# models/ side: every models/*/metadata.json (git_hash, pin_history, lineage,
# original_command, launcher_command, cli_args) and every models/*/model_config.json,
# read read-only from the MAIN checkout: 231 run directories, 216 with metadata.json.
```

## Summary

| class | n | disk | what it means |
|---|---:|---:|---|
| **MAIN** | 1 | — | `/home/goodlad/dev/gen3ai` itself. Clean. |
| **LIVE** | 4 | 1.2 GB | a running process's cwd or argv points into it |
| **RESUMABLE** | 18 | 16.5 GB | detached at (or branched on) a commit some run's `metadata.json` records as its pin |
| **DIRTY-UNMERGED** | 37 | 5.1 GB | commits not on main, or working-tree content that exists in no git object |
| **AGENT-LEFTOVER** | 17 | 3.9 GB | fully merged, nothing unique on disk, no reference, no process |
| **UNKNOWN** | 0 | — | every row resolved |
| **total** | 77 | 26.7 GB | |

Precedence when a row could be two things: **LIVE > RESUMABLE > DIRTY-UNMERGED >
AGENT-LEFTOVER**. A worktree is never called a leftover while anything still points at it.

## The finding that decides the whole row: **a pin is a COMMIT, not a DIRECTORY**

The backlog row suspects that the old `gen*-run-*` trees may be unreferenced now that a run's
pin is a `/tmp/launcher-<sha>-*` tree. Verified, and it is stronger than that:

1. `worktree.resolve_pin` pins to a **commit**, and `_create_run_worktree` makes a **fresh**
   `/tmp/launcher-<sha>-*` checkout for it. No resume ever re-enters `.claude/worktrees/gen3-run-0807`.
2. **Every pin commit in this census is an ancestor of `main`** — checked one by one with
   `git merge-base --is-ancestor <sha> main`, 16/16 true. So `git worktree remove` cannot
   orphan a pin: the commit is reachable from `main` and is not gc-eligible.
3. No `models/*/metadata.json` names a worktree PATH except nine `v8rep_*_0905` runs, whose
   `original_command` begins `/tmp/v8rep_era/src/main/train_rl_agent.py` — a **historical**
   record of where the code ran, not a resume input (`--model` names a `models/` zip).

So the 18 RESUMABLE trees, **16.5 GB — 62% of the 26.7 GB of worktrees** — are recoverable at
the cost of one `git worktree add` each, not at the cost of a run. They are still listed as a
**human decision** rather than auto-removable, because a decision to spend that is the owner's.

---

## Block 1 — 7 AGENT-LEFTOVERS the owner can remove with one paste

Every row: no live process, no `models/` reference, `main..HEAD` is empty (so every commit is
already on main), and every tracked modification is **byte-identical to main today** (checked
with `git hash-object <file>` against `git rev-parse main:<file>`) — there is nothing on disk
that git does not already have.

```bash
cd /home/goodlad/dev/gen3ai   # 🚨 the SOP trap: NEVER run this from inside a worktree
# ~wt/admiring-lamport-5a16f3  (114 MB, last commit 2026-05-31, 99d old, has a submodule checkout; 1 untracked scratch file(s))
git worktree remove /home/goodlad/dev/gen3ai/.claude/worktrees/admiring-lamport-5a16f3 --force
# ~wt/agent-a1f66f946603d8820  (0 MB, last commit 2026-08-23, 15d old, directory already gone; LOCKED — see the note below)
git worktree remove /home/goodlad/dev/gen3ai/.claude/worktrees/agent-a1f66f946603d8820
# ~wt/amazing-dijkstra-14a0b4  (44 MB, last commit 2026-06-10, 89d old, 1 untracked scratch file(s))
git worktree remove /home/goodlad/dev/gen3ai/.claude/worktrees/amazing-dijkstra-14a0b4 --force
# ~wt/bridge-cse_01NASVy92Pf1MGTCkSCJZbvP  (1147 MB, last commit 2026-08-03, 35d old, has a submodule checkout; 1 untracked scratch file(s))
git worktree remove /home/goodlad/dev/gen3ai/.claude/worktrees/bridge-cse_01NASVy92Pf1MGTCkSCJZbvP --force
# ~wt/great-agnesi-47f33e  (22 MB, last commit 2026-05-20, 110d old)
git worktree remove /home/goodlad/dev/gen3ai/.claude/worktrees/great-agnesi-47f33e --force
# ~wt/hungry-gauss-9de252  (22 MB, last commit 2026-05-20, 110d old)
git worktree remove /home/goodlad/dev/gen3ai/.claude/worktrees/hungry-gauss-9de252 --force
# ~wt/wizardly-mendel-5d59c7  (220 MB, last commit 2026-06-02, 97d old, has a submodule checkout; 1 untracked scratch file(s))
git worktree remove /home/goodlad/dev/gen3ai/.claude/worktrees/wizardly-mendel-5d59c7 --force

git worktree prune            # drops the registration of any directory already deleted
```

Reclaims **1569 MB**.

**Three traps, all from the SOP and all real here** (the `--force` above is only for untracked scratch files — every tracked change in these seven is byte-identical to main):

* 🚨 **`cd` to the main checkout first.** `git worktree remove` from inside a worktree that is
  itself being removed leaves the shell in a deleted directory (hit 4x per memory).
* 🚨 **A worktree with a submodule may resist `git worktree remove`** — 47 of the 76 have a
  populated `deps/pokemon-showdown/.git`. If `remove` refuses, `rm -rf <path> && git worktree
  prune` from the main checkout is the documented follow-up.
* 🚨 **`~wt/agent-a1f66f946603d8820` is LOCKED and its directory is already gone**
  (`locked claude agent agent-a1f66f946603d8820 (pid 1436002 start 28456866)`; pid 1436002 is
  not running). `git worktree remove` refuses a locked entry — `git worktree unlock <path>`
  first, or let `git worktree prune` take it.

## Block 2 — 10 worktrees created TODAY, clean and fully merged

Same evidence as Block 1 (clean, `main..HEAD` empty, no reference, no process), but every one
was branched **today** and carries a branch whose work has already landed. They read as this
evening's sibling agent sessions. **A session with no process right now may still be alive and
idle**, so these are printed separately: remove them once the owner knows those sessions ended.

```bash
cd /home/goodlad/dev/gen3ai
git worktree remove /home/goodlad/dev/gen3ai-wt-bl   # branch backlog-draw-bucket, 224 MB
git worktree remove /home/goodlad/dev/gen3ai-wt-d1   # branch d1-verdict-docs, 224 MB
git worktree remove /home/goodlad/dev/gen3ai-wt-quiet   # branch quiet-cadence, 269 MB
git worktree remove /home/goodlad/dev/gen3ai-wt-r2   # branch restart2-ruling, 224 MB
git worktree remove /home/goodlad/dev/gen3ai-wt-ra5   # branch ra5-discharged, 224 MB
git worktree remove /home/goodlad/dev/gen3ai-wt-rr   # branch restart-read-rulings, 224 MB
git worktree remove /home/goodlad/dev/gen3ai-wt-rule15   # branch rule15-regime, 224 MB
git worktree remove /home/goodlad/dev/gen3ai-wt-sop   # branch sop-opus-drive, 268 MB
git worktree remove /home/goodlad/dev/gen3ai-wt-u42   # branch u42-mechanism, 224 MB
git worktree remove /home/goodlad/dev/gen3ai-wt/census   # branch census-0907, 270 MB  # THIS census's own worktree — scripts/land.sh removes it
```

Reclaims a further **2375 MB**.

---

## Needs a human decision — 55 worktrees

### (a) 18 RESUMABLE — a run's pin lives here (16.5 GB)

Each is detached at, or branched on, a commit that at least one run's `metadata.json` records
as `git_hash` (the value `resolve_pin` would use on a resume) or in `pin_history`. Per the
finding above, **removing the directory does not endanger the resume** — but it is the owner's
call, and this is where the disk is.

| worktree | HEAD | run(s) that record this commit | size |
|---|---|---|---:|
| `~wt/gen3-run-0807` | `e60a1e14` (detached) | `ai_v9_04_gen3_k6_recency_40m_0807` | 2708 MB |
| `~wt/gen1-run-0804` | `1d1e7ef5` (detached) | `ai_v9_01_gen1_edges6_40m_0804` | 2637 MB |
| `~wt/gen2-run-0805` | `ffa851e4` (detached) | `ai_v9_02_gen2_full11_40m_0805` | 2591 MB |
| `~wt/gen25-run-0806` | `5db97312` (detached) | `ai_v9_03_gen25_consequence_25m_0806` | 1862 MB |
| `~wt/gen4-run-0808` | `d9af909b` (detached) | `ai_v9_05_gen4_rehome_25m_0808` | 1809 MB |
| `~wt/gen5-run-0809` | `6aac795a` (detached) | `ai_v9_06_gen5_no_concat_0809` | 1613 MB |
| `~wt/gen6-run-0810` | `22cf5779` (detached) | `ai_v9_07_gen6_seed_vicreg_0810` | 1336 MB |
| `~wt/gen7-run-0811` | `ec32c93b` (detached) | `ai_v9_08_gen7_seed_quantile_0811` | 1068 MB |
| `~wt/g13-ladder` | `1fa47332` (detached) | `ai_v9_15_gen13_hb_events_stack_0817` | 179 MB |
| `~wt/bridge-cse_01TSFR2s8tvVD5gjPj9i384e` | `b13b30b2` (worktree-bridge-cse_01TSFR2s8tvVD5gjPj9i384e) | `ai_v8_11_offense10_exploiter_0724`, `ai_v8_12_defensive20_exploiter_0724` +12 more | 176 MB |
| `~wt/gen15-pin` | `ff1daaef` (detached) | `ai_v9_18_gen15_v8rewards_0818` | 174 MB |
| `/tmp/v8rep_era` | `b13b30b2` (detached) | `ai_v8_11_offense10_exploiter_0724`, `ai_v8_12_defensive20_exploiter_0724` +12 more | 161 MB |
| `~wt/bridge-cse_019GdmzD7b4P1f3D2LiTJdfM` | `c27d843f` (worktree-bridge-cse_019GdmzD7b4P1f3D2LiTJdfM) | `ai_v7_14_league_capstone_0712` | 149 MB |
| `/tmp/probeP_v8era` | `b13b30b2` (detached) | `ai_v8_11_offense10_exploiter_0724`, `ai_v8_12_defensive20_exploiter_0724` +12 more | 118 MB |
| `~wt/bridge-cse_01Ssk4KAWEXVzvoaqk9TB9uw` | `74b32f1e` (worktree-bridge-cse_01Ssk4KAWEXVzvoaqk9TB9uw) | `ai_v8_19_def20_lut_zeroinit_0727`, `ai_v8_20_rand10_nolut_0727` | 117 MB |
| `~wt/wf_276609d4-2e2-4` | `bf896e49` (worktree-wf_276609d4-2e2-4) | `ai_v6_01_belief_53m_0613` | 89 MB |
| `~wt/wf_361e712e-ad0-4` | `bf896e49` (worktree-wf_361e712e-ad0-4) | `ai_v6_01_belief_53m_0613` | 89 MB |
| `~wt/bridge-cse_01E91kQMm1QCt26piTXtm1yh` | `33f564f7` (worktree-bridge-cse_01E91kQMm1QCt26piTXtm1yh) | `ai_v6_11_unified_obs_fixed_0618` | 60 MB |

### (b) 37 DIRTY-UNMERGED — something here is in no git object

A row lands here when `main..HEAD` is non-empty, **or** at least one modified tracked file's
working content hashes to a blob that `git cat-file -e` cannot find in the object database.
That is the only test that separates *uncommitted work* from *a stale checkout of work that
later landed*: if the blob exists, the content was committed somewhere and survives removal.

**Four have real commits ahead of main** — these are the ones to read first:

| worktree | branch | commits not on main |
|---|---|---|
| `~wt/agent-ab03e34b7b7519ae4` | `worktree-agent-ab03e34b7b7519ae4` | 3 |
| `~wt/agent-a43d9bbef6428c986` | `worktree-agent-a43d9bbef6428c986` | 1 |
| `~wt/agent-ac11057e5760f5822` | `worktree-agent-ac11057e5760f5822` | 1 |
| `~wt/agent-ac6b41845e37b446b` | `worktree-agent-ac6b41845e37b446b` | 1 |

`~wt/agent-ab03e34b7b7519ae4`:

  * `b8a7a1dd docs(research): iter 3 scored — paired ROLLOUTS reject 97.5% of what the leaf certifies, and resolve the rest at a COIN FLIP`
  * `cf1c227a docs(research): iter-3 scoring script — three arms through ONE arm_block, plus the paired-difference read the verdict counters cannot show`
  * `3c8eb97f feat(search): the built --defensive-confirm had no reachable CLOCK — every confirm would have declined for want of a second`

`~wt/agent-a43d9bbef6428c986`:

  * `27598c23 probe(K): the certified overrules are REAL — G's +2.2pp survives a marginalized opponent, and the leaf is ACQUITTED`

`~wt/agent-ac11057e5760f5822`:

  * `803581e6 feat(research): the OWN-SIDE IMPUTATION meter — the confound Metamon never quantified, sized`

`~wt/agent-ac6b41845e37b446b`:

  * `f9bebb8d docs(research_state): diversity-first tick-1 ordering for the exploiter queue`

**The other 33 have no commits ahead of main** — only uncommitted working-tree content that
exists in no git object. `age` is the age of the worktree's HEAD commit; `gone` counts
modified paths that **no longer exist on `main` at all** (the file was deleted or renamed since,
so the delta is against a file the tree no longer has — strong evidence the work is superseded).

| worktree | age (d) | files only here | modified paths gone from main | size |
|---|---:|---:|---:|---:|
| `~wt/gracious-lamarr-9f30c9` | 113 | 2 | 0 | 21 MB |
| `~wt/dreamy-hopper-120932` | 112 | 0 | 0 | 23 MB |
| `~wt/beautiful-chatelet-38f4b3` | 111 | 1 | 1 | 22 MB |
| `~wt/great-robinson-eb89bb` | 111 | 5 | 1 | 24 MB |
| `~wt/reverent-lovelace-af97be` | 110 | 1 | 0 | 70 MB |
| `~wt/clever-panini-eb4d41` | 109 | 7 | 0 | 25 MB |
| `~wt/flamboyant-curie-f6e31f` | 107 | 2 | 1 | 25 MB |
| `~wt/vigorous-faraday-7b286f` | 106 | 1 | 0 | 24 MB |
| `~wt/loving-moore-4d73b1` | 104 | 13 | 2 | 337 MB |
| `~wt/relaxed-boyd-8cb411` | 101 | 3 | 1 | 85 MB |
| `~wt/gifted-lehmann-4a43d7` | 100 | 6 | 3 | 430 MB |
| `~wt/priceless-agnesi-ebcb9a` | 100 | 2 | 0 | 115 MB |
| `~wt/flamboyant-dijkstra-2c2723` | 99 | 3 | 0 | 189 MB |
| `~wt/hopeful-tharp-be0911` | 99 | 7 | 0 | 115 MB |
| `~wt/romantic-faraday-d97cb0` | 98 | 5 | 0 | 46 MB |
| `~wt/blissful-swanson-62495b` | 95 | 9 | 4 | 92 MB |
| `~wt/sweet-jackson-5d9d4f` | 95 | 2 | 0 | 42 MB |
| `~wt/optimistic-allen-c14db0` | 93 | 5 | 0 | 94 MB |
| `~wt/keen-pasteur-f858a5` | 92 | 1 | 0 | 42 MB |
| `~wt/pensive-chatelet-6d9c29` | 91 | 14 | 1 | 230 MB |
| `~wt/amazing-mcnulty-a6170f` | 89 | 16 | 2 | 150 MB |
| `~wt/loving-allen-1e99ee` | 89 | 6 | 2 | 45 MB |
| `~wt/confident-saha-22b6c6` | 87 | 2 | 0 | 97 MB |
| `~wt/kind-mclean-f32b40` | 87 | 5 | 2 | 90 MB |
| `~wt/bridge-cse_019E1rkd1eyXh4HCJ6y5NkuZ` | 85 | 2 | 0 | 46 MB |
| `~wt/bridge-cse_01Vf3Bs2s2obseG8qKduvaz8` | 85 | 16 | 4 | 204 MB |
| `~wt/bridge-cse_01EGcKR8QPNJS1ioJwoBV5Yy` | 82 | 7 | 1 | 121 MB |
| `~wt/bridge-cse_01B8c76UWCWqajvGStqv3NJ3` | 80 | 4 | 2 | 103 MB |
| `~wt/bridge-cse_01HLevJPvmEx81GR3d2HZjsz` | 72 | 4 | 0 | 96 MB |
| `~wt/bridge-cse_01XEDrp4tHiLqbpRt1ynTzA9` | 64 | 1 | 0 | 49 MB |
| `~wt/cleanup-p2` | 23 | 23 | 2 | 193 MB |
| `~wt/bridge-cse_011tmX2mVpE25NJz7hMzxe5S` | 22 | 1 | 0 | 1042 MB |
| `~wt/gen16-package` | 17 | 1 | 0 | 168 MB |

**No verdict is offered on these.** A three-month-old checkout whose modified file no longer
exists on main is almost certainly superseded, but "almost certainly" is not a measurement and
the content is genuinely unrecoverable after a `--force` removal. The per-file lists are below.

---

## LIVE — 4, do not touch

| worktree | what is running |
|---|---|
| `/tmp/launcher-f971caf2-ypftpd3a` | **the production arm.** `train_rl_agent.py` pid 3101175 runs out of `<wt>/src/`; launcher pid 2635463 holds `--pin-commit f971caf2`. Its run `ai_v12_02_winprob_critic` records `f971caf2…` in both `git_hash` and `pin_history`. |
| `/home/goodlad/dev/gen3ai-wt/stgate` | a sibling agent's pytest child (pid 3277963, `--debug` search-teacher composition test) plus 5 more pids with this cwd |
| `/home/goodlad/dev/gen3ai-wt/slowgate` | 51 pids with this cwd — a sibling agent running the suite |
| `~wt/docs-trim` | one bash (pid 2952406, ~6.5 h) — branch `debt-0907` |

`/home/goodlad/dev/gen3ai` itself holds 207 pids' cwd and is clean.

Two of these — the launcher tree and `/tmp/probeP_v8era` — **cannot be `git status`ed**:
`error: expected submodule path 'deps/pokemon-showdown' not to be a symbolic link`. Their
dirtiness is therefore **UNDECIDABLE**, recorded as such rather than guessed. It changes no
verdict (the launcher tree is LIVE, `probeP_v8era` is RESUMABLE), but it is the reason those
two rows read `dirty = ?`.

---

<details>
<summary><b>Per-row detail — all 77 entries</b></summary>

| worktree | branch / detached | HEAD | last commit | dirty | ahead | class |
|---|---|---|---|---|---:|---|
| `~-wt-bl` | `backlog-draw-bucket` | `0fafb78d` | 2026-09-07 | 0 | 0 | AGENT-LEFTOVER |
| `~-wt-d1` | `d1-verdict-docs` | `1e1f478c` | 2026-09-07 | 0 | 0 | AGENT-LEFTOVER |
| `~-wt-quiet` | `quiet-cadence` | `10f168d0` | 2026-09-07 | 0 | 0 | AGENT-LEFTOVER |
| `~-wt-r2` | `restart2-ruling` | `4b4b4f54` | 2026-09-07 | 0 | 0 | AGENT-LEFTOVER |
| `~-wt-ra5` | `ra5-discharged` | `08de50d4` | 2026-09-07 | 0 | 0 | AGENT-LEFTOVER |
| `~-wt-rr` | `restart-read-rulings` | `70f984c3` | 2026-09-06 | 0 | 0 | AGENT-LEFTOVER |
| `~-wt-rule15` | `rule15-regime` | `62d4948e` | 2026-09-06 | 0 | 0 | AGENT-LEFTOVER |
| `~-wt-sop` | `sop-opus-drive` | `d4bff9a2` | 2026-09-07 | 0 | 0 | AGENT-LEFTOVER |
| `~-wt-u42` | `u42-mechanism` | `0005cafe` | 2026-09-07 | 0 | 0 | AGENT-LEFTOVER |
| `~-wt/census` | `census-0907` | `a1e8424b` | 2026-09-07 | 0 | 0 | AGENT-LEFTOVER |
| `~wt/admiring-lamport-5a16f3` | `claude/admiring-lamport-5a16f3` | `e35df881` | 2026-05-31 | 1 | 0 | AGENT-LEFTOVER |
| `~wt/agent-a1f66f946603d8820` | `worktree-agent-a1f66f946603d8820` | `7014b2ce` | 2026-08-23 | ? | 0 | AGENT-LEFTOVER |
| `~wt/amazing-dijkstra-14a0b4` | `claude/amazing-dijkstra-14a0b4` | `c5f23ec2` | 2026-06-10 | 1 | 0 | AGENT-LEFTOVER |
| `~wt/bridge-cse_01NASVy92Pf1MGTCkSCJZbvP` | `worktree-bridge-cse_01NASVy92Pf1MGTCkSCJZbvP` | `fc3637e1` | 2026-08-03 | 1 | 0 | AGENT-LEFTOVER |
| `~wt/great-agnesi-47f33e` | `claude/great-agnesi-47f33e` | `cd55b57b` | 2026-05-20 | 1 | 0 | AGENT-LEFTOVER |
| `~wt/hungry-gauss-9de252` | `claude/hungry-gauss-9de252` | `cd55b57b` | 2026-05-20 | 1 | 0 | AGENT-LEFTOVER |
| `~wt/wizardly-mendel-5d59c7` | `claude/wizardly-mendel-5d59c7` | `4fc99dcf` | 2026-06-02 | 1 | 0 | AGENT-LEFTOVER |
| `~wt/agent-a43d9bbef6428c986` | `worktree-agent-a43d9bbef6428c986` | `27598c23` | 2026-08-29 | 0 | 1 | DIRTY-UNMERGED |
| `~wt/agent-ab03e34b7b7519ae4` | `worktree-agent-ab03e34b7b7519ae4` | `b8a7a1dd` | 2026-08-31 | 0 | 3 | DIRTY-UNMERGED |
| `~wt/agent-ac11057e5760f5822` | `worktree-agent-ac11057e5760f5822` | `803581e6` | 2026-08-23 | 0 | 1 | DIRTY-UNMERGED |
| `~wt/agent-ac6b41845e37b446b` | `worktree-agent-ac6b41845e37b446b` | `f9bebb8d` | 2026-08-23 | 0 | 1 | DIRTY-UNMERGED |
| `~wt/amazing-mcnulty-a6170f` | `claude/outgoing-ko` | `c5f23ec2` | 2026-06-10 | 24 | 0 | DIRTY-UNMERGED |
| `~wt/beautiful-chatelet-38f4b3` | `claude/beautiful-chatelet-38f4b3` | `02f165b9` | 2026-05-19 | 1 | 0 | DIRTY-UNMERGED |
| `~wt/blissful-swanson-62495b` | `claude/blissful-swanson-62495b` | `fb7e753d` | 2026-06-04 | 9 | 0 | DIRTY-UNMERGED |
| `~wt/bridge-cse_011tmX2mVpE25NJz7hMzxe5S` | `worktree-bridge-cse_011tmX2mVpE25NJz7hMzxe5S` | `f6e5b07b` | 2026-08-16 | 1 | 0 | DIRTY-UNMERGED |
| `~wt/bridge-cse_019E1rkd1eyXh4HCJ6y5NkuZ` | `worktree-bridge-cse_019E1rkd1eyXh4HCJ6y5NkuZ` | `f5ae4996` | 2026-06-14 | 2 | 0 | DIRTY-UNMERGED |
| `~wt/bridge-cse_01B8c76UWCWqajvGStqv3NJ3` | `worktree-bridge-cse_01B8c76UWCWqajvGStqv3NJ3` | `816ec848` | 2026-06-19 | 5 | 0 | DIRTY-UNMERGED |
| `~wt/bridge-cse_01EGcKR8QPNJS1ioJwoBV5Yy` | `worktree-bridge-cse_01EGcKR8QPNJS1ioJwoBV5Yy` | `4a476339` | 2026-06-17 | 7 | 0 | DIRTY-UNMERGED |
| `~wt/bridge-cse_01HLevJPvmEx81GR3d2HZjsz` | `worktree-bridge-cse_01HLevJPvmEx81GR3d2HZjsz` | `ee8f2b92` | 2026-06-27 | 7 | 0 | DIRTY-UNMERGED |
| `~wt/bridge-cse_01Vf3Bs2s2obseG8qKduvaz8` | `worktree-bridge-cse_01Vf3Bs2s2obseG8qKduvaz8` | `4e56fb3b` | 2026-06-14 | 17 | 0 | DIRTY-UNMERGED |
| `~wt/bridge-cse_01XEDrp4tHiLqbpRt1ynTzA9` | `worktree-bridge-cse_01XEDrp4tHiLqbpRt1ynTzA9` | `5d77790d` | 2026-07-05 | 2 | 0 | DIRTY-UNMERGED |
| `~wt/cleanup-p2` | `worktree-cleanup-p2` | `5516a41f` | 2026-08-15 | 35 | 0 | DIRTY-UNMERGED |
| `~wt/clever-panini-eb4d41` | `claude/clever-panini-eb4d41` | `7a4321d2` | 2026-05-21 | 7 | 0 | DIRTY-UNMERGED |
| `~wt/confident-saha-22b6c6` | `claude/confident-saha-22b6c6` | `740cb5d2` | 2026-06-12 | 11 | 0 | DIRTY-UNMERGED |
| `~wt/dreamy-hopper-120932` | `claude/dreamy-hopper-120932` | `0933b461` | 2026-05-18 | 1 | 0 | DIRTY-UNMERGED |
| `~wt/flamboyant-curie-f6e31f` | `claude/flamboyant-curie-f6e31f` | `cbd2801a` | 2026-05-23 | 2 | 0 | DIRTY-UNMERGED |
| `~wt/flamboyant-dijkstra-2c2723` | `claude/flamboyant-dijkstra-2c2723` | `5a21b84c` | 2026-05-31 | 3 | 0 | DIRTY-UNMERGED |
| `~wt/gen16-package` | `gen16-package` | `4ce0c5b4` | 2026-08-21 | 1 | 0 | DIRTY-UNMERGED |
| `~wt/gifted-lehmann-4a43d7` | `claude/gifted-lehmann-4a43d7` | `9e1adf0f` | 2026-05-30 | 7 | 0 | DIRTY-UNMERGED |
| `~wt/gracious-lamarr-9f30c9` | `claude/gracious-lamarr-9f30c9` | `673827b2` | 2026-05-17 | 2 | 0 | DIRTY-UNMERGED |
| `~wt/great-robinson-eb89bb` | `claude/great-robinson-eb89bb` | `564e3f5b` | 2026-05-19 | 6 | 0 | DIRTY-UNMERGED |
| `~wt/hopeful-tharp-be0911` | `claude/hopeful-tharp-be0911` | `32588e93` | 2026-05-31 | 7 | 0 | DIRTY-UNMERGED |
| `~wt/keen-pasteur-f858a5` | `claude/keen-pasteur-f858a5` | `934c78f3` | 2026-06-07 | 1 | 0 | DIRTY-UNMERGED |
| `~wt/kind-mclean-f32b40` | `claude/kind-mclean-f32b40` | `740cb5d2` | 2026-06-12 | 7 | 0 | DIRTY-UNMERGED |
| `~wt/loving-allen-1e99ee` | `claude/loving-allen-1e99ee` | `c5f23ec2` | 2026-06-10 | 6 | 0 | DIRTY-UNMERGED |
| `~wt/loving-moore-4d73b1` | `claude/loving-moore-4d73b1` | `b5025fc2` | 2026-05-26 | 14 | 0 | DIRTY-UNMERGED |
| `~wt/optimistic-allen-c14db0` | `claude/optimistic-allen-c14db0` | `9a37b712` | 2026-06-06 | 8 | 0 | DIRTY-UNMERGED |
| `~wt/pensive-chatelet-6d9c29` | `claude/pensive-chatelet-6d9c29` | `21587844` | 2026-06-08 | 14 | 0 | DIRTY-UNMERGED |
| `~wt/priceless-agnesi-ebcb9a` | `claude/priceless-agnesi-ebcb9a` | `7c997724` | 2026-05-30 | 2 | 0 | DIRTY-UNMERGED |
| `~wt/relaxed-boyd-8cb411` | `claude/relaxed-boyd-8cb411` | `07190d37` | 2026-05-29 | 3 | 0 | DIRTY-UNMERGED |
| `~wt/reverent-lovelace-af97be` | `claude/reverent-lovelace-af97be` | `d77e2bc9` | 2026-05-20 | 1 | 0 | DIRTY-UNMERGED |
| `~wt/romantic-faraday-d97cb0` | `claude/romantic-faraday-d97cb0` | `248c4bcb` | 2026-06-01 | 5 | 0 | DIRTY-UNMERGED |
| `~wt/sweet-jackson-5d9d4f` | `claude/sweet-jackson-5d9d4f` | `27da5fc5` | 2026-06-04 | 2 | 0 | DIRTY-UNMERGED |
| `~wt/vigorous-faraday-7b286f` | `claude/vigorous-faraday-7b286f` | `716cdc3e` | 2026-05-24 | 1 | 0 | DIRTY-UNMERGED |
| `~-wt/slowgate` | `slowgate-0907` | `bbbb66c1` | 2026-09-07 | 9 | 0 | LIVE |
| `~-wt/stgate` | `stgate-0907` | `8723090a` | 2026-09-07 | 7 | 0 | LIVE |
| `~wt/docs-trim` | `debt-0907` | `60075ea9` | 2026-09-07 | 0 | 0 | LIVE |
| `/tmp/launcher-f971caf2-ypftpd3a` | detached | `f971caf2` | 2026-09-06 | ? | 0 | LIVE |
| `~` | `main` | `a1e8424b` | 2026-09-07 | 0 | 0 | MAIN |
| `~wt/bridge-cse_019GdmzD7b4P1f3D2LiTJdfM` | `worktree-bridge-cse_019GdmzD7b4P1f3D2LiTJdfM` | `c27d843f` | 2026-07-12 | 7 | 0 | RESUMABLE |
| `~wt/bridge-cse_01E91kQMm1QCt26piTXtm1yh` | `worktree-bridge-cse_01E91kQMm1QCt26piTXtm1yh` | `33f564f7` | 2026-06-18 | 5 | 0 | RESUMABLE |
| `~wt/bridge-cse_01Ssk4KAWEXVzvoaqk9TB9uw` | `worktree-bridge-cse_01Ssk4KAWEXVzvoaqk9TB9uw` | `74b32f1e` | 2026-07-27 | 5 | 0 | RESUMABLE |
| `~wt/bridge-cse_01TSFR2s8tvVD5gjPj9i384e` | `worktree-bridge-cse_01TSFR2s8tvVD5gjPj9i384e` | `b13b30b2` | 2026-07-24 | 6 | 0 | RESUMABLE |
| `~wt/g13-ladder` | detached | `1fa47332` | 2026-08-16 | 1 | 0 | RESUMABLE |
| `~wt/gen1-run-0804` | detached | `1d1e7ef5` | 2026-08-04 | 1 | 0 | RESUMABLE |
| `~wt/gen15-pin` | detached | `ff1daaef` | 2026-08-18 | 2 | 0 | RESUMABLE |
| `~wt/gen2-run-0805` | detached | `ffa851e4` | 2026-08-04 | 1 | 0 | RESUMABLE |
| `~wt/gen25-run-0806` | detached | `5db97312` | 2026-08-06 | 1 | 0 | RESUMABLE |
| `~wt/gen3-run-0807` | detached | `e60a1e14` | 2026-08-07 | 1 | 0 | RESUMABLE |
| `~wt/gen4-run-0808` | detached | `d9af909b` | 2026-08-08 | 1 | 0 | RESUMABLE |
| `~wt/gen5-run-0809` | detached | `6aac795a` | 2026-08-09 | 1 | 0 | RESUMABLE |
| `~wt/gen6-run-0810` | detached | `22cf5779` | 2026-08-10 | 1 | 0 | RESUMABLE |
| `~wt/gen7-run-0811` | detached | `ec32c93b` | 2026-08-11 | 2 | 0 | RESUMABLE |
| `~wt/wf_276609d4-2e2-4` | `worktree-wf_276609d4-2e2-4` | `bf896e49` | 2026-06-13 | 1 | 0 | RESUMABLE |
| `~wt/wf_361e712e-ad0-4` | `worktree-wf_361e712e-ad0-4` | `bf896e49` | 2026-06-13 | 4 | 0 | RESUMABLE |
| `/tmp/probeP_v8era` | detached | `b13b30b2` | 2026-07-24 | ? | 0 | RESUMABLE |
| `/tmp/v8rep_era` | detached | `b13b30b2` | 2026-07-24 | 0 | 0 | RESUMABLE |

</details>

<details>
<summary><b>Per-file detail for the DIRTY-UNMERGED rows</b></summary>

**`~wt/agent-a43d9bbef6428c986`** — HEAD `27598c23` (2026-08-29), 192 MB

* 1 commit(s) not on main:
  * `27598c23 probe(K): the certified overrules are REAL — G's +2.2pp survives a marginalized opponent, and the leaf is ACQUITTED`

**`~wt/agent-ab03e34b7b7519ae4`** — HEAD `b8a7a1dd` (2026-08-31), 241 MB

* 3 commit(s) not on main:
  * `b8a7a1dd docs(research): iter 3 scored — paired ROLLOUTS reject 97.5% of what the leaf certifies, and resolve the rest at a COIN FLIP`
  * `cf1c227a docs(research): iter-3 scoring script — three arms through ONE arm_block, plus the paired-difference read the verdict counters cannot show`
  * `3c8eb97f feat(search): the built --defensive-confirm had no reachable CLOCK — every confirm would have declined for want of a second`

**`~wt/agent-ac11057e5760f5822`** — HEAD `803581e6` (2026-08-23), 232 MB

* 1 commit(s) not on main:
  * `803581e6 feat(research): the OWN-SIDE IMPUTATION meter — the confound Metamon never quantified, sized`

**`~wt/agent-ac6b41845e37b446b`** — HEAD `f9bebb8d` (2026-08-23), 131 MB

* 1 commit(s) not on main:
  * `f9bebb8d docs(research_state): diversity-first tick-1 ordering for the exploiter queue`

**`~wt/amazing-mcnulty-a6170f`** — HEAD `c5f23ec2` (2026-06-10), 150 MB

* content in NO git object (16): `CLAUDE.md`, `src/agents/gen3_data/priors.py`, `src/agents/gen3_data/priors_test.py`, `src/agents/model/CLAUDE.md`, `src/agents/model/model_version.py`, `src/agents/observation/CLAUDE.md`, `src/agents/observation/constants.py`, `src/agents/observation/incoming_damage_encoder.py`, `src/agents/observation/incoming_damage_encoder_test.py`, `src/agents/observation/reactive.py`, `src/agents/training/golden_obs_fixture.json`, `src/agents/training/poke_env_gaps/incoming_damage_fuzz_test.py`, `src/main/prober/CLAUDE.md`, `src/main/prober/engine.py`, `src/main/prober/engine_test.py`, `src/main/prober/model.py`
* modified paths that no longer exist on main: `src/agents/model/model_version.py`, `src/main/prober/engine.py`
* untracked (6): `designs/ai_v5/design_outgoing_damage_obs.md`, `designs/ai_v5/falsifier_missed_ko_attribution.py`, `designs/ai_v5/impl_step9_outgoing_ko_obs.md`, `src/agents/observation/outgoing_damage_encoder.py`, `src/agents/observation/outgoing_damage_encoder_test.py`, `src/agents/training/poke_env_gaps/outgoing_damage_fuzz_test.py`

**`~wt/beautiful-chatelet-38f4b3`** — HEAD `02f165b9` (2026-05-19), 22 MB

* content in NO git object (1): `src/main/launcher_ui.py`
* modified paths that no longer exist on main: `src/main/launcher_ui.py`

**`~wt/blissful-swanson-62495b`** — HEAD `fb7e753d` (2026-06-04), 92 MB

* content in NO git object (9): `src/agents/training/distill/CLAUDE.md`, `src/agents/training/distill/manager.py`, `src/agents/training/distill/manager_test.py`, `src/agents/training/distill/worker.py`, `src/agents/training/selfplay_callback.py`, `src/agents/training/selfplay_callback_test.py`, `src/main/launcher/app.py`, `src/main/launcher_app_test.py`, `src/main/train_rl_agent.py`
* modified paths that no longer exist on main: `src/agents/training/distill/CLAUDE.md`, `src/agents/training/distill/manager.py`, `src/agents/training/distill/manager_test.py`, `src/agents/training/distill/worker.py`

**`~wt/bridge-cse_011tmX2mVpE25NJz7hMzxe5S`** — HEAD `f6e5b07b` (2026-08-16), 1042 MB

* content in NO git object (1): `designs/research_state/measurements/gen11_label_only_winprob_verdict.json`

**`~wt/bridge-cse_019E1rkd1eyXh4HCJ6y5NkuZ`** — HEAD `f5ae4996` (2026-06-14), 46 MB

* content in NO git object (2): `src/agents/training/battle_recorder.py`, `src/agents/training/battle_recorder_test.py`

**`~wt/bridge-cse_01B8c76UWCWqajvGStqv3NJ3`** — HEAD `816ec848` (2026-06-19), 103 MB

* content in NO git object (4): `src/main/prober/CLAUDE.md`, `src/main/prober/app.py`, `src/main/prober/engine.py`, `src/main/prober/engine_test.py`
* modified paths that no longer exist on main: `src/main/prober/app.py`, `src/main/prober/engine.py`
* untracked (1): `designs/learning/belief_head_output_structure.md`

**`~wt/bridge-cse_01EGcKR8QPNJS1ioJwoBV5Yy`** — HEAD `4a476339` (2026-06-17), 121 MB

* content in NO git object (7): `src/agents/model/CLAUDE.md`, `src/agents/model/features_extractor.py`, `src/agents/training/CLAUDE.md`, `src/agents/training/grad_balance.py`, `src/agents/training/grad_balance_test.py`, `src/agents/training/instrumented_ppo.py`, `src/main/launcher/format.py`
* modified paths that no longer exist on main: `src/agents/training/instrumented_ppo.py`

**`~wt/bridge-cse_01HLevJPvmEx81GR3d2HZjsz`** — HEAD `ee8f2b92` (2026-06-27), 96 MB

* content in NO git object (4): `src/agents/training/CLAUDE.md`, `src/agents/training/wrappers.py`, `src/agents/training/wrappers_test.py`, `src/main/train_rl_agent.py`
* untracked (3): `designs/research_state/team_archetype_and_exploration_report.md`, `src/agents/training/team_archetype.py`, `src/agents/training/team_archetype_test.py`

**`~wt/bridge-cse_01Vf3Bs2s2obseG8qKduvaz8`** — HEAD `4e56fb3b` (2026-06-14), 204 MB

* content in NO git object (16): `CLAUDE.md`, `src/agents/gen3_mechanics.py`, `src/agents/gen3_mechanics_test.py`, `src/agents/model/model_version.py`, `src/agents/model/unified_belief_test.py`, `src/agents/observation/CLAUDE.md`, `src/agents/observation/constants.py`, `src/agents/observation/reactive.py`, `src/agents/training/obs_materializer.py`, `src/main/prober/CLAUDE.md`, `src/main/prober/app.py`, `src/main/prober/app_test.py`, `src/main/prober/engine.py`, `src/main/prober/engine_test.py`, `src/main/prober/forensics_test.py`, `src/main/prober/model.py`
* modified paths that no longer exist on main: `src/agents/model/model_version.py`, `src/main/prober/app.py`, `src/main/prober/app_test.py`, `src/main/prober/engine.py`
* untracked (1): `src/agents/training/poke_env_gaps/opp_trap_fuzz_test.py`

**`~wt/bridge-cse_01XEDrp4tHiLqbpRt1ynTzA9`** — HEAD `5d77790d` (2026-07-05), 49 MB

* content in NO git object (1): `src/agents/training/CLAUDE.md`
* untracked (1): `src/agents/training/TENSORBOARD_METRICS.md`

**`~wt/cleanup-p2`** — HEAD `5516a41f` (2026-08-15), 193 MB

* content in NO git object (23): `CLAUDE.md`, `designs/ARCHITECTURE.md`, `designs/CHANGELOG.md`, `designs/CLAUDE.md`, `designs/ai_v9/design_cleanup_journey.md`, `designs/architecture_graph.dot`, `designs/architecture_viewer.html`, `designs/flag_registry.md`, `designs/production_config.json`, `designs/pubval_deletion_decision.md`, `src/agents/model/CLAUDE.md`, `src/agents/model/arch_tables.py`, `src/agents/model/delivery_graph_snapshot.json`, `src/agents/model/features_extractor.py`, `src/agents/model/flag_registry.py`, `src/agents/model/model_version.py`, `src/agents/model/snapshot.py`, `src/agents/model/tier_contract.py`, `src/agents/model/value_entity_pool_test.py`, `src/agents/training/CLAUDE.md`, `src/agents/training/gen3_env.py`, `src/agents/training/instrumented_ppo.py`, `src/main/train_rl_agent.py`
* modified paths that no longer exist on main: `src/agents/model/model_version.py`, `src/agents/training/instrumented_ppo.py`
* untracked (1): `src/agents/model/diagnostics_default_on_test.py`

**`~wt/clever-panini-eb4d41`** — HEAD `7a4321d2` (2026-05-21), 25 MB

* content in NO git object (7): `CLAUDE.md`, `src/agents/model/features_extractor.py`, `src/agents/observation/active_context.py`, `src/agents/observation/constants.py`, `src/agents/observation/pokemon.py`, `src/agents/observation/state_encoder.py`, `src/agents/observation/turn_delta_encoder.py`

**`~wt/confident-saha-22b6c6`** — HEAD `740cb5d2` (2026-06-12), 97 MB

* content in NO git object (2): `designs/research_state/README.md`, `designs/research_state/ledger.md`
* untracked (9): `designs/research_state/levers/strong_opp_grind.md`, `designs/research_state/levers/upstream_multiply.md`, `designs/research_state/scripts/`, `grind_analysis.py`, `grind_classify.py`, `scratch/`, `src/main/prober/upstream_reframe_probe.py`, `verify_auc.py` …

**`~wt/dreamy-hopper-120932`** — HEAD `0933b461` (2026-05-18), 23 MB


**`~wt/flamboyant-curie-f6e31f`** — HEAD `cbd2801a` (2026-05-23), 25 MB

* content in NO git object (2): `src/agents/training/selfplay_callback.py`, `src/main/launcher/ui.py`
* modified paths that no longer exist on main: `src/main/launcher/ui.py`

**`~wt/flamboyant-dijkstra-2c2723`** — HEAD `5a21b84c` (2026-05-31), 189 MB

* content in NO git object (3): `CLAUDE.md`, `src/agents/training/eval_callback.py`, `src/agents/training/eval_callback_test.py`

**`~wt/gen16-package`** — HEAD `4ce0c5b4` (2026-08-21), 168 MB

* content in NO git object (1): `designs/research_state/substrate_exploiter_gates.md`

**`~wt/gifted-lehmann-4a43d7`** — HEAD `9e1adf0f` (2026-05-30), 430 MB

* content in NO git object (6): `src/agents/action/ordering_integrity.py`, `src/agents/action/ordering_integrity_test.py`, `src/agents/training/battle_context.py`, `src/agents/training/move_attribution_test.py`, `src/agents/training/poke_env_gaps/move_outcome_fuzz_e2e_test.py`, `src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py`
* modified paths that no longer exist on main: `src/agents/training/battle_context.py`, `src/agents/training/poke_env_gaps/move_outcome_fuzz_e2e_test.py`, `src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py`
* untracked (1): `designs/ai_v4/design_turn_pairing_unification.md`

**`~wt/gracious-lamarr-9f30c9`** — HEAD `673827b2` (2026-05-17), 21 MB

* content in NO git object (2): `src/agents/model/snapshot.py`, `src/main/train_rl_agent.py`

**`~wt/great-robinson-eb89bb`** — HEAD `564e3f5b` (2026-05-19), 24 MB

* content in NO git object (5): `src/agents/training/gen3_env.py`, `src/main/launcher/run.py`, `src/main/launcher/state.py`, `src/main/launcher/ui.py`, `src/main/train_rl_agent.py`
* modified paths that no longer exist on main: `src/main/launcher/ui.py`
* untracked (1): `src/agents/training/win_rate_eval_callback.py`

**`~wt/hopeful-tharp-be0911`** — HEAD `32588e93` (2026-05-31), 115 MB

* content in NO git object (7): `CLAUDE.md`, `designs/ai_v4/todo_live_battle.md`, `src/agents/battle/event_log_fuzz_test.py`, `src/agents/battle/live_view.py`, `src/agents/battle/live_view_test.py`, `src/agents/battle/strict_view_test.py`, `src/agents/observation/state_encoder_test.py`

**`~wt/keen-pasteur-f858a5`** — HEAD `934c78f3` (2026-06-07), 42 MB

* content in NO git object (1): `src/agents/training/elo.py`

**`~wt/kind-mclean-f32b40`** — HEAD `740cb5d2` (2026-06-12), 90 MB

* content in NO git object (5): `src/main/prober/CLAUDE.md`, `src/main/prober/engine.py`, `src/main/prober/query.py`, `src/main/prober/session.py`, `src/main/prober/session_test.py`
* modified paths that no longer exist on main: `src/main/prober/engine.py`, `src/main/prober/session.py`
* untracked (2): `src/main/prober/anticipation_headroom.py`, `src/main/prober/probe_battery_lossgate.py`

**`~wt/loving-allen-1e99ee`** — HEAD `c5f23ec2` (2026-06-10), 45 MB

* content in NO git object (6): `src/main/prober/CLAUDE.md`, `src/main/prober/engine.py`, `src/main/prober/engine_test.py`, `src/main/prober/query.py`, `src/main/prober/session.py`, `src/main/prober/session_test.py`
* modified paths that no longer exist on main: `src/main/prober/engine.py`, `src/main/prober/session.py`

**`~wt/loving-moore-4d73b1`** — HEAD `b5025fc2` (2026-05-26), 337 MB

* content in NO git object (13): `designs/ai_v4/todo.md`, `src/agents/observation/turn_delta_encoder.py`, `src/agents/observation/turn_delta_encoder_test.py`, `src/agents/training/battle_context.py`, `src/agents/training/battle_context_test.py`, `src/agents/training/battle_recorder.py`, `src/agents/training/battle_recorder_test.py`, `src/agents/training/episode_tracker.py`, `src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py`, `src/agents/training/reward_invariants_e2e_test.py`, `src/agents/training/reward_manager.py`, `src/agents/training/reward_manager_test.py`, `src/poke_env/battle/abstract_battle.py`
* modified paths that no longer exist on main: `src/agents/training/battle_context.py`, `src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py`
* untracked (1): `src/agents/training/poke_env_gaps/cant_reason_fuzz_e2e_test.py`

**`~wt/optimistic-allen-c14db0`** — HEAD `9a37b712` (2026-06-06), 94 MB

* content in NO git object (5): `data/pokemon/gen3_moves.json`, `src/agents/gen3_data/moves.py`, `src/agents/gen3_data/moves_test.py`, `src/agents/training/choice_band_tracker.py`, `tools/pokemon_data_extractor/sync.py`
* untracked (3): `src/agents/training/speed_belief.py`, `src/agents/training/speed_belief_test.py`, `src/agents/training/speed_belief_tracker_fuzz_test.py`

**`~wt/pensive-chatelet-6d9c29`** — HEAD `21587844` (2026-06-08), 230 MB

* content in NO git object (14): `src/agents/model/snapshot.py`, `src/agents/training/CLAUDE.md`, `src/agents/training/elo.py`, `src/agents/training/elo_test.py`, `src/agents/training/eval_callback.py`, `src/agents/training/eval_sharding/__init__.py`, `src/agents/training/eval_sharding/units.py`, `src/agents/training/eval_sharding_test.py`, `src/agents/training/rating.py`, `src/main/eval_worker.py`, `src/main/prober/CLAUDE.md`, `src/main/prober/discovery.py`, `src/main/prober/discovery_test.py`, `src/main/prober/session.py`
* modified paths that no longer exist on main: `src/main/prober/session.py`

**`~wt/priceless-agnesi-ebcb9a`** — HEAD `7c997724` (2026-05-30), 115 MB

* content in NO git object (2): `src/agents/training/battle_recorder.py`, `src/agents/training/battle_recorder_test.py`

**`~wt/relaxed-boyd-8cb411`** — HEAD `07190d37` (2026-05-29), 85 MB

* content in NO git object (3): `src/agents/training/battle_recorder.py`, `src/agents/training/eval_callback.py`, `src/agents/training/replay_recorder.py`
* modified paths that no longer exist on main: `src/agents/training/replay_recorder.py`

**`~wt/reverent-lovelace-af97be`** — HEAD `d77e2bc9` (2026-05-20), 70 MB

* content in NO git object (1): `src/agents/model/features_extractor.py`

**`~wt/romantic-faraday-d97cb0`** — HEAD `248c4bcb` (2026-06-01), 46 MB

* content in NO git object (5): `src/main/launcher/CLAUDE.md`, `src/main/launcher/__init__.py`, `src/main/launcher/checkpoint.py`, `src/main/launcher/run.py`, `src/main/launcher_test.py`

**`~wt/sweet-jackson-5d9d4f`** — HEAD `27da5fc5` (2026-06-04), 42 MB

* content in NO git object (2): `src/agents/training/selfplay_callback.py`, `src/agents/training/selfplay_callback_test.py`

**`~wt/vigorous-faraday-7b286f`** — HEAD `716cdc3e` (2026-05-24), 24 MB

* content in NO git object (1): `designs/ai_v3/todo.md`

</details>

