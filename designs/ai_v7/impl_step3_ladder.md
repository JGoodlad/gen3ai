# Implementation: Step 3 — Ladder Run

Deploy each of the three specialised MCTS players on the real Showdown Gen 3 OU ladder.
Track ELO per team, identify the primary ladder entry, and collect replays for future
meta-alignment work.

## Motivation

The league and self-play pool are proxies for the ladder. This step is the first time the
agent plays against real humans, which tests whether the training distribution is a good
approximation and surfaces any systematic weaknesses that did not appear in self-play.
ELO is the ground truth metric — everything before this step was optimising a proxy.

---

## Deployment Setup

### Account and Authentication

Each team runs under a separate Showdown account so ELO is tracked independently per
team. Account names and passwords are stored in environment variables, not in code:

```bash
export PS_ACCOUNT_A="gen3ai_team_a"
export PS_PASSWORD_A="..."
export PS_ACCOUNT_B="gen3ai_team_b"
...
```

`PSClient` already supports authenticated login (`username` / `password` args). Pass
these from env vars in `play_ladder.py`.

### SOCKS5 Proxy

All ladder traffic routes through `proxy.g5d.io` (the existing SSH tunnel — see
`GCP_INFRASTRUCTURE.md`). Pass `--proxy socks5h://127.0.0.1:1080` to `play_ladder.py`.
This prevents the home IP from being exposed to Showdown and keeps the tunnel consistent
with the replay collector setup from v6 Step 1.

---

## Run Command

```bash
export PYTHONPATH=$PYTHONPATH:src

/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/play_ladder.py \
  --model models/v6_team_a/best.zip \
  --fixed-team data/team_eval/top3/team_a.txt \
  --account $PS_ACCOUNT_A \
  --password $PS_PASSWORD_A \
  --format gen3ou \
  --n-battles 100 \
  --proxy socks5h://127.0.0.1:1080 \
  --save-replays replays/ladder/team_a/ \
  --log-elo logs/team_a_elo.jsonl
```

Run all three in parallel (separate terminals or tmux panes). Each runs 100 battles and
exits. Review ELO logs before deciding whether to run another 100.

---

## ELO Tracking

After each battle, fetch the player's current ELO from the Showdown API and append to
the `--log-elo` file:

```json
{"battle": 1, "result": "win", "elo": 1203, "opponent": "SomePlayer", "turns": 34}
{"battle": 2, "result": "loss", "elo": 1189, "opponent": "AnotherPlayer", "turns": 22}
...
```

Plot `elo` vs. `battle` to assess trajectory. A rising ELO after 100 games means continue
playing. A stable or declining ELO after 100 games suggests a ceiling has been hit —
either the specialisation did not help or the meta-mismatch (see Future Work in
`todo.md`) is limiting further progress.

**Stopping criterion**: stop when ELO has been within ±30 points for 30 consecutive
games. This indicates the model has found its natural rating for that team.

---

## Replay Collection

All ladder replays are saved to `replays/ladder/team_{a,b,c}/` in the same `.log` format
as the v6 spectator daemon. These feed into:

1. **Team completion model updates** (v6 Step 3): ladder opponents run different teams
   from the league — more replays improve the completion model's prior over real
   ladder compositions.

2. **Future meta-alignment** (noted in `todo.md` Future Work): the loss replays in
   particular reveal which opponent strategies the agent struggles against. These are the
   inputs for a future ladder-weighted opponent pool.

---

## Selecting the Primary Entry

After 100 battles per team, compare ELO trajectories:

| Outcome | Decision |
|---------|----------|
| One team clearly higher ELO (≥ 50 points above the others) | That team is the primary entry; run 200 more games with it |
| Teams within 50 ELO of each other | Continue all three until one separates, or declare the highest as primary |
| All teams declining or stagnant below 1200 ELO | Meta-mismatch is likely — proceed to Future Work direction |

The primary entry is the team deployed for longer ladder runs and used as the reference
point for future meta-alignment iterations.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/main/play_ladder.py` | Ladder battle loop: auth login, fixed team, MCTS decisions, ELO logging, replay saving |

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/inference/mcts_player.py` | Ensure `fixed_team` flows through to the player's team selection (already wired from Step 2) |

---

## Verification

1. **Auth smoke test**: before running 100 games, run 1 game with `--n-battles 1` and
   confirm the account logs in, the team is accepted, and the battle completes. ELO logged
   correctly.

2. **Proxy check**: confirm the connection routes through `proxy.g5d.io` (check the Showdown
   user page — IP should show the GCP VM's static IP, not the home IP).

3. **Replay integrity**: after 10 games, verify that saved `.log` files parse cleanly
   through `SpectatedBattle` (the same path the v6 daemon already exercises).

---

## Final State

Step 3 is complete when all three teams have run 100+ ladder battles and ELOs have
stabilised. The primary entry team is identified. Replays are archived in
`replays/ladder/` for future meta-alignment work.

**This is the end of the v7 arc.** The agent is now on the ladder with a real ELO.
Future iterations (meta-alignment, ladder-weighted team selection) are documented in
`designs/ai_v7/todo.md` under Future Work.
