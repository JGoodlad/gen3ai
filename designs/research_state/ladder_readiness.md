# Ladder readiness — what it takes to put OUR model on the public gen3ou ladder

**Audited 2026-08-23.** The owner's permanent constraint is "we must be able to play online
on the ladder in the future". Nobody had ever laddered this project's model, so nothing in
that path had ever been exercised. This is the honest gap list: what WORKS, what was FIXED
in this pass, and what is SIZED because it is a day or more.

The external milestone is Metamon's published gen3ou result — **~Elo 1511 / GXE 64 / 83rd
percentile** (the GXE→Elo conversion and its caveats are in
[`metamon_replay_feasibility.md`](metamon_replay_feasibility.md) § *GXE 64 in ladder-Elo
terms*; the 1511 is a median over a 1490–1560 band, so treat it as a target region). This
document is about the PLUMBING to get a rated game at all; it says nothing about whether we
would win it.

---

## Verdict

**~2–3 days of work from a first rated game, and none of it is architectural.** The model
plays a real websocket ladder game today. The three genuine gaps are an account, a
reconnect loop, and a session runner. There is no protocol wall and no latency wall — both
were measured, not assumed.

The single biggest gap is **no reconnect** (poke-env has none, at any layer), and the
ladder's disconnection timer is 60 s. Everything else is smaller than it looked going in.

---

## The measurements

Two things this audit set out to find turned out to be non-problems, and both had been
argued about from first principles before anyone measured them.

### Protocol drift: ZERO, measured over a real ladder corpus

`agents.battle.battle_event.classify` raises `UnknownMessageType` on any keyword it does not
know — a deliberate tripwire (`src/agents/battle/battle_event.py:616`). That is correct for
a closed local sim and *potentially fatal* on the open public server: the raise kills the
parse task, no choice is ever sent, and the battle is lost on the timer. Our
`deps/pokemon-showdown` is pinned at a 2026-05-09 commit; the public server runs master.

So it was measured. `src/main/ladder_drift_scan.py` (new, this pass) pulls real gen3ou games
from the public replay archive — read-only HTTP, no account, no websocket — and runs each one
through a real `Gen3Battle`, both as a keyword census and as a full structural parse.

| | result |
|---|---|
| replays scanned | **59** (one day's public gen3ou uploads) |
| protocol lines | **20 589** |
| distinct keywords | **53** |
| unclassified (would raise `UnknownMessageType`) | **0** |
| non-gen3 mechanics (would raise `UnsupportedMessageType`) | **0** |
| structurally clean through `Gen3Battle.parse_message` | **59 / 59** |

The census includes every line a rated game adds and a local sim does not: `inactive` (379 —
the ladder timer is *busy*), `rated` (42), `raw` (80 — the rating-change block), `c` (193 —
chat), `j`/`l`/`n` (spectators). All classified, all CONTROL/COSMETIC, none reaching the obs.

⚠️ **This is a reading of one day's ladder, not a proof about every future one.** Re-run the
scan before going live; that is what it is for.

### Latency: ~3 500× of margin

The ladder timer (`deps/pokemon-showdown/server/room-battle.ts:41–52`): **150 s starting bank
+ 60 s grace, +10 s per turn, capped at 150 s per turn**. (A *challenge* gets 300/300 — the
ladder is the tighter one, which is the one measured against.)

Measured on our own end-to-end websocket path, with a live training run on the box (load
~16–22 on 16 cores — i.e. the pessimistic reading):

| | ms |
|---|---|
| mean per decision | **18.2** |
| median | 18.2 |
| p95 | 23.7 |
| max | 43.5 |
| first decision (includes torch warm-up) | 25.2 |

Checkpoint load is **1.3 s** and happens before the socket opens, so it is not on any clock.

Two reasons to read these as an **upper bound**: the box was carrying a training run, and the
measurement predates gap #12 — the checkpoint's `ObservationDebugger` was still printing a
full 12-mon board on every forward while these were taken. The real ladder number is lower.
Either way there is no path where we time out on THINKING; the only timer exposure is a
network stall, which is the reconnect gap below.

### Game length vs our 250-turn self-forfeit

`RLPlayer._handle_stall` forfeits at `StallConfig.threshold = MAX_TURNS = 250`
(`src/agents/training/stall.py:16`), which on the ladder would be a real loss. Measured over
the same 59-replay corpus: **median 24 turns, p90 44, max 109, and 0/59 over 250.** Not a
practical risk. Left alone — and deliberately so, since `MAX_TURNS` is also the obs clock's
normaliser (`gen3_deadline_clock_v1`), so moving it for the ladder would put the critic's
deadline scalars out of distribution. Revisit only if a stall-vs-stall matchup is ever
observed running long.

---

## Gap list

### WORKS (verified this pass, no change needed)

| # | Item | Evidence |
|---|---|---|
| 1 | **The vendored fork has the whole ladder machinery.** `Player.ladder(n)` → `PSClient.search_ladder_game` → `/utm <packed>` + `/search <format>` (`src/poke_env/player/player.py:545`, `src/poke_env/ps_client/ps_client.py`). `accept_challenges` / `send_challenges` / `battle_against` all present. | Ran a real 5-game `/search gen3ou` session against a local server: **5/5 matched, played and won.** |
| 2 | **Registered login is implemented.** `PSClient.log_in` POSTs `act=login` with name/pass/challstr to `action.php` and `/trn`s the assertion. Guest path (no password) also works. Avatar via `/avatar`. Proxy (SOCKS5) threaded through both the websocket and the auth POST. | `src/poke_env/ps_client/ps_client.py`; the proxy path is already used by `collect_replays.py` against the public server. |
| 3 | **`wss://` to the official server is configured.** `ShowdownServerConfiguration` = `wss://sim3.psim.us/showdown/websocket` + `https://play.pokemonshowdown.com/action.php?` — both current. | `src/poke_env/ps_client/server_configuration.py:28` |
| 4 | **Team submission is correct and every pool team is legal.** `Gen3Teambuilder` packs via poke-env's `join_team` and validates each team against the real Showdown validator in `deps/pokemon-showdown` before packing (`src/utils/teambuilder.py:36`, `src/utils/bridge/team_validator.py`). | **719/719 pool teams validate for `gen3ou` and pack.** |
| 5 | **Rated-ladder protocol parses.** A rated game adds `\|rated\|`, a 6-field `\|player\|p1\|name\|avatar\|RATING`, `\|raw\|` rating-change HTML and `\|askreg\|`. | Ran a real *rated* local-ladder game and read the frames back: `['', 'player', 'p1', 'RLPlayer 1', '266', '1080']`. poke-env's `player` branch has the 6-field arm (`abstract_battle.py:1343`). |
| 6 | **Observation parity with training.** The websocket path (`--use-bridge off`) is the identical poke-env → `Gen3Battle` → encoder stack the trainer uses; every ladder-specific line is CONTROL or COSMETIC and is dropped before the event log. `StrictBattleView` reads the same read-models. | Item 5 + the drift scan: no ladder-only keyword is `Policy.EVENT`. |

### FIXED HERE

| # | Item | What was wrong |
|---|---|---|
| 7 | **`play.py` is now a real ladder entry point.** `--mode {selfplay,ladder,accept,challenge}`, `--server {local,official}`, `--model`, `--username/--password` (env-defaulted), `--avatar`, `--team`/`--team-pool`, `--temperature`, `--proxy`, `--concurrency`. | It was two `RandomPlayer`s on hardcoded localhost with no model, no auth and no ladder mode — the entry point the whole constraint depends on could not do the thing. Design intent existed since `designs/ai_v7/impl_step3_ladder.md`; `play_ladder.py` was never built. |
| 8 | **Reserved-port refusal, in CODE.** `--port 8000/8001` exits with a message instead of connecting. | The root `CLAUDE.md` rule ("never touch :8001") was a docs warning only. A laddering process on :8001 drops every poke-env websocket on the live run at once. Now `main.play.RESERVED_PORTS`, gated by `src/main/play_test.py`. |
| 9 | **The guest-rename race that hung `battle_against`.** `PSClient` set `logged_in` on the server's opening `\|updateuser\| Guest N` greeting — which arrives BEFORE `\|challstr\|` — for any passwordless client. `send_challenges` waits on exactly that event, so the challenge went to a name that did not exist yet, the server replied `\|popup\|The user '…' was not found`, and **both sides hung forever.** | It is a RACE, and it lost almost every time: `play.py --mode selfplay` hung on 2/2 plain runs (one instrumented run won the race, which is exactly how a bug like this gets written off as a flake); 3/3 clean after. Training never saw it at all because every eval / self-play account carries a password. Now gated on `_trn_sent`. |
| 10 | **A refused login presented as a HANG, not an error.** `json.loads(text[1:])["assertion"]` raised `KeyError` on a wrong password, `JSONDecodeError` on a rate-limit HTML page, and *silently accepted* a `;`-prefixed soft refusal — all inside a fire-and-forget task, so `logged_in` was never set and every `await logged_in.wait()` blocked forever. | New `poke_env.exceptions.LoginError` + `_parse_login_assertion`; the client closes its own socket on refusal so `Gen3Player._await_connected` turns it into a named `ShowdownConnectionError` instead of a stall. This is the failure mode a first ladder attempt is MOST likely to hit. |
| 11 | **Room-layer keywords a live server can emit and a local sim cannot.** `c:` (timestamped chat), `noinit`, `popup`, `notify`, `tempnotify`, `tempnotifyoff`, `uhtmlchange` were in NEITHER `MESSAGE_POLICY` nor poke-env's `MESSAGES_TO_IGNORE` — each one a `UnknownMessageType` that wedges the battle. | Classified as COSMETIC in both places. **Honest note:** none of these was reproduced on a battle room. Battle rooms set `noLogTimes = true` (`server/rooms.ts:1928`), so ladder chat is the plain `\|c\|` we already ignored — measured, after the opposite was assumed. These are defensive; the drift scan is the evidence that nothing reachable is missing. |
| 12 | **The `ObservationDebugger` no longer spams a ladder run.** A checkpoint trained with `--log-level periodic` carries a live debugger that `print()`s a full 12-mon board on every forward. `play.py` silences it (as the prober already did); `--debug-obs` keeps it. | Observed in the smoke: megabytes of stdout per battle. |
| 13 | **A protocol-drift pre-flight gate exists.** `src/main/ladder_drift_scan.py`, exit 0/1, re-runnable, `--offline` for a cached corpus. | It is the only way to check drift against the LIVE server without an account, and it is the check to run before every ladder session. |

### SIZED (≥ a day — not built)

| # | Item | Size | Why it matters |
|---|---|---|---|
| 14 | **Reconnect.** `PSClient` has *no* reconnect at any layer: an abnormal close sets `_disconnected` and every consumer raises (`ps_client.py:327–339`). The battle layer merely *tracks* `\|inactive\|disconnected` and never acts. The ladder's disconnection timer is **60 s** (`DISCONNECTION_TIME`, and the ladder uses the non-bank variant), so one dropped socket = one forfeited rated game. | **1–2 days** | THE biggest gap. Needs: reconnect, re-`/join` the battle room, re-issue `\|request\|` (the server re-sends on rejoin), and resume the decision loop without double-choosing. The spectator client already has a reconnect loop (`spectator_client.py:133–137`) — a shape to copy, not code to reuse. |
| 15 | **A ladder session runner.** `play.py --mode ladder N` plays N games and exits. A campaign wants: crash restart, a per-game JSONL result log, rating readback, a stop condition, and a "don't re-queue after a refusal" rule. | **1 day** | Without it, a 300-game ladder run is babysat by a human. Model it on `main.launcher` (restart loop + result log), not on a bash `while`. |
| 16 | **Rating readback.** No code can read our own gen3ou Elo/GXE. The websocket route is a dead end (`/cmd laddertop` returns `null` on the main server); the working route is plain HTTP — `pokemonshowdown.com/users/<id>.json` or `play.pokemonshowdown.com/api/ladderget?user=<id>` (strip the leading `]`, send a real User-Agent). See the bot-policy section. | **0.5 day** (mostly de-risked — it is one HTTP GET + JSON parse, not a protocol subscription) | Without it we cannot report the milestone number, which is the entire point. Report **GXE with W-L and rprd**, not Elo (Elo decays while idle; GXE does not). |
| 17 | **Non-blocking auth.** `requests.post` to `action.php` is a **synchronous** call on POKE_LOOP with a 10 s timeout — it blocks the whole event loop, including every other client's battle. | **0.5 day** | Harmless for a single ladder client (once, at startup). It becomes real the moment two accounts ladder in one process. Left alone deliberately; noted so nobody re-derives it as a mystery stall. |
| 18 | **Outbound message throttling.** `PSClient.send_message` writes straight to the socket; `_sending_lock` is declared (`ps_client.py:162`) and never used. Showdown throttles per connection server-side. | **0.5 day** | Only bites on `--concurrency > 1` or a reconnect storm. `--concurrency 1` (the default) is the mitigation. |

---

## Showdown's rules on bots

**Short answer: bots are permitted on the ladder, conditionally, and the condition is
behavioural rather than technical.** There is no written "bot policy"; there is an
administrator's statement of enforcement, and the rules that DO exist are about conduct.

**The rules page** ([pokemonshowdown.com/pages/rules](https://pokemonshowdown.com/pages/rules))
has five headings — *be nice to people · follow US laws · no sex · no cheating · moderators
have discretion* — and **says nothing about bots, scripts or automation at all.** Its
"no cheating" clause is about a different thing: *"Don't exploit bugs to gain an unfair
advantage. Don't game the system (by intentionally losing against yourself or a friend in a
ladder match, by tricking your opponent into forfeiting, etc)."* Playing a strong policy
honestly is not on that list. The username rules DO bind us: *"Names may not impersonate"* —
so the account must not read as a human player of that name.

**The actual enforcement standard** is a staff statement, conditional not prohibitive.
**Maia** (Battle Simulator Admin, speaking "on behalf of the PS Admin team", Aug 2022, in the
thread about a bot playing ~223 games/day on the NU ladder —
[Remove Bots from Ladder during Suspect Tests](https://www.smogon.com/forums/threads/remove-bots-from-ladder-during-suspect-tests.3706925/),
marked *Implemented*):

> "our official policy stance is as follows: **If a bot is negatively affecting human
> experience, then we will remove it from the ladder via a permaban.** While it is impossible
> to completely ban bots due to the difficulty tracking them across the sim, if people feel
> that a bot is negatively impacting the integrity of a suspect test, then we can and will do
> the same during the suspect."

Reaffirmed by **Hecate** (Battle Simulator Administrator, Dec 2025,
[Ladder Bots and Usage-Based Tiering](https://www.smogon.com/forums/threads/ladder-bots-and-usage-based-tiering.3774656/)):
*"Banning bots is ultimately not feasible in any capacity that would solve this problem."*
That thread closed (Apr 2026) with **rate limiters, not bans** — what the limiters are is
unverified, so do not assume the 2022 posture is permanent. The repo itself links an official
**Bot FAQ** ("making Pokemon Showdown bots — mainly chatbots and battle bots") from its
README, and `PROTOCOL.md` documents the public endpoint. The operative risk is not "will we
be caught", it is "are we a nuisance" — volume, timing, and whether we are sitting on a
ladder people are trying to qualify on. No ladder-vs-challenge distinction is written
anywhere; every recorded complaint concerns ladder VOLUME.

**Precedent — including peer-reviewed work on gen3ou specifically.** All laddered under
ordinary human-looking names; none banned for botting:

| project | laddered? | result |
|---|---|---|
| **Metamon** (UT Austin, RLC 2025) | yes — public ladders, ≥400 battles/gen over 4–8 days, accounts like `SmallSparks`, `TheDeadlyTriad` | two top-300 **gen3ou** appearances; accounts intact |
| **pmariglia/foul-play** | yes — `--bot-mode search_ladder` vs `wss://sim3.psim.us` is a documented flag | 1930+ gen9ou; reported #4 gen3ou |
| **Future Sight AI** | yes | top ~5% gen8ou; one *lock* ("all it does is battle and say 'gg'") — most probably the VPS auto-lock below, account intact today |

No documented ban *for botting* was found across policy threads 2022–2026. One practitioner
caveat cuts the other way (pmariglia): *"it is certainly exploitable if the opponent knows it
is playing a bot"* — a name that declares the bot trades a little strength for standing.

**🚨 The biggest practical constraint is not a rule — it is the datacenter-IP auto-lock.**
`server/punishments.ts:1760`: a host classified `'proxy'` ("datacenters, VPNs, proxy
services") auto-locks any non-trusted user (`#hostfilter`). The official Bot FAQ says exactly
this happens to popular VPS hosts and the remedy is asking a Rooms Operator to mark the bot
trusted. A lock blocks CHAT, not battling — but a laddering account that cannot speak (even
to say "gg") is a worse look than a home IP. **This inverts the earlier proxy advice: do NOT
route the ladder session through the GCP tunnel** — its egress is a datacenter IP. Ladder
from the residential connection, or pre-arrange trusted status first.

**Registration: NOT required for rated play — verified in source.** `prepBattle`
(`server/ladders.ts:311–327`) has no `user.registered` gate, and `room-battle.ts:877–881`
handles an *unregistered* winner of a *rated* game (it just sends the `|askreg|` nag we saw
in our own smoke). The one conditional gate is `Config.forceregisterelo`
(`server/chat-commands/core.ts:1466`) — refuse `/search` above a threshold — which defaults
to `false` and whose production value is private/unverified. Register anyway: a guest name
can be taken by anyone, and our login flow assumes a password. No minimum account age gates
laddering ("autoconfirmed" = registered ≥1 week + 1 rated win gates chat, not search).
⚠️ Our client only *logs* the forceregisterelo `|popup|` — it would silently stop laddering;
worth a handler when #15 is built.

**Rate limits — actual constants from server source** (defaults the main server is believed
to run; skipped under `--no-security` locally):

| limit | constant | where |
|---|---|---|
| battles + team validations | **12 per 3 min per IP** | `Monitor.countPrepBattle`, `server/monitor.ts:229` |
| concurrent battles | **5 per user** | `monitor.ts:242` |
| any client→server message (incl. `/choose`) | **600 ms** apart (25 ms with the staff-granted `*` bot rank); 6-deep queue, overflow DROPPED | `THROTTLE_DELAY`, `server/users.ts:33`, enforced `users.ts:1429` |
| connections | 500 per 30 min per IP → auto "cflood" ban | `monitor.ts:186` |
| challenges | 10 s apart | `ladders.ts:176` |

The binding ceiling at `--concurrency 1` is the 600 ms message throttle (~1.7 decisions/s —
still 30× our 18 ms think time, irrelevant in practice), and the 12-per-3-min prep cap bounds
any campaign at ≤240 games/hour. Matchmaking also never pairs two users on the same IP and
never re-matches the immediately previous opponent (`Ladder.matchmakingOK`,
`ladders.ts:332–371`) — so two of OUR accounts on one IP cannot farm each other even by
accident.

**Ratings, decay, and what to report.** Elo starts/floors at 1000. Decay (Elo only, daily
09:00 UTC, above 1400): >5 games → none; 1–5 → `(elo−1400)/100`; 0 → `1+(elo−1400)/50` — and
**non-current-gen formats like gen3ou subtract 2 from each day's decay**, so it only bites
meaningfully above ~1500–1600 (right at the milestone band; a parked 1511 bleeds slowly).
GXE is a pure function of Glicko (`gxe = 100/(1+10^((1500−rpr)/400/√(1+0.0000100724·(rprd²+130²))))`,
from the live login-server source, reproduced against all 500 live gen3ou top-500 rows) and
**barely moves while idle** — which is why **GXE (with W-L and rprd) is the headline number
for a fixed-budget run, not Elo**. Appearing on the ladder page needs `rprd ≤ 100` (~6 games
from a fresh start); a *converged* rating is much further out (≈40 games to RD 50, ≈180 to
the RD-25 floor — pyuk, loginserver contributor). Current gen3ou top-500 entry bar: Elo
≈1548. gen3ou has never been reset in ≥8 years (a 2018 row is still served), so a rating,
once earned, persists.

**Reading our rating programmatically** (this largely de-risks sized item #16): there is
**no websocket path** — `/cmd laddertop` returns `null` on the main server. Use HTTP:
`https://pokemonshowdown.com/users/<userid>.json` (official, documented in the client repo's
WEB-API.md; **send a real User-Agent** — default python-urllib gets a Cloudflare 403, which
our drift scan already learned the hard way) or
`https://play.pokemonshowdown.com/api/ladderget?user=<id>` (richest fields; strip the leading
`]` byte — same quirk as the login response). The post-game `|raw|<user>'s rating: 1099 →
<strong>1116</strong>` line is Elo-only and possibly stale — use it as a trigger, then re-read
`ladderget` for truth.

**Therefore, the etiquette we adopt** (none required by a written rule; all cheap insurance
against the discretionary standard above):

1. A name that cannot be mistaken for a human — declare the bot in the name itself.
2. Do not ladder during a gen3ou suspect test.
3. `--concurrency 1`, one game at a time.
4. Never forfeit-spam or requeue in a tight loop after a refusal.
5. Do not chat. (We cannot anyway — nothing in our client sends chat.)
6. Stop if asked.
7. Ladder from a residential IP, not the datacenter tunnel.

⚠️ **Not verified:** the main server's production `Config` values (`forceregisterelo`, any
custom throttles); what the April 2026 anti-bot "limiters" are; why GXE's reference RD is 130
(vs X-Act's published 350); whether any AI account has ever been actioned for anything other
than volume.

---

## What the smoke actually did, and what broke

Run against a throwaway local server on **:9017** (`npm run showdown -- 9017`), never 8000 or
8001, with `models/ai_v9_26_baitent_probe_0823/legB_final_model.zip` read-only (its
`arch_signature` is `gen3_critic_route_wave_v1`, i.e. current HEAD, so it loads).

| smoke | result |
|---|---|
| `RLPlayer` vs `Gen3HeuristicV2Player` over a real websocket, 1 and 3 battles | **PASS** — 3/3 finished, model won 3/3, 85 decisions at 18.2 ms mean |
| the real `/search gen3ou` ladder path, model vs a laddering `RandomPlayer`, 2 and 5 games | **PASS** — 5/5 matched and completed, won 5/5 |
| rated-frame capture on a local rated ladder game | **PASS** — `\|rated\|`, 6-field `\|player\|` with ratings 1080/1000, `\|raw\|`, `\|askreg\|` all parsed |
| `play.py --mode selfplay` (the default path) | **BROKE — 2/2 runs hung forever.** Root cause: gap #9, the guest-rename race. Fixed; 3/3 after. |
| chat injected into a live battle room | **Inconclusive by design** — the local server refused it ("You must be registered to chat in temporary rooms"). That refusal itself arrived as `\|html\|` in the battle room and parsed clean. The chat question was then settled from the server source + the 193 real `\|c\|` lines in the replay corpus. |
| 60 real public-ladder replays through `Gen3Battle` | **PASS** — 59/59 clean, 0 unknown keywords |

Nothing else broke. The websocket path is in better shape than the absence of any prior
ladder attempt would suggest — because it is the same stack `--use-bridge off` eval has
always used; only the *entry* to it was missing.

---

## Go-live checklist

Every step from here to the first rated game.

**Before writing any more code**

1. `python src/main/ladder_drift_scan.py --n 200` — must exit 0. Re-run on the day.
2. `git submodule update --remote deps/pokemon-showdown` in a scratch worktree, then
   re-run `validate_teams_locally("gen3ou", …)` over all 719 teams against the UPDATED
   validator. Our pin is 2026-05-09; a banlist change since then would fail team submission
   at `/utm` with an unhelpful error. (719/719 pass on the current pin.)

**Account**

3. Register a Showdown account. Not strictly required for rated play (verified in source —
   see the bot-policy section), but a guest name can be taken by anyone, our login flow
   assumes a password, and `Config.forceregisterelo` may gate `/search` above a threshold —
   a refusal our client currently only logs.
4. Name it so it is obviously a bot, set an avatar, and put the project in the profile — see
   the bot-policy section.
5. Put the password in a file / env var (`$PS_PASSWORD`), never on the command line — this box
   shares a process list with a training run.

**Build (the SIZED items, in this order)**

6. Reconnect (#14). Do this first; without it every other number is measured on a lucky
   network.
7. Rating readback (#16) — nothing is reportable without it.
8. Session runner (#15).

**Dry runs**

9. `python src/main/play.py --mode challenge --server official --opponent <your own alt> …` —
   ONE unrated challenge game against yourself before ever touching `/search`. This exercises
   real auth, real `wss://`, real protocol, and rates nothing.
10. Re-read the frames from that game and diff the keyword census against the drift scan's.
11. `--mode ladder --n-battles 1` on the real ladder. Read the `|raw|` rating block.

**Campaign**

12. **Do NOT route through the GCP SOCKS5 tunnel.** Its egress is a datacenter IP, and
    Showdown auto-locks users on hosts classified `'proxy'` ("datacenters, VPNs, proxy
    services" — `server/punishments.ts:1760`); the official Bot FAQ names popular VPS hosts
    as exactly the ones this hits. Ladder from the residential connection, or pre-arrange
    trusted status via a Rooms Operator first. (This reverses the ai_v7 step-3 plan's proxy
    advice, which predates finding the host filter.)
13. `--concurrency 1`. One game at a time — the polite setting, and it keeps us far under
    the 600 ms message throttle and the 12-battles-per-3-min IP cap.
14. `--temperature 0` (greedy) is the measurement; consider `>0` only if a repeat opponent
    starts exploiting a deterministic line. (Matchmaking never re-pairs the immediately
    previous opponent, which blunts single-game exploitation.)
15. Play to a **converged** rating: ~6 games merely puts the account on the board
    (`rprd ≤ 100`); ≈40 games reaches RD 50 and ≈180 the RD-25 floor. Do not quote a rating
    from 20 games; report **GXE + W-L + rprd** as the headline, Elo alongside.
16. Report GXE alongside Elo — the Metamon comparison is stated in both.

**Never**

- `--port 8000` / `--port 8001` (refused in code, #8).
- `--device cuda` while a training run holds the GPU. `play.py` defaults to `cpu`; the model
  needs 18 ms a decision on CPU and the ladder gives it 150 000.

---

## Files touched in this pass

| file | change |
|---|---|
| `src/main/play.py` | rewritten as the ladder entry point (#7, #8, #12) |
| `src/main/ladder_drift_scan.py` | new — the protocol-drift pre-flight gate (#13) |
| `src/main/play_test.py` | new — port-refusal + guest-refusal guards |
| `src/poke_env/ps_client/ps_client.py` | guest-rename gate (#9), `_parse_login_assertion` + fail-loud refusal (#10) |
| `src/poke_env/ps_client/login_test.py` | new — both of the above, verified failing on revert |
| `src/poke_env/exceptions.py` | `LoginError` |
| `src/poke_env/battle/abstract_battle.py` | room-layer keywords in `MESSAGES_TO_IGNORE` (#11) |
| `src/agents/battle/battle_event.py` | the same keywords in `MESSAGE_POLICY` (#11) |
| `src/agents/battle/battle_event_test.py` | live-room chrome coverage, both registries |
| `src/agents/inference/player.py` | `_await_connected` names a refused login instead of blaming the network |
