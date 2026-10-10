# The team downloaders — manifests, the dedupe, the encodings

Lifted out of `tools/CLAUDE.md` on 2026-10-10 (always-current). Covers
`sample_team_downloader/sync.py` (Smogon forum sample-team thread → `data/teams/sample/`) and
`others_team_downloader/sync.py` (PokePaste dumps → `data/teams/others/`); the per-tool usage is each
tool's own `README.md` where it has one.

## One manifest entry per team (`gen3_team_pool_dedupe`)

Each sync writes a per-folder `teams.json` manifest (`{id, name, format, valid, errors, file, source}`
rows) next to the `.txt` team files. **The invariant every manifest must hold: one entry per distinct
team (== per `id` == per `.txt` file).** The runtime `TeamLoader` (`src/utils/team_loader/`) appends a
team's text once per **entry**, then draws uniformly — so a duplicated entry silently multiplies that
team's training/eval draw weight.

This bit us once: the Yak Attack PokePaste dump names a team once **per Pokémon**
(`"<Mon>/<Team Name>"`, six headers sharing one 6-mon text → one id/file), and the generator didn't
dedupe, so `others/yak_attack/teams.json` had 1122 rows over 185 files (174 valid). That made
yak_attack ~66% of **every** training and eval team draw (pool 1601 over 719 unique). Fix:
`others_team_downloader/sync.py::collapse_duplicate_teams` collapses rows by `id` before writing
(name strips the `"<Mon>/"` prefix, `valid` = AND over the group, `errors` = union, idempotent), so a
re-run can't reproduce it. The deduped pool is **719 unique teams** — since `main.promote_teams`
moved teams into the curated set, **72 sample + 647 others** (it was 32 + 687 at the dedupe; the pin
is `src/utils/team_loader/loader_test.py`). `TeamLoader._load_teams` also dedupes by resolved file
path as defense-in-depth (loud warning if a manifest references a file twice). Guards:
`src/utils/team_loader/team_manifest_test.py` (data-contract: one-entry-per-file over all manifests +
collapse-fn units) and `loader_test.py` (synthetic per-mon dedupe + the count pin). A changed team
pool is a **data-distribution change** (training *and* eval) — land it at a clean retrain boundary,
never mid-A/B.

## The promotion writer (`main.promote_teams`)

`data/teams/sample/` has a SECOND writer: `python -m main.promote_teams` promotes a seed-recorded
random draw of already-downloaded pool teams into the curated set, and de-lists each from its source
`teams.json` so the pool total is unchanged. It is a *promotion* tool, not an acquisition tool — it
knows no upstream and downloads nothing — which is why it lives in `src/main/`. The interaction:
`sample_team_downloader/sync.py` rebuilds `data/teams/sample/teams.json` **from the forum thread
alone**, so a re-sync would silently drop every promoted entry. Check for
`data/teams/sample/PROMOTION_MANIFEST.json` before running it, and re-promote from that manifest's
recorded seed afterwards.

## Both downloaders NAME their encodings, and the committed data already carries a mojibake

`requests` falls back to **ISO-8859-1** for a `text/*` response with no `charset` (RFC 2616), so
`res.text` on a UTF-8 PokePaste turns `é` into `Ã©` — and the file is then written back as UTF-8,
baking it in. `data/teams/others/mcmegan/*.txt` hold the bytes of `PtÃ©ra` where `Ptéra` was meant,
and that nickname surfaced years later as a `KeyError: 'ptãra'` inside a depth-2 search replay,
where it was filed as a *chunk-transport double-encode* and chased in the wrong subsystem entirely
(the real defect was a missing ply — `gen3_search_depth2_chunk_gap_v1` in `designs/CHANGELOG.md`).
Both syncs now override an ISO-8859-1 guess with `apparent_encoding` and write with an explicit
`encoding="utf-8"` (a bare `open(..., 'w')` used the LOCALE encoding, so the same download was not
even reproducible across machines).

**The already-committed bytes are deliberately NOT rewritten.** A team file is hashed into
`pin_sha` (`MatchupSpec`) and keys `data/teams/gen3_team_archetypes.json`, and `others_team_downloader`
keys `teams.json` by `sha256(team_text)[:16]` (the sample downloader keys by PokePaste id) — so "fixing" a nickname re-ids the team, orphans its archetype
label and breaks every provenance record that named it. A nickname is cosmetic to the model (the
obs never reads one), so the correction belongs at the next deliberate pool boundary, not as a
tidy-up. `src/utils/team_loader/committed_team_bytes_test.py` pins the current bytes so a re-sync
that changes them is a visible event rather than a silent re-id. (The pin was the pure file-I/O half
of the depth-2 search battery's integration test, deleted with `main.search_dividend` in P6 slice
6d-1, 2026-10-08; it moved here.)
