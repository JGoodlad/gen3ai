"""The committed pool team files keep their bytes — including the one mojibake nickname.

Moved here from `main/search_dividend/depth2_replay_integration_test.py` when P6 slice 6d-1 (2026-10-08)
deleted that battery; the ONE pure file-I/O test it held survives, because the property it pins has no other
guard. (The test's other half — the depth-2 chunk-gap regression, `gen3_search_depth2_chunk_gap_v1` — played a
poke-env battle through `SearchEngine` and went with the engine.)

`KeyError: 'ptãra'` was once filed as a chunk-transport double-encode. It is not one: the committed team file
holds ``c3 83 c2 a9`` — the UTF-8 encoding of ``Ã©``, i.e. ``é`` already mangled once before it was written
(the downloader's `requests` fell back to ISO-8859-1 on a `text/*` response with no charset) — so the nickname is
mojibake AT REST and every layer above merely carries it faithfully. Pinned so the next reader of that error does
not go looking for an encoder again, and so a future re-sync of ``data/teams/`` that silently CHANGES these bytes
is a visible event: a team file is hashed into ``pin_sha`` (`MatchupSpec`) and keys
``data/teams/gen3_team_archetypes.json``, so rewriting one re-ids the team and orphans its archetype label (see
`designs/tools/team_downloaders.md`, "Both downloaders NAME their encodings").
"""
from __future__ import annotations

from utils.paths import repo_path
from utils.showdown_id import to_id_str

#: A committed pool team whose mons carry NICKNAMES (five of them).
_NICKNAMED_TEAM = repo_path("data", "teams", "others", "mcmegan", "7fda48d98ca8efdc.txt")


def test_the_reported_mojibake_is_in_the_TEAM_FILE_not_the_transport():
    raw = _NICKNAMED_TEAM.read_bytes()
    assert b"Pt\xc3\x83\xc2\xa9ra" in raw, (
        "the mcmegan fixture team no longer carries the double-encoded nickname the depth-2 "
        "KeyError was reported against")
    nick = "PtÃ©ra"
    assert to_id_str(nick) == "ptãra", "the reported key is to_id_str of the mojibake"
    assert nick.encode("latin-1").decode("utf-8") == "Ptéra", (
        "the intended nickname is recoverable by undoing exactly one utf-8/latin-1 pass")
