// showdown_pack.js — gen3_peer_showdown_pack_v1: a team file packed the way a REAL Showdown server plays it.
//
// The server (`server/team-validator-async.ts`) imports the team, runs the validator (which writes a named Hidden
// Power's IVs and sets hpType) and plays `Teams.pack` of the VALIDATED sets (`server/ladders.ts`:
// `team: valResult.slice(1)`). Showdown's import also reads an explicit `Happiness: 0` and the "Frustration =>
// happiness 0" rule. The Metamon peer used upstream poke-env's packer, which does none of that (a mis-ordered Hidden
// Power table, `Happiness: 0` packed as the default 255, a `Hidden Power: <Type>` line ignored), so it packs through
// this script instead (`main/anchors/peer_scripts/team_pack.py`).
//
// stdin:  {"format": "gen3ou", "rows": [{"paste": "...", "legacy": "<packed or null>"}, ...]}
// stdout: [{"packed": "<packed>" | null, "errors": [...], "legacy_diffs": [...]}, ...]
//   `errors`: the validator's problems (non-empty => `packed` is null).
//   `legacy_diffs`: how the set the LEGACY pack played (its raw unpack: our front end plays the pack as sent) differs
//   from the set this pack plays, field by field over pack_norm.js's fields plus gender (as the sim resolves it) and
//   nickname (edge whitespace trimmed). A null legacy (the old packer could not pack the file) is one diff.
'use strict';

const path = require('path');
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Teams, TeamValidator, Dex, toID } = require(psPath);
const { norm } = require('./pack_norm');

function played(dex, s) {
    return Object.assign(norm(dex, toID, s), {
        // As the sim resolves it (`sim/pokemon.ts`): a set's M / F, else the species' fixed gender, else '' (drawn).
        gender: s.gender || dex.species.get(s.species || s.name).gender || '',
        // The nickname shows in every protocol ident; edge whitespace (a paste's trailing spaces, which the old packer
        // kept) changes no mechanic and is not a difference.
        name: String(s.name || dex.species.get(s.species).name).trim(),
    });
}

function legacyDiffs(dex, legacy, packed) {
    if (legacy === null || legacy === undefined) return ['the legacy packer could not pack this file'];
    const a = Teams.unpack(legacy), b = Teams.unpack(packed);
    if (!a) return ['the legacy pack does not unpack'];
    if (a.length !== b.length) return [`legacy ${a.length} mons, this pack ${b.length}`];
    const out = [];
    for (let i = 0; i < a.length; i++) {
        const x = played(dex, a[i]), y = played(dex, b[i]);
        for (const k of Object.keys(y)) {
            if (String(x[k]) !== String(y[k])) out.push(`mon ${i + 1} (${y.species}) ${k}: legacy ${x[k]}, now ${y[k]}`);
        }
    }
    return out;
}

let input = '';
process.stdin.on('data', c => { input += c; });
process.stdin.on('end', () => {
    const req = JSON.parse(input);
    const format = req.format || 'gen3ou';
    const validator = new TeamValidator(format);
    const dex = Dex.forFormat(format);
    const out = [];
    for (const row of req.rows) {
        try {
            const team = Teams.import(row.paste);
            if (!team || !team.length) {
                out.push({ packed: null, errors: ['import failed'], legacy_diffs: [] });
                continue;
            }
            const errors = validator.validateTeam(team) || [];
            if (errors.length) {
                out.push({ packed: null, errors, legacy_diffs: [] });
                continue;
            }
            const packed = Teams.pack(team);
            out.push({ packed, errors: [], legacy_diffs: 'legacy' in row ? legacyDiffs(dex, row.legacy, packed) : [] });
        } catch (e) {
            out.push({ packed: null, errors: [`showdown_pack error: ${e.message}`], legacy_diffs: [] });
        }
    }
    console.log(JSON.stringify(out));
});
