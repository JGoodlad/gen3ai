// pack_oracle.js — gen3_team_pack_oracle_v1 (2026-10-26 GIGO audit C-2 / C-3).
//
// Is OUR packed team the team Showdown would play? For each {paste, packed} row: Showdown's own reading of the
// PASTE (Teams.import, then the validator's mutations — it fills a Hidden Power type's IVs, it sets hpType) against
// Showdown's unpack of OUR packed string, field by field over everything the engine reads. Our packer dropped an
// explicit `Happiness: 0` and Showdown's "Frustration => happiness 0" import rule (Frustration at base power 1), and
// ignored a `Hidden Power: <Type>` line on a mon without the move (IVs left at 31). Validating the PACK cannot see
// either: every such pack is a legal team, just not the authored one.
//
// stdin: {"format": "gen3ou", "rows": [{"paste": "...", "packed": "..."}, ...]}
// stdout: one list of defect strings per row (empty = identical); a row whose paste the validator refuses is
// reported as such (the caller drops those before packing, so one reaching here is itself a defect).
const path = require('path');
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Teams, TeamValidator, Dex, toID } = require(psPath);

const { norm: normWith } = require('./pack_norm');
const norm = (dex, s) => normWith(dex, toID, s);

let input = '';
process.stdin.on('data', c => { input += c; });
process.stdin.on('end', () => {
    const req = JSON.parse(input);
    const format = req.format || 'gen3ou';
    const validator = new TeamValidator(format);
    const dex = Dex.forFormat(format);
    const out = [];
    for (const row of req.rows) {
        const defects = [];
        try {
            const authored = Teams.import(row.paste);
            const errs = authored ? validator.validateTeam(authored) : ['import failed'];
            const ours = Teams.unpack(row.packed);
            if (errs && errs.length) {
                defects.push(`paste refused by the validator: ${errs[0]}`);
            } else if (!ours) {
                defects.push('our pack does not unpack');
            } else if (ours.length !== authored.length) {
                defects.push(`${ours.length} mons packed, ${authored.length} authored`);
            } else {
                for (let i = 0; i < authored.length; i++) {
                    const a = norm(dex, authored[i]), b = norm(dex, ours[i]);
                    for (const k of Object.keys(a)) {
                        if (String(a[k]) !== String(b[k])) {
                            defects.push(`mon ${i + 1} (${a.species}) ${k}: Showdown plays ${a[k]}, our pack ${b[k]}`);
                        }
                    }
                }
            }
        } catch (e) {
            defects.push(`oracle error: ${e.message}`);
        }
        out.push(defects);
    }
    console.log(JSON.stringify(out));
});
