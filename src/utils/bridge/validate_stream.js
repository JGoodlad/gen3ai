// validate_stream.js
// A PERSISTENT team validator over the pinned deps/pokemon-showdown (the world builder's legality check, F2 of
// designs/endstate/design_world_sampler.md): one JSON request per stdin line, one JSON reply per stdout line, so a
// caller validating many small teams pays Node's start-up and the dex load ONCE (validate_team.js is one-shot).
//
//   request:  {"format": "gen3ou", "teams": ["<packed team>", ...]}
//   reply:    [{"valid": true|false, "errors": [...]}, ...]          (one per team, in order; a team whose import
//             or validation RAISED is {"valid": null, "raised": "<message>"}: the caller raises, never a verdict)
//             {"fatal": "<message>"}                                  (the whole request failed: never a verdict)
const path = require('path');
const readline = require('readline');

const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Teams, TeamValidator } = require(psPath);

const validators = new Map();
function validatorFor(formatId) {
    if (!validators.has(formatId)) validators.set(formatId, new TeamValidator(formatId));
    return validators.get(formatId);
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', line => {
    if (!line.trim()) return;
    let out;
    try {
        const req = JSON.parse(line);
        const validator = validatorFor(req.format || 'gen3ou');
        out = req.teams.map(text => {
            try {
                const team = Teams.import(text);
                if (!team || !team.length) return { valid: null, raised: 'the team text did not import' };
                const errors = validator.validateTeam(team);
                return { valid: !(errors && errors.length), errors: errors || [] };
            } catch (err) {
                return { valid: null, raised: err.message };     // infrastructure, never a verdict
            }
        });
    } catch (e) {
        out = { fatal: e.message };
    }
    process.stdout.write(JSON.stringify(out) + '\n');
});
