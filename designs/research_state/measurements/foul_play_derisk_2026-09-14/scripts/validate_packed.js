const path = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown/dist/sim/index.js';
const { Teams, TeamValidator } = require(path);
const fs = require('fs');
const packed = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const validator = new TeamValidator('gen3ou');
const out = {};
for (const [name, p] of Object.entries(packed)) {
  let errs;
  try {
    const team = Teams.unpack(p);
    errs = validator.validateTeam(team);
  } catch (e) { errs = ['unpack/validate threw: ' + e.message]; }
  out[name] = errs || null;
}
console.log(JSON.stringify(out, null, 1));
