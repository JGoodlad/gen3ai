// Replay a core episode script (START {json} + CHOOSE pN choice lines) in Showdown's own sim; print the spectator log.
const path = process.argv[2]; const ps = process.argv[3];
const {Battle} = require(ps + '/dist/sim/battle');
const fs = require('fs');
const lines = fs.readFileSync(path, 'utf8').split('\n').filter(Boolean);
const start = JSON.parse(lines[0].slice(6));
const b = new Battle({formatid: start.formatid, seed: start.seed.split(',').map(Number)});
b.setPlayer('p1', {name: start.p1.name, team: start.p1.team});
b.setPlayer('p2', {name: start.p2.name, team: start.p2.team});
for (const l of lines.slice(1)) {
  const m = l.match(/^CHOOSE (p[12]) (.*)$/);
  if (!m) { console.log('?? ' + l); continue; }
  b.log.push(`|AUDIT-CHOICE|${m[1]}|${m[2]}`);
  b.choose(m[1], m[2]);
}
console.log(b.log.join('\n'));
