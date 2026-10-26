// Compare Showdown's own import(+validator mutations) of each PASTE with Showdown's unpack of OUR packed string.
const path = require('path');
const ps = path.resolve('/home/goodlad/dev/gen3ai-wt/gigo-audit-obs-1026/deps/pokemon-showdown');
const { Teams, TeamValidator, Dex, toID } = require(ps);
const rows = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
const v = new TeamValidator('gen3ou');
const dex = Dex.mod('gen3');
const STATS = ['hp','atk','def','spa','spd','spe'];
function norm(s) {
  const sp = dex.species.get(s.species || s.name);
  const mv = (s.moves || []).map(m => { const d = dex.moves.get(m); return d.id === 'hiddenpower' && d.type !== 'Normal' ? 'hiddenpower' + toID(d.type) : toID(m); }).filter(x => x).sort();
  const ivs = s.ivs ? STATS.map(k => s.ivs[k]) : [31,31,31,31,31,31];
  const evs = s.evs ? STATS.map(k => s.evs[k]) : [0,0,0,0,0,0];
  let hpType = s.hpType || '';
  // the sim's own HP type: set.hpType || getHiddenPower(ivs).type (gen<=7)
  const hpEff = (s.moves||[]).some(m => dex.moves.get(m).id === 'hiddenpower') ? (hpType || dex.getHiddenPower(s.ivs ? s.ivs : {hp:31,atk:31,def:31,spa:31,spd:31,spe:31}).type) : '';
  const hpPow = (s.moves||[]).some(m => dex.moves.get(m).id === 'hiddenpower') ? dex.getHiddenPower(s.ivs ? s.ivs : {hp:31,atk:31,def:31,spa:31,spd:31,spe:31}).power : 0;
  return {
    species: sp.id, item: toID(s.item), ability: toID(s.ability), moves: mv.join(','),
    nature: toID(s.nature) || 'serious', evs: evs.join(','), ivs: ivs.join(','),
    level: s.level || 100, happiness: (typeof s.happiness === 'number') ? s.happiness : 255,
    hp: hpEff ? (hpEff + ':' + hpPow) : '',
  };
}
const out = { n: 0, invalid_paste: 0, invalid_pack: 0, diffs: [], field_counts: {} };
for (const r of rows) {
  out.n++;
  let A, B;
  try { A = Teams.import(r.paste); } catch (e) { out.diffs.push({ id: r.id, err: 'import ' + e.message }); continue; }
  const ea = v.validateTeam(A);
  if (ea && ea.length) { out.invalid_paste++; continue; }   // a paste the validator refuses never trains
  B = Teams.unpack(r.packed);
  if (!B) { out.diffs.push({ src: r.src, id: r.id, err: 'unpack null' }); continue; }
  const Bv = Teams.unpack(r.packed);
  const eb = v.validateTeam(Bv);
  if (eb && eb.length) out.invalid_pack++;
  if (A.length !== B.length) { out.diffs.push({ src: r.src, id: r.id, err: `size ${A.length} vs ${B.length}` }); continue; }
  for (let i = 0; i < A.length; i++) {
    const a = norm(A[i]), b = norm(B[i]);
    for (const k of Object.keys(a)) if (String(a[k]) !== String(b[k])) {
      out.field_counts[k] = (out.field_counts[k] || 0) + 1;
      out.diffs.push({ src: r.src, id: r.id, mon: a.species, field: k, showdown: a[k], ours: b[k], pack_valid: !(eb && eb.length) });
    }
  }
}
console.log(JSON.stringify(out, null, 1));
