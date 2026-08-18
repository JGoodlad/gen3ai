// PROBE 3: the ONLY natural gen-3 carrier — CASTFORM (Forecast). The user is RETYPED by
// the same weather that retypes the ball, so STAB composes. Also pins the emitted line
// order around |-formechange|.
const path=require('path'); const PS='/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const {BattleStream}=require(path.join(PS,'dist/sim/battle-stream.js'));
const {BattleActions}=require(path.join(PS,'dist/sim/battle-actions.js'));
let calls=[];
const o=BattleActions.prototype.getDamage;
BattleActions.prototype.getDamage=function(s,t,m){
  const rec=(m&&typeof m==='object'&&m.id)?{id:m.id,type:m.type,bp:m.basePower,cat:m.category,
    stab:s.hasType(m.type),utypes:s.getTypes().join('/'),spa:s.getStat('spa'),atk:s.getStat('atk')}:null;
  const r=o.apply(this,arguments); if(rec){rec.dmg=r;calls.push(rec);} return r;};
const RE=/^\|(move|-damage|-crit|-supereffective|-resisted|-immune|-formechange|-weather|turn|switch)\|/;
async function play(p1,p2,script){calls=[];const s=new BattleStream();const ch=[];
 (async()=>{for await(const c of s)ch.push(c);})();
 s.write(`>start {"formatid":"gen3customgame","seed":[9,9,9,9]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
 await new Promise(r=>setTimeout(r,200));
 for(const c of script){s.write(c);await new Promise(r=>setTimeout(r,200));}
 await new Promise(r=>setTimeout(r,200));
 return {omni:ch.filter(c=>!c.startsWith('sideupdate')).join('\n').split('\n').filter(l=>RE.test(l)),calls:calls.slice()};}
(async()=>{
 const CF='Castform||Leftovers|Forecast|weatherball,raindance,sunnyday,splash|Hardy|0,0,0,0,0,252|M||||';
 const T ='Mew||Leftovers|Synchronize|splash|Hardy|0,0,0,0,0,0|||||';
 for(const [lab,set] of [['NO WEATHER','>p1 move 4'],['RAIN','>p1 move 2'],['SUN','>p1 move 3']]){
   const r=await play(CF,T,[`${set}\n>p2 move 1`,'>p1 move 1\n>p2 move 1']);
   console.log(`\n== CASTFORM / ${lab}\n   `+r.omni.join('\n   '));
   for(const c of r.calls) if(c.id==='weatherball')
     console.log(`   getDamage: userTypes=${c.utypes} type=${c.type} bp=${c.bp} cat=${c.cat} STAB=${c.stab} atk=${c.atk} spa=${c.spa} -> ${c.dmg}`);
 }
})();
