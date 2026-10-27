// pack_norm.js — the fields of one set that the engine reads, as pack_oracle.js and showdown_pack.js compare them.
//
// ONE definition so the training pack oracle (gen3_team_pack_oracle_v1) and the Metamon peer's pack
// (gen3_peer_showdown_pack_v1) compare the same fields. `ivs` missing read 31, `happiness` missing reads 255 (the sim's
// default, `sim/pokemon.ts`), `hiddenpower` is the IV-derived type:power when the set carries the move.
'use strict';

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];

function norm(dex, toID, s) {
    const moves = (s.moves || []).map(m => dex.moves.get(m).id).filter(x => x);
    const hasHp = moves.some(m => m.startsWith('hiddenpower'));
    const ivsObj = s.ivs || { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
    const hpEff = hasHp ? dex.getHiddenPower(ivsObj) : null;
    return {
        species: dex.species.get(s.species || s.name).id,
        item: toID(s.item),
        ability: toID(s.ability),
        moves: moves.join(','),
        nature: toID(s.nature) || '',
        evs: STATS.map(k => (s.evs && s.evs[k]) || 0).join(','),
        ivs: STATS.map(k => ivsObj[k] === undefined ? 31 : ivsObj[k]).join(','),
        level: s.level || 100,
        happiness: (typeof s.happiness === 'number' && !isNaN(s.happiness)) ? s.happiness : 255,
        hiddenpower: hpEff ? `${hpEff.type}:${hpEff.power}` : '',
    };
}

module.exports = { STATS, norm };
