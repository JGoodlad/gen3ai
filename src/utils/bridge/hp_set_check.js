// hp_set_check.js — the pinned Showdown's verdict on every Hidden Power set of a list of pastes.
//
// stdin:  {"format": "gen3ou", "teams": ["<paste>", ...]}
// stdout: [{"valid": bool, "errors": [...], "sets": [{species, hp_move, hp_type, power, ivs_in,
//            ivs_after_validate, validator_hp_type, stats}]}, ...]
//
// For each team: Teams.import -> TeamValidator.validateTeam (which may rewrite set.ivs / set.hpType,
// sim/team-validator.ts ~1144-1167), then dex.getHiddenPower(ivs) under gen 3 and the battle's stored
// stats (sim/battle.ts spreadModify, via a constructed Battle). Used by agents.gen3_data.hp_repair's
// integration tests and the heldout72_hpfix regeneration (utils.bridge.hp_set_check).
const path = require('path');
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex, Teams, TeamValidator, Battle } = require(psPath);

let input = '';
process.stdin.on('data', c => { input += c; });
process.stdin.on('end', () => {
	const req = JSON.parse(input);
	const format = req.format || 'gen3ou';
	const dex = Dex.forFormat(format);
	const validator = new TeamValidator(format);
	const out = [];
	for (const text of req.teams) {
		const team = Teams.import(text);
		const ivsIn = team.map(s => Object.assign({}, s.ivs));
		const errors = validator.validateTeam(team) || [];
		let stored = [];
		try {
			const battle = new Battle({ formatid: format, seed: [1, 2, 3, 4] });
			battle.setPlayer('p1', { name: 'a', team: Teams.pack(Teams.import(text)) });
			stored = battle.sides[0].pokemon.map(p => Object.assign({ hp: p.maxhp }, p.storedStats));
		} catch (e) {
			errors.push('battle construction: ' + e.message);
		}
		const sets = team.map((s, i) => {
			const hpMove = s.moves.find(m => dex.moves.get(m).id === 'hiddenpower');
			const gh = dex.getHiddenPower(ivsIn[i]);
			return {
				species: dex.species.get(s.species).id,
				hp_move: hpMove || null,
				hp_type: hpMove ? gh.type : null,
				power: hpMove ? gh.power : null,
				ivs_in: ivsIn[i],
				ivs_after_validate: s.ivs,
				validator_hp_type: s.hpType || null,
				stats: stored[i] || null,
			};
		});
		out.push({ valid: errors.length === 0, errors, sets });
	}
	console.log(JSON.stringify(out));
});
