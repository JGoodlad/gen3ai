// EXCERPT: the blocks main.format_drift reads
	evasionitemsclause: {
		effectType: 'ValidatorRule',
		name: 'Evasion Items Clause',
		desc: "Bans items that lower the accuracy of moves used against the user",
		banlist: ['Bright Powder', 'Lax Incense'],
		onBegin() {
			this.add('rule', 'Evasion Items Clause: Evasion items are banned');
		},
	},
	evasionmovesclause: {
		effectType: 'ValidatorRule',
		name: 'Evasion Moves Clause',
		desc: "Bans moves that consistently raise the user's evasion when used",
		banlist: ['Acupressure', 'Minimize', 'Double Team'],
		onBegin() {
			this.add('rule', 'Evasion Moves Clause: Evasion moves are banned');
		},
	},
	oneboostpasserclause: {
		effectType: 'ValidatorRule',
		name: 'One Boost Passer Clause',
		desc: "Stops teams from having a Pok&eacute;mon with Baton Pass that has multiple ways to boost its stats, and no more than one Baton Passer may be able to boost its stats",
		onBegin() {
			this.add('rule', 'One Boost Passer Clause: Limit one Baton Passer that has a way to boost its stats');
		},
		onValidateTeam(team) {
			const boostingEffects = [
				'acidarmor', 'agility', 'amnesia', 'apicotberry', 'barrier', 'bellydrum', 'bulkup', 'calmmind', 'cosmicpower', 'curse',
				'defensecurl', 'dragondance', 'ganlonberry', 'growth', 'harden', 'howl', 'irondefense', 'liechiberry', 'meditate',
				'petayaberry', 'recycle', 'salacberry', 'sharpen', 'speedboost', 'starfberry', 'swordsdance', 'tailglow', 'withdraw',
			];
			let passers = 0;
			for (const set of team) {
				if (!set.moves.includes('Baton Pass')) continue;
				let passableBoosts = 0;
				const item = this.toID(set.item);
				const ability = this.toID(set.ability);
				for (const move of set.moves) {
					if (boostingEffects.includes(this.toID(move))) passableBoosts++;
				}
				if (boostingEffects.includes(item)) passableBoosts++;
				if (boostingEffects.includes(ability)) passableBoosts++;
				if (passableBoosts === 1) passers++;
				if (passableBoosts > 1) {
					return [
						`${set.name || set.species} has Baton Pass and multiple ways to boost its stats, which is banned by One Boost Passer Clause.`,
					];
				}
				if (passers > 1) {
					return [
						`Multiple Pokemon have Baton Pass and a way to boost their stats, which is banned by One Boost Passer Clause.`,
					];
				}
			}
		},
	},
	speedpassclause: {
		effectType: 'ValidatorRule',
		name: 'Speed Pass Clause',
		desc: "Stops teams from having a Pok&eacute;mon with Baton Pass that can boost its Speed",
		onBegin() {
			this.add('rule', 'Baton Pass Stat Clause: No Baton Passer may have a way to boost its Speed');
		},
		onValidateTeam(team) {
			const boostingEffects = [
				'agility', 'dragondance', 'ancientpower', 'silverwind', 'salacberry', 'speedboost', 'starfberry',
			];
			for (const set of team) {
				const moves = set.moves.map(this.toID);
				if (!moves.includes('batonpass' as ID)) continue;
				let passableBoosts = false;
				const item = this.toID(set.item);
				const ability = this.toID(set.ability);
				if (
					moves.some(m => boostingEffects.includes(m)) || boostingEffects.includes(item) ||
					boostingEffects.includes(ability)
				) {
					passableBoosts = true;
				}
				if (passableBoosts) {
					return [
						`${set.name || set.species} has Baton Pass and a way to boost its Speed, which is banned by Speed Pass Clause.`,
					];
				}
			}
		},
	},
	accuracytrapclause: {
		effectType: 'ValidatorRule',
		name: 'Accuracy Trap Clause',
		desc: "Bans guaranteed accuracy-dropping moves when used with a trapping move/ability on the same Pok&eacute;mon",
		onValidateSet(set, format, setHas, teamHas) {
			const trapping = [
				'arenatrap', 'magnetpull', 'shadowtag', 'block', 'meanlook', 'spiderweb', 'anchorshot', 'jawlock', 'octolock', 'spiritshackle', 'thousandwaves',
			];
			const accuracy = this.dex.moves.all().filter(move => {
				if (move.boosts?.accuracy) return move.boosts.accuracy < 0;
				return move.secondaries?.some(x => x.chance === 100 && x.boosts?.accuracy && x.boosts.accuracy < 0);
			}).map(x => x.id);
			if (set.moves.map(this.toID).some(x => accuracy.includes(x)) && (
				trapping.includes(this.toID(set.ability)) || set.moves.map(this.toID).some(x => trapping.includes(x))
			)) {
				return [
					`${set.species} has the combination of a trapping move/ability and a guaranteed accuracy-lowering move, which is banned.`,
				];
			}
		},
	},
	sleepclausemod: {
		effectType: 'Rule',
		name: 'Sleep Clause Mod',
		desc: "Prevents players from putting more than one of their opponent's Pok&eacute;mon to sleep at a time, and bans Mega Gengar from using Hypnosis",
		banlist: ['Hypnosis + Gengarite'],
		onBegin() {
			this.add('rule', 'Sleep Clause Mod: Limit one foe put to sleep');
		},
		onSetStatus(status, target, source) {
			if (source?.isAlly(target)) {
				return;
			}
			if (status.id === 'slp') {
				for (const pokemon of target.side.pokemon) {
					if (pokemon.hp && pokemon.status === 'slp') {
						if (!pokemon.statusState.source?.isAlly(pokemon)) {
							this.add('-message', 'Sleep Clause Mod activated.');
							this.hint("Sleep Clause Mod prevents players from putting more than one of their opponent's Pokémon to sleep at a time");
							return false;
						}
					}
				}
			}
		},
	},
	freezeclausemod: {
		effectType: 'Rule',
		name: 'Freeze Clause Mod',
		desc: "Prevents players from freezing more than one of their opponent's Pok&eacute;mon at a time",
		onBegin() {
			this.add('rule', 'Freeze Clause Mod: Limit one foe frozen');
		},
		onSetStatus(status, target, source) {
			if (source?.isAlly(target)) {
				return;
			}
			if (status.id === 'frz') {
				for (const pokemon of target.side.pokemon) {
					if (pokemon.status === 'frz') {
						this.add('-message', 'Freeze Clause activated.');
						return false;
					}
				}
			}
		},
	},
	ohkoclause: {
		effectType: 'ValidatorRule',
		name: 'OHKO Clause',
		desc: "Bans all OHKO moves, such as Fissure",
		onBegin() {
			this.add('rule', 'OHKO Clause: OHKO moves are banned');
		},
		onValidateSet(set) {
			const problems: string[] = [];
			if (set.moves) {
				for (const moveId of set.moves) {
					const move = this.dex.moves.get(moveId);
					if (move.ohko) problems.push(move.name + ' is banned by OHKO Clause.');
				}
			}
			return problems;
		},
	},
	standardag: {
		effectType: 'ValidatorRule',
		name: 'Standard AG',
		desc: "The minimal ruleset for Anything Goes",
		ruleset: [
			'Obtainable', 'Team Preview', 'HP Percentage Mod', 'Cancel Mod', 'Endless Battle Clause',
		],
	},
	standard: {
		effectType: 'ValidatorRule',
		name: 'Standard',
		desc: "The standard ruleset for all official Smogon singles tiers (Ubers, OU, etc.)",
		ruleset: [
			'Standard AG',
			'Sleep Clause Mod', 'Species Clause', 'Nickname Clause', 'OHKO Clause', 'Evasion Items Clause', 'Evasion Moves Clause',
		],
	},
	obtainable: {
		effectType: 'ValidatorRule',
		name: 'Obtainable',
		desc: "Makes sure the team is possible to obtain in-game.",
		ruleset: ['Obtainable Moves', 'Obtainable Abilities', 'Obtainable Formes', 'EV Limit = Auto', 'Obtainable Misc'],
		banlist: ['Unreleased', 'Unobtainable', 'Nonexistent'],
		// Mostly hardcoded in team-validator.ts
		onValidateTeam(team, format) {
			let kyuremCount = 0;
			let necrozmaDMCount = 0;
			let necrozmaDWCount = 0;
			let calyrexCount = 0;
			for (const set of team) {
				if (set.species === 'Kyurem-White' || set.species === 'Kyurem-Black') {
					if (kyuremCount > 0) {
						return [
							`You cannot have more than one Kyurem-Black/Kyurem-White.`,
							`(It's untradeable and you can only make one with the DNA Splicers.)`,
						];
					}
					kyuremCount++;
				}
				if (set.species === 'Necrozma-Dusk-Mane') {
					if (necrozmaDMCount > 0) {
						return [
							`You cannot have more than one Necrozma-Dusk-Mane`,
							`(It's untradeable and you can only make one with the N-Solarizer.)`,
						];
					}
					necrozmaDMCount++;
				}
				if (set.species === 'Necrozma-Dawn-Wings') {
					if (necrozmaDWCount > 0) {
						return [
							`You cannot have more than one Necrozma-Dawn-Wings`,
							`(It's untradeable and you can only make one with the N-Lunarizer.)`,
						];
					}
					necrozmaDWCount++;
				}
				if (set.species === 'Calyrex-Ice' || set.species === 'Calyrex-Shadow') {
					if (calyrexCount > 0) {
						return [
							`You cannot have more than one Calyrex-Ice/Calyrex-Shadow.`,
							`(It's untradeable and you can only make one with the Reins of Unity.)`,
						];
					}
					calyrexCount++;
				}
			}
			return [];
		},
	},
	speciesclause: {
		effectType: 'ValidatorRule',
		name: 'Species Clause',
		desc: "Prevents teams from having more than one Pok&eacute;mon from the same species",
		onBegin() {
			this.add('rule', 'Species Clause: Limit one of each Pokémon');
		},
		onValidateTeam(team, format) {
			const speciesTable = new Set<number>();
			for (const set of team) {
				const species = this.dex.species.get(set.species);
				if (speciesTable.has(species.num)) {
					return [`You are limited to one of each Pokémon by Species Clause.`, `(You have more than one ${species.baseSpecies})`];
				}
				speciesTable.add(species.num);
			}
		},
	},
	nicknameclause: {
		effectType: 'ValidatorRule',
		name: 'Nickname Clause',
		desc: "Prevents teams from having more than one Pok&eacute;mon with the same nickname",
		onValidateTeam(team, format) {
			const nameTable = new Set<string>();
			for (const set of team) {
				const name = set.name;
				if (name) {
					if (name === this.dex.species.get(set.species).baseSpecies) continue;
					if (nameTable.has(name)) {
						return [`Your Pokémon must have different nicknames.`, `(You have more than one ${name})`];
					}
					nameTable.add(name);
				}
			}
			// Illegality of impersonation of other species is
			// hardcoded in team-validator.js, so we are done.
		},
	},
	switchpriorityclausemod: {
		effectType: 'Rule',
		name: 'Switch Priority Clause Mod',
		desc: "Makes a faster Pokémon switch first when double-switching, unlike in Emerald link battles, where player 1's Pokémon would switch first",
		onBegin() {
			this.add('rule', 'Switch Priority Clause Mod: Faster Pokémon switch first');
		},
	},
	endlessbattleclause: {
		effectType: 'Rule',
		name: 'Endless Battle Clause',
		desc: "Prevents players from forcing a battle which their opponent cannot end except by forfeit",
		// implemented in sim/battle.js, see https://dex.pokemonshowdown.com/articles/battlerules#endlessbattleclause for the specification.
		onBegin() {
			this.add('rule', 'Endless Battle Clause: Forcing endless battles is banned');
		},
	},
	hppercentagemod: {
		effectType: 'Rule',
		name: 'HP Percentage Mod',
		desc: "Shows the HP of Pok&eacute;mon in percentages",
		onBegin() {
			this.add('rule', 'HP Percentage Mod: HP is shown in percentages');
			this.reportPercentages = true;
		},
	},
	cancelmod: {
		effectType: 'Rule',
		name: 'Cancel Mod',
		desc: "Allows players to change their own choices before their opponents make one",
		onBegin() {
			this.supportCancel = true;
		},
	},
	beatupnicknamesmod: {
		effectType: 'Rule',
		name: 'Beat Up Nicknames Mod',
		desc: "Prevents Beat Up from revealing any party members, enforcing gameplay that assumes optimal Pok&eacute;mon nicknaming strategies.",
		onBegin() {
			if (this.gen <= 4) {
				this.add('rule', `Beat Up Nicknames Mod: Beat Up will not reveal any party members`);
			}
		},
		// https://www.smogon.com/forums/posts/8992145/
		// hardcoded in data/mods/gen3/moves.ts, data/mods/gen4/moves.ts
	},
