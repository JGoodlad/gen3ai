"""Packed teams (validator-bypassed, straight to the Rust core) carrying the dex's edge ids: forme species
(Deoxys-Speed / Castform under Sunny Day / Unown), the highest species / move nums (Deoxys 386, Psycho Boost 354,
typed HP), Ditto's Transform, Kecleon's Color Change, a '???'-type move (Curse), Struggle-prone PP-less sets."""
import json, sys
A = [
 "Deoxys||lumberry|pressure|psychoboost,superpower,extremespeed,hiddenpowerfire|Hasty|,252,,4,,252||,30,,30,,30|||",
 "Castform||leftovers|forecast|sunnyday,weatherball,flamethrower,curse|Modest|4,,,252,,252|||||",
 "Unown||twistedspoon|levitate|hiddenpowerpsychic|Modest|,,,252,4,252||,2,,30,,|||",
 "Ditto||leftovers|limber|transform|Relaxed|252,,252,,4,|||||",
 "Kecleon||leftovers|colorchange|return,shadowball,brickbreak,trick|Adamant|252,252,,,4,|||||",
 "Smeargle||lumberry|owntempo|spore,spikes,batonpass,transform|Jolly|252,,4,,,252|||||",
]
B = [
 "Rayquaza||leftovers|airlock|dragondance,extremespeed,earthquake,hiddenpowerflying|Adamant|,252,,4,,252||,,,30,30,|||",
 "Kyogre||leftovers|drizzle|surf,thunder,icebeam,calmmind|Modest|4,,,252,,252|||||",
 "Groudon||leftovers|drought|earthquake,rockslide,overheat,swordsdance|Adamant|252,252,,,4,|||||",
 "Mew||leftovers|synchronize|softboiled,psychic,icebeam,curse|Calm|252,,,4,252,|||||",
 "Wobbuffet||leftovers|shadowtag|counter,mirrorcoat,encore,destinybond|Bold|252,,252,,4,|||||",
 "Shedinja||brightpowder|wonderguard|shadowball,swordsdance,batonpass,silverwind|Adamant|,252,,4,,252|||||",
]
json.dump([']'.join(A), ']'.join(B)], open(sys.argv[1], 'w'))
