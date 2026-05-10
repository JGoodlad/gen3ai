// get_hp.js
// This script bridges Python and the local Pokemon Showdown library to calculate Hidden Power.
const path = require('path');

// Point to the local Pokemon Showdown installation
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(psPath);

let inputData = '';
process.stdin.on('data', chunk => { inputData += chunk; });

process.stdin.on('end', () => {
    try {
        if (!inputData) {
            console.log(JSON.stringify({ error: 'No input data received' }));
            return;
        }

        const request = JSON.parse(inputData);
        // Expecting { "hp": 31, "atk": 31, ... }
        const ivs = {
            hp: request.hp,
            atk: request.atk,
            def: request.def,
            spa: request.spa,
            spd: request.spd,
            spe: request.spe
        };

        const hpData = Dex.forGen(3).getHiddenPower(ivs);
        console.log(JSON.stringify(hpData));
    } catch (e) {
        console.log(JSON.stringify({ error: e.message }));
    }
});
