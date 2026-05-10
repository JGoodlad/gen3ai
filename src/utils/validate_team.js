// validate_team.js
// This script bridges Python and the local Pokemon Showdown library for team validation.
const path = require('path');

// Point to the local Pokemon Showdown installation
// Based on the repository structure: deps/pokemon-showdown
const psPath = path.resolve(__dirname, '../../deps/pokemon-showdown');
const { Teams, TeamValidator } = require(psPath);

let inputData = '';
process.stdin.on('data', chunk => { inputData += chunk; });

process.stdin.on('end', () => {
    try {
        if (!inputData) {
            console.log(JSON.stringify({ valid: false, errors: ['No input data received'] }));
            return;
        }

        const request = JSON.parse(inputData);
        const formatId = request.format || 'gen3ou'; 
        const teamText = request.team;   

        if (!teamText) {
            console.log(JSON.stringify({ valid: false, errors: ['No team text provided'] }));
            return;
        }

        const validator = new TeamValidator(formatId);
        const teamJson = Teams.import(teamText);
        const errors = validator.validateTeam(teamJson);

        if (errors && errors.length > 0) {
            console.log(JSON.stringify({ valid: false, errors: errors }));
        } else {
            console.log(JSON.stringify({ valid: true, errors: [] }));
        }
    } catch (e) {
        console.log(JSON.stringify({ valid: false, errors: [e.message] }));
    }
});
