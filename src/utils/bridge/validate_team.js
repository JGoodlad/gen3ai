// validate_team.js
// This script bridges Python and the local Pokemon Showdown library for team validation.
const path = require('path');

// Point to the local Pokemon Showdown installation
// Based on the repository structure: deps/pokemon-showdown
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
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
        const validator = new TeamValidator(formatId);

        // Handle both single team and array of teams
        const teams = Array.isArray(request.team) ? request.team : [request.team];
        const results = [];

        for (const teamText of teams) {
            if (!teamText) {
                results.push({ valid: false, errors: ['No team text provided'] });
                continue;
            }
            try {
                const teamJson = Teams.import(teamText);
                const errors = validator.validateTeam(teamJson);
                if (errors && errors.length > 0) {
                    results.push({ valid: false, errors: errors });
                } else {
                    results.push({ valid: true, errors: [] });
                }
            } catch (err) {
                results.push({ valid: false, errors: [err.message] });
            }
        }

        // Return single object for single team, array for multiple
        console.log(JSON.stringify(Array.isArray(request.team) ? results : results[0]));
    } catch (e) {
        console.log(JSON.stringify({ valid: false, errors: [e.message] }));
    }
});
