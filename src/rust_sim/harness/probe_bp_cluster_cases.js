const { run } = require('./probe_bp_modifier_cluster.js');
const MEW = (moves, item = 'leftovers') =>
  `Mew||${item}|synchronize|${moves}|Hardy|85,85,85,85,85,85|N||||`;
(async () => {
  // ---- FALSE SWIPE: must leave the target at >= 1 HP.
  // ⚠️ Shedinja+WonderGuard makes False Swipe IMMUNE, so that board tests nothing. Use a
  // Magikarp chipped to a sliver instead: the clamp is only observable when the move WOULD KO.
  await run('FS1 False Swipe that WOULD KO must leave exactly 1 HP',
    MEW('falseswipe,seismictoss,splash'),
    `Magikarp||leftovers|swiftswim|splash|Hardy|0,0,0,0,0,0|M|0,0,0,0,0,0||5|`,
    ['>p1 move 1\n>p2 move 1'], [9,9,9,9]);
  await run('FS2 False Swipe normal chip (control)',
    MEW('falseswipe,splash'), MEW('splash'),
    ['>p1 move 1\n>p2 move 1'], [9,9,9,9]);

  // ---- REVENGE: bp doubles if the user was damaged THIS TURN by the target.
  await run('RV1 Revenge AFTER being hit (should be 2x)',
    MEW('revenge,splash'), MEW('tackle,splash'),
    ['>p1 move 1\n>p2 move 1'], [9,9,9,9]);
  await run('RV2 Revenge NOT hit (control, base bp)',
    MEW('revenge,splash'), MEW('tackle,splash'),
    ['>p1 move 1\n>p2 move 2'], [9,9,9,9]);

  // ---- SMELLING SALTS: 2x vs a PARALYZED target, and CURES the paralysis.
  await run('SS1 Smelling Salts vs a paralyzed foe (2x + cure)',
    MEW('thunderwave,smellingsalts,splash'), MEW('splash'),
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1'], [9,9,9,9]);
  await run('SS2 Smelling Salts vs an UNstatused foe (control)',
    MEW('thunderwave,smellingsalts,splash'), MEW('splash'),
    ['>p1 move 3\n>p2 move 1', '>p1 move 2\n>p2 move 1'], [9,9,9,9]);

  // ---- FURY CUTTER: BP doubles per CONSECUTIVE use; resets on a miss / other move.
  await run('FC1 Fury Cutter x4 consecutive (BP ladder)',
    MEW('furycutter,splash'), MEW('splash'),
    ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1'], [9,9,9,9]);
  await run('FC2 Fury Cutter, BROKEN by another move (resets)',
    MEW('furycutter,splash'), MEW('splash'),
    ['>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 1'], [9,9,9,9]);

  // ---- DREAM EATER: only hits a SLEEPING target; drains half.
  await run('DE1 Dream Eater vs an AWAKE foe (should fail)',
    MEW('dreameater,splash'), MEW('splash'),
    ['>p1 move 1\n>p2 move 1'], [9,9,9,9]);
  await run('DE2 Dream Eater vs a SLEEPING foe (hits + drains)',
    MEW('spore,dreameater,seismictoss,splash'), MEW('splash'),
    ['>p1 move 3\n>p2 move 1', '>p1 move 1\n>p2 move 1', '>p1 move 2\n>p2 move 1'], [9,9,9,9]);
})();
