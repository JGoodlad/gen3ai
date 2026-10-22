# X52 — the practice ceiling: the two team halves (drawn 2026-10-22, nothing played)

The design is [`design_beat_metamon_path.md`](../../../endstate/design_beat_metamon_path.md) §6.2. The registration
(the lever, the read, the rules, the predictions and FINDINGS F-X52-1 to 6) is the X35 launch sheet's §0, AM-X52
([`BATCH1_LAUNCH_SHEET.md`](../x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md)).

- **`teams_practised/`** holds 36 teams. They are X52's `--trainee-teams`, the only teams our side pilots in the fork.
- **`teams_unpractised/`** holds 36 teams that no X52 run pilots or faces: the transfer test.
- **`x52_teams_manifest.json`** records the source, the rule, every team's sha256, half and stratum, and the overlap
  histograms. These include the cross-half overlap: ≤ 3 shared mon sets by construction, and 26 of 36 unpractised
  teams share 4–6 SPECIES with some practised team.
- **`draw_x52_teams.py`** reproduces both halves byte for byte from `w_forum_v1` tier A at the production pin
  `f4b25238`: `python <it> --check` exits 0.

**Never move, edit or redraw these files.** X52's launch reads the practised files from main's checkout, and
`main.x35.x52` refuses a launch whose files do not match the manifest. These teams are unfamiliar ONLY to P\*'s lineage:
`w_forum_v1` holds all 72, so a D1-lineage model has seen them.
