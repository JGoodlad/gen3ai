# HP Ghost IV fix: impact on the pool (2026-10-19)

`utils.gen3_utils.GEN3_HP_IVS['Ghost']` was 30 Atk / 30 SpD; Showdown's `typechart.ts` `HPivs` (the validator's fill) is
30 Def / 30 SpD. All 16 rows were compared with `typechart.ts`: Ghost was the only mismatch (the other 15 are equal).
`scripts/impact.py` runs `fix_gen3_hp_ivs` on every `data/teams/**/*.txt` and every `w_forum_v1` team with the old and the new
table and lists the mons whose IVs differ (a Hidden Power Ghost set whose IVs are all 31, i.e. no `IVs:` line):

- `data/teams`: **37 mons in 36 of 813 files** (old 31/30/31/31/30/31 -> new 31/31/30/31/30/31: +1 Atk, -1 Def).
- `src/utils/team_corpora/w_forum_v1`: **0 of 2,415 teams** (its HP sets carry explicit IVs lines).

Affected files (mon, count when more than one):

- `data/teams/others/giraffe/25603321f1932c95.txt` Tyranitar (1 HP Ghost mon(s))
- `data/teams/others/giraffe/28679d29ac69f521.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/2bfb4e58227e6722.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/30f1a487881e3ac3.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/4847ded117ac0ce6.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/4a0deca178cff942.txt` Swampert (1 HP Ghost mon(s))
- `data/teams/others/giraffe/50bffebdd7cbc754.txt` Gyarados (1 HP Ghost mon(s))
- `data/teams/others/giraffe/6cfe6f2a1212db5f.txt` Gyarados (1 HP Ghost mon(s))
- `data/teams/others/giraffe/6e5a9bcc043944b6.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/726f22237e16b1e5.txt` Gyarados (1 HP Ghost mon(s))
- `data/teams/others/giraffe/7a0897054f2a39df.txt` Gyarados (1 HP Ghost mon(s))
- `data/teams/others/giraffe/821ca63b11ab9fd1.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/853aefd0527907f7.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/8d2fb6d8959c2ba5.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/8daf23fb861838f3.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/9b06a1104209d2e9.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/giraffe/bbd3b6da43ede1c4.txt` Gyarados (1 HP Ghost mon(s))
- `data/teams/others/giraffe/c477910ef0fec296.txt` Gyarados (1 HP Ghost mon(s))
- `data/teams/others/giraffe/cf125ab2401ea7de.txt` Tyranitar (1 HP Ghost mon(s))
- `data/teams/others/giraffe/d77c3226e2bc5417.txt` Gyarados (1 HP Ghost mon(s))
- `data/teams/others/giraffe/e39fddb0da5bb4e3.txt` Tyranitar (1 HP Ghost mon(s))
- `data/teams/others/giraffe/e8114d30c6b8192b.txt` Swampert (1 HP Ghost mon(s))
- `data/teams/others/giraffe/ec21f99b72e5767d.txt` Forretress (2 HP Ghost mon(s))
- `data/teams/others/giraffe/fd5e8ec3c57d7070.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/johnnyg2/48159c2b99b533b5.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/johnnyg2/7050a8a45aad1a9e.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/johnnyg2/9751d9b5e8b96537.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/johnnyg2/ac11b908610cf835.txt` Forretress (1 HP Ghost mon(s))
- `data/teams/others/johnnyg2/da25c47e03b50001.txt` Marowak (1 HP Ghost mon(s))
- `data/teams/others/mcmegan/039021d0cba54f63.txt` Hariyama (1 HP Ghost mon(s))
- `data/teams/others/mcmegan/4d83e95608b53766.txt` Hariyama (1 HP Ghost mon(s))
- `data/teams/others/schmuck_nick/46a1ddc038ec76d5.txt` Machamp (1 HP Ghost mon(s))
- `data/teams/others/schmuck_nick/7e5c0056122cd255.txt` Breloom (1 HP Ghost mon(s))
- `data/teams/others/schmuck_nick/ccd0dbf70c3eb09d.txt` Breloom (1 HP Ghost mon(s))
- `data/teams/sample/0972146213a667c9.txt` Forretress (1 HP Ghost mon(s))
w_forum_v1: 0 of 2415 teams affected, 0 mons
