"""BUILD the perturbed sets and their Rust encodings (no policy forward). Writes draws.npz + draws_meta.json.

For each panel (home72, F, H, P72) and each of the first 48 teams (the teams side 0 piloted in the archived rows),
each slot whose species is D3-eligible:
  * PRIMARY: K=3 Smogon usage draws of the same species by D3's own generator (`build_d3_setmut.mutate_block`, its
    engine prefilter, the HP repair), rule 4 (a NEW mon set, `team_overlap` key) and the pinned gen3ou validator on the
    whole team; seeds sha256(n2_identity_v1, panel, team, slot, attempt);
  * S3: (i) the move lines reversed, (ii) a nickname, (iii) a stat-identical EV change (4 -> 6, else 252 -> 255).
Every team is encoded by the Rust core's turn-1 observation; the target species' slot vector is kept."""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter

import numpy as np

import n2common as N
from agents.observation import constants as K

C = N.C
K_DRAWS = 3
ATTEMPT_ROUNDS = ((0, 8), (8, 20), (20, 40))
N_TEAMS = 48
PANELS = ("home72", "F", "H", "P72")


def seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in ("n2_identity_v1",) + parts).encode()).hexdigest()[:16], 16)


def reverse_moves(block: str) -> str:
    lines = block.split("\n")
    mv = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("- ")]
    out = list(lines)
    for i, j in zip(mv, reversed(mv)):
        out[i] = lines[j]
    return "\n".join(out)


def add_nickname(block: str) -> str:
    from utils.team_corpora.build_d3_setmut import _NICK
    lines = block.split("\n")
    left, sep, right = lines[0].partition(" @ ")
    m = _NICK.match(left)
    if m and m.group("species").strip() not in ("M", "F"):
        new_left = "Zzyzx" + left[len(m.group("nick")):]
    else:
        g = re.search(r"\s*\((?:M|F)\)\s*$", left)
        base = left[:g.start()] if g else left
        new_left = f"Zzyzx ({base.strip()})" + (g.group(0) if g else "")
    lines[0] = new_left + sep + right
    return "\n".join(lines)


def ev_equivalent(block: str):
    """A stat-identical EV change: an EV of 4 -> 6 (total <= 510), else 252 -> 255; None if neither applies."""
    lines = block.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("evs:"):
            parts = [p.strip() for p in ln.split(":", 1)[1].split("/")]
            vals = [(int(p.split()[0]), p.split()[1]) for p in parts if p]
            tot = sum(v for v, _ in vals)
            for old, new in ((4, 6), (252, 255)):
                for k, (v, lab) in enumerate(vals):
                    if v == old and tot - old + new <= 510:
                        vals2 = list(vals)
                        vals2[k] = (new, lab)
                        lines[i] = "EVs: " + " / ".join(f"{a} {b}" for a, b in vals2)
                        return "\n".join(lines)
            return None
    return None


def main() -> None:
    from agents.gen3_data import priors
    from main.anchors.team_overlap import parse_team
    from static_probe import SLOT, signature, turn1_slots
    from utils.bridge.team_validator import validate_teams_locally
    from utils.team_corpora import build_d3_setmut as D3
    from utils.team_sources import packed_teams

    lib = N.lib()
    pool_sigs = {signature(s) for s in turn1_slots(packed_teams("pool"), lib).reshape(-1, SLOT)}
    texts = {p: C.panel_texts(p)[:N_TEAMS] for p in PANELS}
    blocks = {p: [D3.split_blocks(t) for t in texts[p]] for p in PANELS}
    for p in PANELS:
        assert all(len(b) == 6 for b in blocks[p]), p
    species = sorted({D3.block_species(b) for p in PANELS for bl in blocks[p] for b in bl})
    all_moves = sorted({m for s in species for m in priors.moves(s) if m})
    all_items = sorted({i for s in species for i in priors.items(s) if i != "nothing"})
    bad_moves = frozenset(D3._engine_unplayable("move", all_moves))
    bad_items = frozenset(D3._engine_unplayable("item", all_items))
    table = {s: D3.species_prior(s, bad_moves, bad_items) for s in species}

    # the originals' turn-1 slots (every panel team validates: C.pack asserts it)
    orig = {p: turn1_slots(C.pack(texts[p]), lib) for p in PANELS}         # [48,6,SLOT]
    sp_of = {p: [[int(round(orig[p][i, k, K.POKEMON_SPECIES_OFFSET])) for k in range(6)] for i in range(N_TEAMS)]
             for p in PANELS}
    for p in PANELS:
        for i in range(N_TEAMS):
            assert len(set(sp_of[p][i])) == 6, (p, i, "species clause")

    # targets: block b of team i whose species is D3-eligible and whose set has the group's familiarity
    targets, stats = [], Counter()
    for p in PANELS:
        for i in range(N_TEAMS):
            for b in range(6):
                inpool = signature(orig[p][i, b]) in pool_sigs
                if inpool != (p == "home72"):
                    stats[f"{p}_wrong_familiarity"] += 1
                    continue
                if not table[D3.block_species(blocks[p][i][b])].eligible:
                    stats[f"{p}_species_not_eligible"] += 1
                    continue
                targets.append((p, i, b))
    stats["targets"] = len(targets)

    # PRIMARY draws, in attempt rounds; validator in one batch per round
    got = {t: [] for t in targets}
    for a0, a1 in ATTEMPT_ROUNDS:
        cands = []
        for (p, i, b) in targets:
            if len(got[(p, i, b)]) >= K_DRAWS:
                continue
            src_key = parse_team(texts[p][i]).mons[b].key
            for att in range(a0, a1):
                rng = random.Random(seed(p, i, b, att))
                new, rec = D3.mutate_block(blocks[p][i][b], table[D3.block_species(blocks[p][i][b])], rng,
                                           seed(p, i, b, att, "hp"))
                bl = list(blocks[p][i])
                bl[b] = new
                text = "\n\n".join(bl)
                if parse_team(text).mons[b].key == src_key:
                    stats["rule4_unchanged"] += 1
                    continue
                cands.append(((p, i, b), att, text, rec))
        val = validate_teams_locally("gen3ou", [c[2] for c in cands]) if cands else []
        for (t, att, text, rec), v in zip(cands, val):
            if not v.get("valid"):
                stats["validator_reject"] += 1
                continue
            if len(got[t]) < K_DRAWS:
                got[t].append((att, text, rec))
    stats["targets_short_of_K"] = sum(1 for t in targets if len(got[t]) < K_DRAWS)
    stats["targets_no_draw"] = sum(1 for t in targets if not got[t])

    # S3 variants
    s3 = {}
    for (p, i, b) in targets:
        for kind, fn in (("reverse", reverse_moves), ("nickname", add_nickname), ("ev_equiv", ev_equivalent)):
            nb = fn(blocks[p][i][b])
            if nb is None or nb == blocks[p][i][b]:
                stats[f"s3_{kind}_not_applicable"] += 1
                continue
            bl = list(blocks[p][i])
            bl[b] = nb
            s3[(p, i, b, kind)] = "\n\n".join(bl)
    keys3 = list(s3)
    val3 = validate_teams_locally("gen3ou", [s3[k] for k in keys3])
    for k, v in zip(keys3, val3):
        if not v.get("valid"):
            stats[f"s3_{k[3]}_validator_reject"] += 1
            del s3[k]

    # encode every accepted team; keep the target species' slot
    jobs = [("draw", t, d, got[t][d][1]) for t in targets for d in range(len(got[t]))]
    jobs += [("s3", k[:3], k[3], txt) for k, txt in s3.items()]
    packed = C.pack([j[3] for j in jobs])
    enc = turn1_slots(packed, lib)                                           # [J,6,SLOT]
    keys, vecs = [], []
    for j, (kind, (p, i, b), d, _) in enumerate(jobs):
        want = sp_of[p][i][b]
        hit = [k for k in range(6) if int(round(enc[j, k, K.POKEMON_SPECIES_OFFSET])) == want]
        assert len(hit) == 1, (p, i, b, d)
        keys.append(f"{kind}|{p}|{i}|{b}|{d}")
        vecs.append(enc[j, hit[0]])
    meta = {"stats": dict(stats), "k_draws": K_DRAWS, "n_teams": N_TEAMS,
            "draw_records": {f"{p}|{i}|{b}|{d}": {"attempt": got[(p, i, b)][d][0], **got[(p, i, b)][d][2]}
                             for (p, i, b) in targets for d in range(len(got[(p, i, b)]))},
            "species_of_slot": {p: sp_of[p] for p in PANELS}}
    np.savez_compressed(N.OUT / "draws.npz", keys=np.array(keys), vecs=np.stack(vecs),
                        **{f"orig_{p}": orig[p] for p in PANELS})
    json.dump(meta, open(N.OUT / "draws_meta.json", "w"), indent=1)
    print(json.dumps(meta["stats"], indent=1), file=sys.stderr)
    print(f"{len(keys)} encoded slots ({sum(1 for k in keys if k.startswith('draw'))} draws)")


if __name__ == "__main__":
    main()
