"""Slice 4 identity corpus: --obs-stream inputs from real core-trace battles (both sides, recorded
actions), plus variants (truncated actions, encode_at subsets, a spectator read). Every read_streams
call goes through the tee wrapper ($POKESIM_CORE_EVENTS_BIN), which banks its stdin."""
import glob
import random
import sys

from main.prober.core_walk import StreamRequest, read_streams, walk, CoreWalkError
from utils.bridge.reconstruction import ReconstructionRecord

n_battles = int(sys.argv[1])
rng = random.Random(20261007)
paths = sorted(glob.glob('/home/goodlad/dev/gen3ai/models/*/eval_traces/*/*/*_reconstruction.json'))
rng.shuffle(paths)
done = refused = 0
for p in paths:
    if done >= n_battles:
        break
    try:
        rec = ReconstructionRecord.load(p)
        reqs = []
        for side in ('p1', 'p2'):
            w = walk(rec, side)
            players = rec.players()
            chunks = [c for s, c in w.chunks if s == side]
            acts = []
            for d in w.decisions:
                inv = {t: a for a, t in d.tokens.items()}
                if d.choice is None or d.choice not in inv:
                    break
                acts.append(inv[d.choice])
            team = players[side]['team']
            reqs.append(StreamRequest(players[side]['name'], team, side, chunks, acts, None))
            k = len(acts)
            reqs.append(StreamRequest(players[side]['name'], team, side, chunks, acts[: k // 2],
                                      sorted(rng.sample(range(max(k, 1)), min(3, k))) if k else []))
        # a spectator-style read (no team, no actions)
        reqs.append(StreamRequest(players['p1']['name'], None, 'p1', [c for s, c in w.chunks if s == 'p1'], (), []))
        try:
            read_streams(reqs)
        except CoreWalkError as e:          # a refused stream is corpus too (the error text is compared)
            refused += 1
            print('refused', p, str(e)[:200])
        done += 1
    except CoreWalkError as e:
        print('walk refused', p, str(e)[:200])
print('battles', done, 'with a refused stream', refused)
