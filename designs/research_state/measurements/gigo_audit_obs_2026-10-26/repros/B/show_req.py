"""Print the protocol context of p1 requests matching a predicate (struggle-only / single-entry / trapped)."""
import json, sys
from utils.rust_env import ffi
lib = ffi.load(ffi.default_path('release'))
kind = sys.argv[1]; lim = int(sys.argv[2])
shown = 0
for path in sys.argv[3:]:
    for gi, g in enumerate(json.load(open(path))['games']):
        t = ffi.trace(lib, g['fin']['script'], commit='a', label='x')
        lines = [json.loads(l)['text'] for l in t['records']['p1'].splitlines()[1:]]
        for i, text in enumerate(lines):
            if not text.startswith('|request|'):
                continue
            r = json.loads(text[9:]); a = (r.get('active') or [None])[0]
            if not a:
                continue
            ids = [m['id'] for m in a['moves']]
            hit = {'struggle': ids == ['struggle'], 'single': len(ids) == 1 and ids != ['struggle'],
                   'trapped': bool(a.get('trapped')) and not r.get('update'), 'maybe': bool(a.get('maybeTrapped'))}[kind]
            if hit and shown < lim:
                shown += 1
                print('==', path.split('/')[-1], 'game', gi)
                print('\n'.join(l for l in lines[max(0, i - 14):i] if not l.startswith('|request|')))
                print('REQ active:', json.dumps(a)[:400])
