import json, sys, os, re
A, B, tag = sys.argv[1], sys.argv[2], sys.argv[3]
ca = json.load(open(os.path.join(A, 'capture', tag + '.json'))); cb = json.load(open(os.path.join(B, 'capture', tag + '.json')))
ka = [t for w, t in ca['chunks'] if w == 'p1']; kb = [t for w, t in cb['chunks'] if w == 'p1']
j = next((i for i in range(min(len(ka), len(kb))) if ka[i] != kb[i]), None)
if j is None: print('no p1-stream chunk diff'); sys.exit()
prev = [t for t in ka[:j] if not t.startswith('|request|')][-2:]
for t in prev: print('  PREV>', t.replace('\n', ' ¶ ')[:700])
print('  A>', ka[j].replace('\n', ' ¶ ')[:300]); print('  B>', kb[j].replace('\n', ' ¶ ')[:300])
