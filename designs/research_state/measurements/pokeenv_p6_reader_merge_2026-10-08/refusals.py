"""The refusal paths: an illegal action, an unknown keyword, a truncated stream — old vs new verdicts."""
import glob, json, subprocess, sys

old, new = sys.argv[1], sys.argv[2]
f = sorted(glob.glob('/tmp/p6coord/s4/corpus/*.in'))[0]
lines = open(f).read().split('\n')
# first stream only
end = lines.index('END')
head, body = lines[0], lines[1:end]
h = json.loads(head[len('STREAM '):])
cases = {}
h1 = dict(h); h1['actions'] = [10] * 3
cases['illegal_action'] = ['STREAM ' + json.dumps(h1)] + body + ['END']
mid = len(body) // 2
cases['unknown_keyword'] = [head] + body[:mid] + ['|bogusword|p1a: X'] + body[mid:] + ['END']
cases['request_dropped'] = [head] + [l for l in body if not l.startswith('|request|')][:200] + ['END']
for name, c in cases.items():
    data = ('\n'.join(c) + '\n').encode()
    a = subprocess.run([old, '--obs-stream'], input=data, capture_output=True).stdout.decode().strip()
    b = subprocess.run([new, '--obs-stream'], input=data, capture_output=True).stdout.decode().strip()
    ra, rb = json.loads(a), json.loads(b)
    print(name, '| old ok', ra['ok'], '| new ok', rb['ok'], '| same bytes', a == b)
    if a != b:
        print('   old:', (ra.get('error') or '')[:160])
        print('   new:', (rb.get('error') or '')[:160])
