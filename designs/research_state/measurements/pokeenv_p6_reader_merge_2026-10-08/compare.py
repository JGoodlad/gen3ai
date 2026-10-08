"""old vs new core_events --obs-stream on every banked stdin: stdout byte-equal, per stream."""
import glob, hashlib, json, subprocess, sys, time

old, new = sys.argv[1], sys.argv[2]
files = sorted(glob.glob('/tmp/p6coord/s4/corpus/*.in'))
n_files = n_streams = n_dec = n_rows = 0
diffs = []
t_old = t_new = 0.0
digest = hashlib.sha256()
for f in files:
    data = open(f, 'rb').read()
    t0 = time.monotonic(); a = subprocess.run([old, '--obs-stream'], input=data, capture_output=True); t1 = time.monotonic()
    b = subprocess.run([new, '--obs-stream'], input=data, capture_output=True); t2 = time.monotonic()
    t_old += t1 - t0; t_new += t2 - t1
    n_files += 1
    if a.returncode != b.returncode or a.stdout != b.stdout:
        la, lb = a.stdout.splitlines(), b.stdout.splitlines()
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                diffs.append((f, i, x[:300], y[:300]))
        if len(la) != len(lb) or a.returncode != b.returncode:
            diffs.append((f, 'shape', a.returncode, b.returncode))
    digest.update(b.stdout)
    for ln in b.stdout.splitlines():
        r = json.loads(ln); n_streams += 1
        if r['ok']:
            n_dec += len(r['decisions']); n_rows += sum('obs' in d for d in r['decisions'])
print(json.dumps({'files': n_files, 'streams': n_streams, 'decisions': n_dec, 'rows': n_rows,
                  'differences': len(diffs), 'old_s': round(t_old, 2), 'new_s': round(t_new, 2),
                  'new_stdout_sha256': digest.hexdigest()}, indent=1))
for d in diffs[:10]:
    print(d)
