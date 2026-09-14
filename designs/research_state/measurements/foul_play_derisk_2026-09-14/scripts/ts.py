"""Line-timestamp a pipe. Foul Play's own formatter prints no timestamps
(`fp/config.py::CustomFormatter`), so per-move think time has to be read off the
wall clock here."""
import sys
import time

for line in sys.stdin:
    sys.stdout.write(f"{time.time():.3f} {line}")
    sys.stdout.flush()
