# Debug Report: Zombie Process Bottleneck (2026-05-10)

## Issue Description
Training performance (FPS) dropped dramatically from ~1000 FPS to ~490 FPS during the 4M step training session.

## Symptoms
- **FPS**: ~500 (Expected: 900-1000)
- **CPU Usage**: High Load Average (~8.0+) but significant Idle time (~40%), indicating resource contention rather than simple CPU saturation.
- **Memory/Network**: No clear bottleneck in IO.

## Root Cause: "Zombie" Training Sessions
Using `ps aux | grep Python`, we identified two concurrent instances of `train_rl_agent.py` running 8 environments each.

1. **PID 71556**: An old 1M-step run that had been "terminated" but remained active in the background.
2. **PID 72018**: The new 4M-step run.

Because both sessions were attempting to spawn 8 parallel environments and communicate with the local `pokemon-showdown` server, they were competing for context switches and server bandwidth, effectively halving the training speed.

## Solution & Prevention

### 1. Manual Cleanup
We performed a forced purge of all orphan Python processes:
```bash
ps aux | grep Python | grep -v grep | grep -v "[Current PID]" | awk '{print $2}' | xargs kill -9
```

### 2. Code-Level Fix: Signal Handlers
We added a robust `signal` handling block to `src/main/train_rl_agent.py`. This ensures that if the process is interrupted (Ctrl+C / SIGINT / SIGTERM), it captures the event, saves the current model state, and exits cleanly.

```python
import signal
def signal_handler(sig, frame):
    print("\nInterrupt received, saving model...")
    model.save(os.path.join(model_dir, "final_model_interrupted"))
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

### 3. Verification
After the cleanup, FPS immediately recovered to **913 FPS**.

## Recommendations
- Always verify no existing training sessions are running before starting a "10M" or "4M" run.
- Use `pkill -9 -f train_rl_agent.py` if a run hangs or needs to be forcefully restarted.
