import time
import sys

class RateLimitedLogger:
    """
    A utility for throttling log output to avoid console flooding.
    """
    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.last_log_time = 0.0

    def should_log(self) -> bool:
        """Returns True if enough time has passed since the last log."""
        now = time.time()
        if now - self.last_log_time >= self.interval:
            self.last_log_time = now
            return True
        return False

    def log(self, message: str, force: bool = False, stream=sys.stderr):
        """
        Writes a message to the specified stream if the interval has passed.
        
        :param message: The string to log.
        :param force: If True, bypasses the rate limit.
        :param stream: The output stream (defaults to stderr for multi-process safety).
        """
        if force or self.should_log():
            stream.write(message)
            stream.flush()
