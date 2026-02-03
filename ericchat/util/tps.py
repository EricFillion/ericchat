import time
from collections import deque


class TPSTracker:
    def __init__(self, window_seconds: float = 5.0):
        self.window = window_seconds
        self.total = 0
        self.events = deque()  # timestamps only

    def step(self) -> float:

        self.total += 1

        if self.total <= 4: # The first few tokens from gpt-oss are hardcoded and come fast so we don't want to count them.
            return 0.0
        t = time.monotonic()
        self.events.append(t)

        cutoff = t - self.window
        while self.events and self.events[0] < cutoff:
            self.events.popleft()  # no unpacking

        n = len(self.events)
        if n <= 5: # start reporting after 5 tokens
            return 0.0

        span = max(1e-6, self.events[-1] - self.events[0])
        return (n - 1) / span   # better estimator than n/span

    def reset(self) -> None:
        self.events.clear()
        self.total =0
