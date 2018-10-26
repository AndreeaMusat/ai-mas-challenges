import numpy as np

from collections import deque
from typing import Tuple


class Experience:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Tuple[np.ndarray, int, int, np.ndarray]):
        """
        experience should have the following form:
        (state, action, reward, next_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        selected = np.random.choice(np.arange(len(self.buffer)),
                                    size=batch_size, replace=False)
        return [self.buffer[idx] for idx in selected]
