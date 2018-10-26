import cv2
import numpy as np

from collections import deque
from typing import Tuple, Deque


def preprocess_frame(observation: np.ndarray, frame_size: Tuple[int, int]):
    # Make the image grayscale and downsample it.
    res = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(res, dsize=frame_size, interpolation=cv2.INTER_CUBIC)

    return res


def stack_frames(stacked_frames: Deque[np.ndarray], observation: np.ndarray,
                 frame_size: Tuple[int, int], stack_size: int):
    processed_frame = preprocess_frame(observation, frame_size)

    # If our stack is empty, populate it with the current observation.
    if len(stacked_frames) == 0:
        for i in range(stack_size - 1):
            stacked_frames.append(processed_frame)

    stacked_frames.append(processed_frame)
    multiple_frames_state = np.stack(
        stacked_frames, axis=2)  # (frame_size, num_frames)

    uniques = np.unique(multiple_frames_state)
    for i in list(uniques):
        multiple_frames_state[multiple_frames_state ==
                              i] = np.random.choice(256)

    return stacked_frames, multiple_frames_state
