# player.py

import time

HAND_RAISE_DURATION_THRESHOLD = 1


class Player:
    def __init__(self, side):
        self.side = side  # 'left' or 'right'
        self.hand_raised = False
        self.hand_raise_start_time = None
        self.hand_raise_counted = False
        self.hand_raise_count = 0

    def update(self, landmarks):
        current_time = time.time()

        # Extract relevant landmarks
        left_shoulder = landmarks[11]
        left_wrist = landmarks[15]

        # Check if left hand is raised
        hand_currently_raised = left_wrist.y < left_shoulder.y

        if hand_currently_raised:
            if not self.hand_raised:
                # Left hand just raised
                self.hand_raised = True
                self.hand_raise_start_time = current_time
                self.hand_raise_counted = False
            else:
                # Left hand still raised
                if not self.hand_raise_counted and (current_time - self.hand_raise_start_time) >= HAND_RAISE_DURATION_THRESHOLD:
                    # Left hand has been raised long enough
                    self.hand_raise_count += 1
                    self.hand_raise_counted = True
        else:
            # Left hand is not raised
            self.hand_raised = False
            self.hand_raise_start_time = None
            self.hand_raise_counted = False
