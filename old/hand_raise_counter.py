import time


class HandRaiseCounter:
    def __init__(self):
        self.left_hand_raised = False
        self.right_hand_raised = False
        self.left_hand_raise_start_time = None
        self.right_hand_raise_start_time = None
        self.left_hand_raise_counted = False
        self.right_hand_raise_counted = False
        self.left_hand_raise_count = 0
        self.right_hand_raise_count = 0

    def update(self, landmarks):
        current_time = time.time()

        # Extract relevant landmarks
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        # Check if left hand is raised
        left_hand_currently_raised = left_wrist.y < left_shoulder.y

        if left_hand_currently_raised:
            if not self.left_hand_raised:
                # Left hand just raised
                self.left_hand_raised = True
                self.left_hand_raise_start_time = current_time
                self.left_hand_raise_counted = False
            else:
                # Left hand still raised
                if not self.left_hand_raise_counted and (current_time - self.left_hand_raise_start_time) >= 2:
                    # Left hand has been raised for at least 2 seconds
                    self.left_hand_raise_count += 1
                    self.left_hand_raise_counted = True
        else:
            # Left hand is not raised
            self.left_hand_raised = False
            self.left_hand_raise_start_time = None
            self.left_hand_raise_counted = False

        # Check if right hand is raised
        right_hand_currently_raised = right_wrist.y < right_shoulder.y

        if right_hand_currently_raised:
            if not self.right_hand_raised:
                # Right hand just raised
                self.right_hand_raised = True
                self.right_hand_raise_start_time = current_time
                self.right_hand_raise_counted = False
            else:
                # Right hand still raised
                if not self.right_hand_raise_counted and (current_time - self.right_hand_raise_start_time) >= 2:
                    # Right hand has been raised for at least 2 seconds
                    self.right_hand_raise_count += 1
                    self.right_hand_raise_counted = True
        else:
            # Right hand is not raised
            self.right_hand_raised = False
            self.right_hand_raise_start_time = None
            self.right_hand_raise_counted = False
