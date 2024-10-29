# main.py

import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from player import Player

# Model path (ensure the model file is in the same directory or provide the correct path)
model_path = "pose_landmarker_lite.task"

video_source = 0  # Default webcam
# video_source = "samples/game_long.mov"  # Default webcam

# Detection parameters
num_poses = 2  # We need to detect up to two people
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5

# Initialize players
players = {'left': Player('left'), 'right': Player('right')}


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    image_height, image_width, _ = annotated_image.shape

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Convert landmarks to protobuf format for drawing
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )

        # Draw landmarks on the image
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

        # Process landmarks to detect hand raising
        if pose_landmarks:
            landmarks = pose_landmarks

            # Determine which side the person is on based on the nose x-coordinate
            nose = landmarks[0]
            x_coord = nose.x  # Normalized x coordinate (0.0 to 1.0)
            side = 'left' if x_coord < 0.5 else 'right'

            player = players[side]
            player.update(landmarks)

    # Display counts on the left and right sides
    left_count = players['left'].hand_raise_count
    right_count = players['right'].hand_raise_count

    # Display counts on the left and right sides
    cv2.putText(
        annotated_image,
        f"Player A: {left_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        8,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated_image,
        f"Player B: {right_count}",
        (image_width - 400, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        8,
        cv2.LINE_AA,
    )

    return annotated_image


to_window = None
last_timestamp_ms = 0


def print_result(
    detection_result: vision.PoseLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int,
):
    global to_window
    global last_timestamp_ms
    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms

    # Convert the image and draw landmarks with counts
    annotated_image = draw_landmarks_on_image(
        output_image.numpy_view(), detection_result
    )
    to_window = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


# Set up the pose landmarker with specified options
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=num_poses,
    min_pose_detection_confidence=min_pose_detection_confidence,
    min_pose_presence_confidence=min_pose_presence_confidence,
    min_tracking_confidence=min_tracking_confidence,
    output_segmentation_masks=False,
    result_callback=print_result,
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    # Start capturing from the webcam
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Image capture failed.")
            break

        # Convert the frame to a MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if to_window is not None:
            cv2.imshow("MediaPipe Pose Landmark", to_window)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
