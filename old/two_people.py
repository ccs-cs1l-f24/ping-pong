import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Model available to download here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
model_path = "pose_landmarker_lite.task"

video_source = 0

num_poses = 2
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
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

    # Convert the image and draw landmarks
    annotated_image = draw_landmarks_on_image(
        output_image.numpy_view(), detection_result)
    to_window = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Process landmarks to detect hand raising
    pose_landmarks_list = detection_result.pose_landmarks

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Check if landmarks are present
        if pose_landmarks:
            landmarks = pose_landmarks  # This is a list of landmarks

            # Extract the required landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]

            # Detect if left hand is raised
            left_hand_raised = left_wrist.y < left_shoulder.y

            # Detect if right hand is raised
            right_hand_raised = right_wrist.y < right_shoulder.y

            # Output or display the result
            if left_hand_raised and right_hand_raised:
                print(f"Person {idx + 1}: Both hands are raised.")
            elif left_hand_raised:
                print(f"Person {idx + 1}: Left hand is raised.")
            elif right_hand_raised:
                print(f"Person {idx + 1}: Right hand is raised.")
            else:
                print(f"Person {idx + 1}: No hands are raised.")


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
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(video_source)

    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Image capture failed.")
            break

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
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
