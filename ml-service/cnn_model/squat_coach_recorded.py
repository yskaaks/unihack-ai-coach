import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained CNN model
cnn_model = load_model("cnn_model_v1.h5")


# Function to preprocess the frame for the CNN model
def preprocess_frame_for_cnn(frame):
    # Resize the frame to match the CNN input
    resized_frame = cv2.resize(frame, (200, 200))
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0
    # Expand dimensions
    return np.expand_dims(normalized_frame, axis=0)


def extract_roi_from_landmarks(landmarks, frame):
    # Define the indices for the landmarks you want to focus on
    # These should be consistent with what your model has been trained on
    # For example, using hip, knee, and ankle points
    hip_index = mp_pose.PoseLandmark.LEFT_HIP.value
    knee_index = mp_pose.PoseLandmark.LEFT_KNEE.value
    ankle_index = mp_pose.PoseLandmark.LEFT_ANKLE.value

    # Extract the coordinates
    hip = [landmarks[hip_index].x, landmarks[hip_index].y]
    knee = [landmarks[knee_index].x, landmarks[knee_index].y]
    ankle = [landmarks[ankle_index].x, landmarks[ankle_index].y]

    # Convert from relative coordinates to pixel values
    frame_height, frame_width, _ = frame.shape
    hip = [int(hip[0] * frame_width), int(hip[1] * frame_height)]
    knee = [int(knee[0] * frame_width), int(knee[1] * frame_height)]
    ankle = [int(ankle[0] * frame_width), int(ankle[1] * frame_height)]

    # Calculate the bounding box for the ROI
    # This can be adjusted based on how much of the surrounding area you want to include
    x_min = max(min(hip[0], knee[0], ankle[0]) - 50, 0)
    x_max = min(max(hip[0], knee[0], ankle[0]) + 50, frame_width)
    y_min = max(min(hip[1], knee[1], ankle[1]) - 50, 0)
    y_max = min(max(hip[1], knee[1], ankle[1]) + 50, frame_height)

    # Ensure valid bounding box dimensions
    if x_max <= x_min or y_max <= y_min:
        return None  # Invalid ROI, return None

    # Extract the ROI
    roi = frame[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return None  # Empty ROI, return None

    # Resize the ROI to match the input size of the CNN model
    roi_resized = cv2.resize(roi, (200, 200))

    return roi_resized


# Function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_detailed_feedback(form_feedback):
    feedback_dict = {
        "good_form": "Good Form",
        "knees_caving_in": "Keep knees aligned",
        "hip_misalignment": "Align your hips",
        # Add other form faults and their feedback
    }
    return feedback_dict.get(form_feedback, "Adjust Form")


# 3. Function for Squat Counter
def count_squats(landmarks, squat_counter, squat_stage, image):
    # Use landmarks for LEFT_HIP, LEFT_KNEE, LEFT_ANKLE (or right side)
    hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
    ]
    knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
    ]
    ankle = [
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
    ]

    # Calculate angle
    angle = calculate_angle(hip, knee, ankle)

    # Squat counting logic
    if angle > 160:
        squat_stage = "up"
    elif angle < 90 and squat_stage == "up":
        squat_stage = "down"
        squat_counter += 1

    roi = extract_roi_from_landmarks(landmarks, image)

    # Check if ROI is valid before proceeding
    if roi is not None:
        preprocessed_frame = preprocess_frame_for_cnn(roi)
        prediction = cnn_model.predict(preprocessed_frame)

        # Assuming '0' is the index for "Good Form" and '1' is for "Adjust Form"
        feedback = "Good Form" if np.argmax(prediction) == 0 else "Adjust Form"
    else:
        feedback = None  # No valid ROI, no feedback

    return squat_counter, squat_stage, angle, feedback


# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize counter and stage variables
squat_count, squat_stage = 0, None

# Path to your video file
video_path = "videos/squats.mov"
output_dir = "squat_cnn_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cap = cv2.VideoCapture(video_path)

# Get input video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change XVID to mp4v for MP4 files
output_path = os.path.join(
    output_dir, "squats_output.mp4"
)  # Ensure the correct file extension
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Setup MediaPipe Pose instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Resize frame for faster processing
        # frame = cv2.resize(frame, (640, 480))  # Change resolution as needed

        if not ret:
            break  # End of video
        # Convert frame color to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process frame with MediaPipe Pose
        results = pose.process(image)

        # Convert frame color back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            squat_count, squat_stage, squat_angle, form_feedback = count_squats(
                landmarks, squat_count, squat_stage, image
            )

            # Visualize angle, counter, and feedback for squats
        cv2.putText(
            image,
            f"Squat Angle: {squat_angle}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Squats: {squat_count}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if form_feedback:
            feedback_msg = get_detailed_feedback(form_feedback)
            cv2.putText(
                image,
                f"Feedback: {feedback_msg}",
                (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        out.write(image)

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
