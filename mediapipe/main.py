import cv2
from process_frame import ProcessFrame
from threshold import get_thresholds_beginner, get_thresholds_pro
import mediapipe as mp


cap = cv2.VideoCapture("good squats/IMG_8112 3.MOV")

fourcc = cv2.VideoWriter_fourcc(*"XVID")  # You can also use 'MP4V' for .mp4 output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))


pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Choose your mode here
thresholds = get_thresholds_pro()  # or get_thresholds_pro()

# Create a ProcessFrame object
processor = ProcessFrame(thresholds)

# Process video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame and process
    flipped_frame = cv2.flip(frame, 1)
    processed_frame, _ = processor.process(flipped_frame, pose)

    out.write(processed_frame)

    # Show the processed frame
    cv2.imshow("Processed Frame", processed_frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
