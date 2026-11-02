import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)

start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            print("Camera not found")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # OpenCV uses BGR, MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp in milliseconds
        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        # Detect hand landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Draw landmarks
        if result.hand_landmarks:

            h, w, _ = frame.shape

            for hand_landmarks in result.hand_landmarks:

                for landmark in hand_landmarks:

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

        cv2.imshow(
            "Hand Detection",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()