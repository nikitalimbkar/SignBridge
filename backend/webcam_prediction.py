import cv2
import numpy as np
import joblib
import mediapipe as mp



MODEL_PATH = "sign_model.pkl"

CONFIDENCE_THRESHOLD = 0.25


# LOAD MODEL


print()
print("==============================================")
print("       ASL WEBCAM PREDICTION")
print("==============================================")

model = joblib.load(MODEL_PATH)

print()
print("Model loaded successfully.")

print("Classes:")
print(model.classes_)


# MEDIAPIPE


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2
)


# NORMALIZE LANDMARKS


def normalize_landmarks(hand_landmarks):

    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    if coords.shape != (21, 3):
        return None

    # Wrist becomes origin
    wrist = coords[0].copy()

    coords = coords - wrist

    # Scale normalization
    distances = np.linalg.norm(
        coords,
        axis=1
    )

    scale = np.max(distances)

    if scale < 1e-6:
        return None

    coords = coords / scale

    return coords.flatten()



# CAMERA


cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print()
    print("ERROR: Could not open webcam.")

    exit()

print()
print("Webcam started.")
print()
print("Show your hand to the camera.")
print("Press Q to quit.")
print()


# MEDIAPIPE


with HandLandmarker.create_from_options(
    options
) as landmarker:

    frame_timestamp = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Could not read webcam frame.")

            break

        
        # ORIGINAL FRAME FOR AI
        

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        frame_timestamp += 1

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        prediction = "UNKNOWN"
        confidence = 0.0

        
        # HAND DETECTED
        

        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]

            features = normalize_landmarks(
                hand_landmarks
            )

            if features is not None:

                # Use DataFrame with feature names
                import pandas as pd

                feature_names = []

                for i in range(1, 22):

                    feature_names.append(
                        f"x{i}"
                    )

                    feature_names.append(
                        f"y{i}"
                    )

                    feature_names.append(
                        f"z{i}"
                    )

                X = pd.DataFrame(
                    [features],
                    columns=feature_names
                )

                
                # PREDICTION
                

                probabilities = model.predict_proba(X)[0]

                best_index = np.argmax(
                    probabilities
                )

                confidence = float(
                    probabilities[best_index]
                )

                prediction = model.classes_[
                    best_index
                ]

                
                # CONFIDENCE CHECK
                

                if confidence < CONFIDENCE_THRESHOLD:

                    prediction = "UNKNOWN"
        else:
            prediction = "NOTHING"
            confidence = 1.0
        
        # DISPLAY
        

        display_frame = cv2.flip(
            frame,
            1
        )

        # Prediction text
        cv2.putText(
            display_frame,
            f"Prediction: {prediction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Confidence
        # cv2.putText(
        #     display_frame,
        #     f"Confidence: {confidence * 100:.1f}%",
        #     (20, 80),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (255, 255, 255),
        #     2
        # )

        # Hand detected
        if result.hand_landmarks:

            cv2.putText(
                display_frame,
                "HAND DETECTED",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display_frame,
                "NO HAND",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "ASL Sign Language Recognition",
            display_frame
        )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


cap.release()

cv2.destroyAllWindows()

print()
print("==============================================")
print("Webcam prediction stopped.")
print("==============================================")