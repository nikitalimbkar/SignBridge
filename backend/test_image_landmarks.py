import cv2
import mediapipe as mp

# Dataset image
image_path = r"C:\Users\ACER\Downloads\archive (1)\asl_alphabet_train\asl_alphabet_train\A\A1.jpg"

# Load image
image = cv2.imread(image_path)

if image is None:
    print("❌ Image not found!")
    exit()

# Convert BGR → RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create MediaPipe Image
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

# Create Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

# Detect hand
with HandLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        print("✅ Hand detected!")

        print("Number of hands:", len(result.hand_landmarks))

        print("Number of landmarks:",
              len(result.hand_landmarks[0]))

        # Draw landmarks
        h, w, _ = image.shape

        for landmark in result.hand_landmarks[0]:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                image,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        cv2.imshow("A - Hand Landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("❌ No hand detected!")