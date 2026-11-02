import cv2
import mediapipe as mp
import csv
import os

# ==============================
# DATASET PATH
# ==============================

DATASET_PATH = r"C:\Users\ACER\Downloads\archive (1)\asl_alphabet_train\asl_alphabet_train"

# Output CSV
OUTPUT_FILE = r"C:\Users\ACER\Documents\SignLangPro\data\landmarks.csv"


# ==============================
# MEDIAPIPE SETUP
# ==============================

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


# ==============================
# CREATE OUTPUT DIRECTORY
# ==============================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ==============================
# CSV HEADER
# ==============================

header = []

for i in range(1, 22):
    header.append(f"x{i}")
    header.append(f"y{i}")
    header.append(f"z{i}")

header.append("label")


# ==============================
# PROCESS DATASET
# ==============================

with open(
    OUTPUT_FILE,
    mode="w",
    newline=""
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow(header)

    with HandLandmarker.create_from_options(options) as landmarker:

        # Process A, B and C
        for label in ["A", "B", "C"]:

            folder_path = os.path.join(
                DATASET_PATH,
                label
            )

            print("\nProcessing:", label)

            image_files = os.listdir(folder_path)

            count = 0
            detected = 0

            for image_file in image_files:

                image_path = os.path.join(
                    folder_path,
                    image_file
                )

                image = cv2.imread(image_path)

                if image is None:
                    continue

                # Convert BGR → RGB
                rgb_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_image
                )

                # Detect landmarks
                result = landmarker.detect(mp_image)

                count += 1

                if result.hand_landmarks:

                    landmarks = result.hand_landmarks[0]

                    row = []

                    for landmark in landmarks:

                        row.append(landmark.x)
                        row.append(landmark.y)
                        row.append(landmark.z)

                    row.append(label)

                    writer.writerow(row)

                    detected += 1

                # Show progress every 100 images
                if count % 100 == 0:

                    print(
                        "Processed:",
                        count,
                        "| Hands detected:",
                        detected
                    )

            print(
                f"{label} completed: "
                f"{count} images processed, "
                f"{detected} hands detected"
            )


print("\n================================")
print("✅ Landmark extraction completed!")
print("================================")
print("CSV saved at:")
print(OUTPUT_FILE)