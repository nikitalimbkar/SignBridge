import os
import cv2
import csv
import random
import numpy as np
import mediapipe as mp
import time

# ============================================================
# SETTINGS
# ============================================================

TRAIN_PATH = r"C:\Users\ACER\Downloads\archive (new)\asl_alphabet_train\asl_alphabet_train"

OUTPUT_CSV = "training_landmarks.csv"
PROGRESS_FILE = "dataset_progress.txt"

IMAGES_PER_CLASS = 1000

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
    "SPACE",
    "NOTHING",
    "DEL"
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

SAVE_EVERY = 100

# Resize large images before MediaPipe
MAX_IMAGE_SIZE = 640

# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.2
)


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(hand_landmarks):

    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    if coords.shape != (21, 3):
        return None

    # Make wrist the origin
    wrist = coords[0].copy()
    coords = coords - wrist

    # Scale normalization
    distances = np.linalg.norm(coords, axis=1)
    scale = np.max(distances)

    if scale < 1e-6:
        return None

    coords = coords / scale

    return coords.flatten().tolist()


# ============================================================
# GET LANDMARKS
# ============================================================

def get_landmarks(image_path, landmarker):

    try:

        image = cv2.imread(image_path)

        if image is None:
            return None

        # Resize large images
        h, w = image.shape[:2]

        if max(h, w) > MAX_IMAGE_SIZE:

            scale = MAX_IMAGE_SIZE / max(h, w)

            new_w = int(w * scale)
            new_h = int(h * scale)

            image = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

        # BGR -> RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]

        return normalize_landmarks(hand_landmarks)

    except Exception as e:

        print()
        print("Error:", image_path)
        print("Reason:", e)

        return None


# ============================================================
# CSV HEADER
# ============================================================

def get_csv_header():

    columns = []

    for i in range(1, 22):

        columns.append(f"x{i}")
        columns.append(f"y{i}")
        columns.append(f"z{i}")

    columns.append("label")

    return columns


# ============================================================
# LOAD EXISTING CSV
# ============================================================

def load_existing_data():

    rows = []

    if not os.path.exists(OUTPUT_CSV):

        return rows

    print()
    print("Existing CSV found.")
    print("Loading previous data...")

    try:

        with open(
            OUTPUT_CSV,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.reader(file)

            header = next(reader, None)

            for row in reader:

                if len(row) == 64:

                    rows.append(row)

        print(f"Existing usable rows: {len(rows)}")

    except Exception as e:

        print("Could not read CSV:", e)

    return rows


# ============================================================
# LOAD PROGRESS
# ============================================================

def load_progress():

    processed_images = set()

    if not os.path.exists(PROGRESS_FILE):

        return processed_images

    try:

        with open(
            PROGRESS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if line:

                    processed_images.add(line)

    except Exception as e:

        print("Could not load progress:", e)

    return processed_images


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_progress(processed_images):

    temp_file = PROGRESS_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:

        for image_path in sorted(processed_images):

            file.write(image_path + "\n")

    os.replace(
        temp_file,
        PROGRESS_FILE
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(rows):

    temp_file = OUTPUT_CSV + ".tmp"

    with open(
        temp_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            get_csv_header()
        )

        writer.writerows(rows)

    os.replace(
        temp_file,
        OUTPUT_CSV
    )


# ============================================================
# MAIN
# ============================================================

def create_training_dataset():

    print()
    print("================================================")
    print("       FAST ASL 29-CLASS DATASET")
    print("================================================")

    print()
    print(f"Images per class: {IMAGES_PER_CLASS}")
    print(f"Maximum images: {IMAGES_PER_CLASS * 29}")

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not os.path.exists(TRAIN_PATH):

        print()
        print("ERROR: Dataset path not found!")
        print(TRAIN_PATH)

        return

    # --------------------------------------------------------
    # FIND CLASSES
    # --------------------------------------------------------

    available_classes = []

    for class_name in CLASSES:

        folder_path = os.path.join(
            TRAIN_PATH,
            class_name
        )

        if os.path.isdir(folder_path):

            available_classes.append(
                class_name
            )

    print()
    print("Classes found:")

    print(
        available_classes
    )

    print()
    print(
        f"Classes found: "
        f"{len(available_classes)}/29"
    )

    # --------------------------------------------------------
    # LOAD OLD DATA
    # --------------------------------------------------------

    rows = load_existing_data()

    processed_images = load_progress()

    print()
    print(
        f"Previously processed images: "
        f"{len(processed_images)}"
    )

    print()
    print("Starting dataset extraction...")

    start_time = time.time()

    total_new = 0
    total_no_hand = 0
    total_skipped = 0

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    try:

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            # =================================================
            # EACH CLASS
            # =================================================

            for class_number, class_name in enumerate(
                available_classes,
                start=1
            ):

                print()
                print("------------------------------------------------")
                print(
                    f"[{class_number}/"
                    f"{len(available_classes)}] "
                    f"CLASS: {class_name}"
                )
                print("------------------------------------------------")

                folder_path = os.path.join(
                    TRAIN_PATH,
                    class_name
                )

                # ------------------------------------------------
                # GET IMAGE FILES
                # ------------------------------------------------

                image_files = [

                    filename

                    for filename in os.listdir(
                        folder_path
                    )

                    if filename.lower().endswith(
                        IMAGE_EXTENSIONS
                    )

                ]

                # ------------------------------------------------
                # RANDOMLY SELECT 1000
                # ------------------------------------------------

                random.seed(42)

                random.shuffle(
                    image_files
                )

                image_files = image_files[
                    :IMAGES_PER_CLASS
                ]

                total_images = len(
                    image_files
                )

                print(
                    f"Images selected: "
                    f"{total_images}"
                )

                class_new = 0
                class_no_hand = 0
                class_skipped = 0

                # ------------------------------------------------
                # PROCESS
                # ------------------------------------------------

                for image_number, filename in enumerate(
                    image_files,
                    start=1
                ):

                    image_path = os.path.abspath(
                        os.path.join(
                            folder_path,
                            filename
                        )
                    )

                    # Already processed
                    if image_path in processed_images:

                        class_skipped += 1
                        total_skipped += 1

                        continue

                    # Extract landmarks
                    features = get_landmarks(
                        image_path,
                        landmarker
                    )

                    # Mark processed
                    processed_images.add(
                        image_path
                    )

                    # ------------------------------------------------
                    # NO HAND
                    # ------------------------------------------------

                    if features is None:

                        class_no_hand += 1
                        total_no_hand += 1

                    # ------------------------------------------------
                    # SUCCESS
                    # ------------------------------------------------

                    else:

                        rows.append(
                            features + [class_name]
                        )

                        class_new += 1
                        total_new += 1

                    # ------------------------------------------------
                    # SAVE EVERY 100
                    # ------------------------------------------------

                    if (
                        len(processed_images)
                        % SAVE_EVERY
                        == 0
                    ):

                        save_csv(rows)

                        save_progress(
                            processed_images
                        )

                    # ------------------------------------------------
                    # PROGRESS
                    # ------------------------------------------------

                    if (
                        image_number % 100
                        == 0
                    ):

                        elapsed = (
                            time.time()
                            - start_time
                        )

                        print(
                            f"    "
                            f"{image_number}/"
                            f"{total_images} | "
                            f"New: {class_new} | "
                            f"No hand: {class_no_hand} | "
                            f"Skipped: {class_skipped} | "
                            f"Time: "
                            f"{elapsed / 60:.1f} min"
                        )

                # ------------------------------------------------
                # CLASS COMPLETE
                # ------------------------------------------------

                print()
                print(
                    f"Finished {class_name}"
                )

                print(
                    f"  Usable: "
                    f"{class_new}"
                )

                print(
                    f"  No hand: "
                    f"{class_no_hand}"
                )

                print(
                    f"  Already done: "
                    f"{class_skipped}"
                )

                # Save after every class
                save_csv(rows)

                save_progress(
                    processed_images
                )

    # =========================================================
    # CTRL + C
    # =========================================================

    except KeyboardInterrupt:

        print()
        print()
        print("================================================")
        print("       EXTRACTION STOPPED")
        print("================================================")

        print()
        print("Saving current progress...")

        save_csv(rows)

        save_progress(
            processed_images
        )

        print()
        print(
            f"Usable rows saved: "
            f"{len(rows)}"
        )

        print(
            f"Images processed: "
            f"{len(processed_images)}"
        )

        print()
        print(
            "You can run the script again."
        )

        print(
            "It will resume from the saved progress."
        )

        return

    # =========================================================
    # FINAL SAVE
    # =========================================================

    save_csv(rows)

    save_progress(
        processed_images
    )

    elapsed = time.time() - start_time

    # =========================================================
    # FINAL RESULT
    # =========================================================

    print()
    print("================================================")
    print("       DATASET EXTRACTION COMPLETE")
    print("================================================")

    print()
    print(
        f"Total usable rows : "
        f"{len(rows)}"
    )

    print(
        f"New rows this run : "
        f"{total_new}"
    )

    print(
        f"Already processed : "
        f"{total_skipped}"
    )

    print(
        f"No hand detected  : "
        f"{total_no_hand}"
    )

    print(
        f"Time              : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print(
        "CSV saved as:"
    )

    print(
        "training_landmarks.csv"
    )

    print()
    print("Next step:")
    print(
        "python train_model.py"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    create_training_dataset()   