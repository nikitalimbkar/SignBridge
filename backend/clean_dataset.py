import csv
import os

INPUT_CSV = "training_landmarks.csv"
OUTPUT_CSV = "training_landmarks_clean.csv"

VALID_CLASSES = set(
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + ["SPACE", "NOTHING", "DEL"]
)

LABEL_MAP = {
    "del": "DEL",
    "space": "SPACE",
    "nothing": "NOTHING"
}


def clean_dataset():

    print()
    print("==============================================")
    print("       CLEANING ASL DATASET")
    print("==============================================")

    if not os.path.exists(INPUT_CSV):
        print("ERROR: training_landmarks.csv not found!")
        return

    rows = []
    removed = 0
    converted = 0

    with open(
        INPUT_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.reader(file)

        header = next(reader)

        for row in reader:

            if len(row) != 64:
                removed += 1
                continue

            label = row[-1]

            # Convert lowercase duplicate labels
            if label in LABEL_MAP:
                row[-1] = LABEL_MAP[label]
                converted += 1

            # Keep only the required 29 classes
            if row[-1] not in VALID_CLASSES:
                removed += 1
                continue

            rows.append(row)

    # Remove the extremely small NOTHING class
    nothing_rows = [
        row for row in rows
        if row[-1] == "NOTHING"
    ]

    if len(nothing_rows) < 2:

        print()
        print(
            "WARNING: NOTHING has fewer than 2 samples."
        )

        rows = [
            row
            for row in rows
            if row[-1] != "NOTHING"
        ]

        print(
            "NOTHING class removed."
        )

    # Save cleaned dataset
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)
        writer.writerows(rows)

    # Count classes
    counts = {}

    for row in rows:

        label = row[-1]

        counts[label] = counts.get(
            label,
            0
        ) + 1

    print()
    print("==============================================")
    print("           CLEANING COMPLETE")
    print("==============================================")

    print()
    print(f"Original rows      : {sum(counts.values()) + removed}")
    print(f"Clean rows         : {len(rows)}")
    print(f"Removed rows       : {removed}")
    print(f"Converted labels   : {converted}")
    print()
    print("CLASS DISTRIBUTION")
    print("----------------------------------------------")

    for label in sorted(counts):

        print(
            f"{label:8} : {counts[label]}"
        )

    print()
    print(
        f"Number of classes: {len(counts)}"
    )

    print()
    print(
        f"Clean CSV saved as:"
    )

    print(
        OUTPUT_CSV
    )

    print()


if __name__ == "__main__":
    clean_dataset()