import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# PATHS

CSV_FILE = "training_landmarks_clean.csv"

MODEL_FILE = "sign_model.pkl"

# ALL 29 CLASSES

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
    "SPACE",
    "NOTHING",
    "DEL"
]

# START

print()
print("==============================================")
print("       ASL 29-CLASS MODEL TRAINING")
print("==============================================")

# CHECK CSV

if not os.path.exists(CSV_FILE):

    print()
    print("ERROR:")
    print(f"{CSV_FILE} not found.")
    print()
    print("First run:")
    print("python create_dataset.py")
    print()

    exit()

# LOAD DATA

df = pd.read_csv(
    CSV_FILE
)

print()
print("Dataset loaded successfully.")

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)

# FEATURES + LABEL

feature_columns = [
    column
    for column in df.columns
    if column != "label"
]

X = df[feature_columns]

y = df["label"]

# CHECK FEATURES

print()
print(
    f"Number of features: "
    f"{len(feature_columns)}"
)

if len(feature_columns) != 63:

    print()
    print("ERROR!")
    print(
        "Expected exactly 63 features."
    )

    print(
        f"Found {len(feature_columns)}."
    )

    exit()

# CHECK CLASSES

classes_found = sorted(
    y.unique()
)

print()
print("Classes found:")
print(classes_found)

print()
print(
    f"Number of classes: "
    f"{len(classes_found)}"
)

# CHECK MISSING CLASSES

missing_classes = [
    class_name
    for class_name in CLASSES
    if class_name not in classes_found
]

if missing_classes:

    print()
    print("WARNING!")
    print("These classes are missing:")

    for class_name in missing_classes:

        print(
            f"  - {class_name}"
        )

    print()

    print(
        "This will NOT be a complete 29-class model."
    )

    print()



# CLASS DISTRIBUTION


print()
print("==============================================")
print("           CLASS DISTRIBUTION")
print("==============================================")

class_counts = (
    y.value_counts()
    .sort_index()
)

for class_name in CLASSES:

    count = class_counts.get(
        class_name,
        0
    )

    print(
        f"{class_name:8} : {count}"
    )



# TRAIN / VALIDATION SPLIT


print()
print("==============================================")
print("           TRAIN / VALIDATION")
print("==============================================")


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print()
print(
    f"Training samples   : "
    f"{len(X_train)}"
)

print(
    f"Validation samples : "
    f"{len(X_test)}"
)



# RANDOM FOREST


print()
print("==============================================")
print("          TRAINING RANDOM FOREST")
print("==============================================")

print()
print("Please wait...")


model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


model.fit(
    X_train,
    y_train
)


print()
print("Training completed.")


# PREDICTION


predictions = model.predict(
    X_test
)



# ACCURACY


accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("==============================================")
print("              MODEL ACCURACY")
print("==============================================")

print()
print(
    f"Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)



# CLASSIFICATION REPORT


print()
print("==============================================")
print("          CLASSIFICATION REPORT")
print("==============================================")


print(
    classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        zero_division=0
    )
)



# CONFUSION MATRIX


print()
print("==============================================")
print("             CONFUSION MATRIX")
print("==============================================")


matrix = confusion_matrix(

    y_test,

    predictions,

    labels=CLASSES
)


# Print header
print()

print(
    "       " +
    " ".join(
        f"{letter:>3}"
        for letter in CLASSES
    )
)


# Print rows
for class_name, row in zip(
    CLASSES,
    matrix
):

    print(
        f"{class_name:>7} " +
        " ".join(
            f"{value:>3}"
            for value in row
        )
    )



# SAVE MODEL


joblib.dump(
    model,
    MODEL_FILE
)


print()
print("==============================================")
print("              MODEL SAVED")
print("==============================================")

print()
print(
    f"Saved as: {MODEL_FILE}"
)

print()
print("Model classes:")

print(
    list(model.classes_)
)

print()
print(
    "Expected: 29 classes"
)

print()
print("==============================================")
print("        TRAINING FINISHED SUCCESSFULLY")
print("==============================================")

print()
print("Next:")
print("python webcam_prediction.py")
print()