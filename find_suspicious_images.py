import os
import csv
import numpy as np
import tensorflow as tf
from PIL import Image

TRAIN_DIR = (
    r"C:\Users\MOHAN\OneDrive\Desktop"
    r"\south indian food dataset\food_scanner_data\train"
)

MODEL_PATH = "food_classifier_v4.keras"
REPORT_PATH = "suspicious_images_report.csv"

CLASS_NAMES = [
    "bisibelebath", "chappati", "dosa",
    "idly", "jamun", "momos",
    "pakoda", "poori", "upma"
]

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 70.0

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: Training folder not found: {TRAIN_DIR}")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    results = []
    total_scanned = 0

    for actual_class in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, actual_class)

        if not os.path.isdir(class_dir):
            print(f"Missing folder: {actual_class}")
            continue

        paths = [
            os.path.join(class_dir, name)
            for name in sorted(os.listdir(class_dir))
            if name.lower().endswith(VALID_EXTENSIONS)
        ]

        print(f"\nChecking {actual_class}: {len(paths)} images")

        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            images = []
            valid_paths = []

            for path in batch_paths:
                try:
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        image = image.resize(IMAGE_SIZE)
                        images.append(np.asarray(image, dtype=np.float32))
                        valid_paths.append(path)
                except Exception as error:
                    results.append([
                        actual_class, "", "", "IMAGE_ERROR", path
                    ])
                    print(f"Unreadable image: {path} ({error})")

            if not images:
                continue

            batch = np.stack(images, axis=0)
            predictions = model.predict(batch, verbose=0)

            for path, prediction in zip(valid_paths, predictions):
                total_scanned += 1
                index = int(np.argmax(prediction))
                predicted_class = CLASS_NAMES[index]
                confidence = float(prediction[index] * 100)

                if predicted_class != actual_class:
                    issue = "PREDICTION_DIFFERS_FROM_FOLDER"
                elif confidence < CONFIDENCE_THRESHOLD:
                    issue = "LOW_CONFIDENCE"
                else:
                    continue

                results.append([
                    actual_class,
                    predicted_class,
                    round(confidence, 2),
                    issue,
                    path
                ])

            print(
                f"  Processed {min(start + BATCH_SIZE, len(paths))}"
                f"/{len(paths)} images"
            )

    with open(REPORT_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Actual folder",
            "Model prediction",
            "Confidence percent",
            "Issue",
            "Image path"
        ])
        writer.writerows(results)

    print("\n========== SCAN COMPLETE ==========")
    print(f"Images scanned: {total_scanned}")
    print(f"Images flagged: {len(results)}")
    print(f"Report: {os.path.abspath(REPORT_PATH)}")
    print("No images were deleted or modified.")


if __name__ == "__main__":
    main()
