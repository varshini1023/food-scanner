import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Model and external image folder
MODEL_PATH = "food_classifier_v4.keras"
IMAGE_FOLDER = r"C:\Users\MOHAN\Downloads\test case external"

CLASS_NAMES = [
    "bisibelebath",
    "chappati",
    "dosa",
    "idly",
    "jamun",
    "momos",
    "pakoda",
    "poori",
    "upma"
]

IMAGE_SIZE = (180, 180)

# Load the existing trained model
print("Loading Food Scanner model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!\n")

# Supported image formats
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

if not os.path.isdir(IMAGE_FOLDER):
    print(f"Folder not found: {IMAGE_FOLDER}")
    print("Create the external_test_images folder first.")
    raise SystemExit

image_files = sorted(
    filename for filename in os.listdir(IMAGE_FOLDER)
    if filename.lower().endswith(VALID_EXTENSIONS)
)

if not image_files:
    print("No images found in external_test_images.")
    print("Add some JPG, PNG or WEBP images and run again.")
    raise SystemExit

print("=" * 65)
print("EXTERNAL IMAGE TEST RESULTS")
print("=" * 65)

for filename in image_files:
    image_path = os.path.join(IMAGE_FOLDER, filename)

    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.array(image).astype("float32")
        image_array = np.expand_dims(image_array, axis=0)

        # Predict using the existing model
        predictions = model.predict(image_array, verbose=0)[0]

        top_indices = np.argsort(predictions)[::-1][:3]

        best_index = int(top_indices[0])
        predicted_food = CLASS_NAMES[best_index]
        confidence = float(predictions[best_index]) * 100

        print(f"\nImage: {filename}")
        print(f"Prediction: {predicted_food}")
        print(f"Confidence: {confidence:.2f}%")
        print("Top 3 predictions:")

        for index in top_indices:
            print(
                f"  {CLASS_NAMES[int(index)]:15s}"
                f" {float(predictions[index]) * 100:.2f}%"
            )

    except Exception as error:
        print(f"\nCould not process {filename}: {error}")

print("\n" + "=" * 65)
print("EXTERNAL IMAGE TEST COMPLETED")
print("=" * 65)
