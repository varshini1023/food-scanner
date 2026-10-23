import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("food_classifier.keras")

class_names = [
    "bisibelebath",
    "chappati",
    "dosa",
    "idly",
    "poori",
    "upma"
]

# Minimum confidence required
CONFIDENCE_THRESHOLD = 70

image_path = input("Enter image path: ")

img = tf.keras.utils.load_img(
    image_path,
    target_size=(180, 180)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array, verbose=0)

predicted_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

predicted_class = class_names[predicted_index]

# NOT RECOGNIZED CHECK
if confidence < CONFIDENCE_THRESHOLD:
    print("\n❌ Food Not Recognized")
    print("Confidence:", round(confidence, 2), "%")
else:
    print("\nPredicted food:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")