import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Dataset and model paths

test_dir = r"C:\Users\MOHAN\OneDrive\Desktop\south indian food dataset\food_scanner_data\test"
model_path = r"C:\Users\MOHAN\OneDrive\Desktop\food scanner\food_classifier_v4.keras"

# Food class order

class_names = [
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

# Check that required files exist

if not os.path.isdir(test_dir):
  raise FileNotFoundError(f"Test folder not found: {test_dir}")

if not os.path.isfile(model_path):
  raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the existing model

print("Loading existing model...")
model = tf.keras.models.load_model(model_path)

# Load test images without shuffling

print("Loading test dataset...")
test_data = tf.keras.utils.image_dataset_from_directory(
test_dir,
class_names=class_names,
image_size=(180, 180),
batch_size=16,
shuffle=False,
label_mode="int"
)

# Confirm the model and class counts match

if model.output_shape[-1] != len(class_names):
  raise ValueError(
f"Model outputs {model.output_shape[-1]} classes, "
f"but {len(class_names)} class names were supplied."
)

# Evaluate the model

print("\nEvaluating existing model...")
loss, accuracy = model.evaluate(test_data, verbose=1)

print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Collect actual labels

true_labels = np.concatenate([
labels.numpy() for _, labels in test_data
])

# Generate predictions

print("\nGenerating predictions...")
probabilities = model.predict(test_data, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

# Print class-by-class performance

print("\nCLASSIFICATION REPORT")
print(classification_report(
true_labels,
predicted_labels,
labels=list(range(len(class_names))),
target_names=class_names,
zero_division=0,
digits=3
))

# Print confusion matrix

matrix = confusion_matrix(
true_labels,
predicted_labels,
labels=list(range(len(class_names)))
)

print("\nCONFUSION MATRIX")
print("Rows = actual food; columns = predicted food")
print("Class order:", class_names)
print(matrix)
