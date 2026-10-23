import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# 1. DATASET PATHS
# --------------------------------------------------

train_dir = r"C:\Users\MOHAN\OneDrive\Desktop\south indian food dataset\food_scanner_data\train"
test_dir = r"C:\Users\MOHAN\OneDrive\Desktop\south indian food dataset\food_scanner_data\test"

model_save_path = "food_classifier_v4.keras"

print("TensorFlow version:", tf.__version__)
print("Training folder:", train_dir)
print("Testing folder:", test_dir)

# --------------------------------------------------
# 2. IMAGE SETTINGS
# --------------------------------------------------

img_height = 180
img_width = 180
batch_size = 16
seed = 123

# --------------------------------------------------
# 3. CHECK DATASET FOLDERS
# --------------------------------------------------

if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"Training folder not found: {train_dir}")

if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Testing folder not found: {test_dir}")

expected_classes = [
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

train_classes = sorted([
    name for name in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, name))
])

test_classes = sorted([
    name for name in os.listdir(test_dir)
    if os.path.isdir(os.path.join(test_dir, name))
])

print("\nTraining folders:", train_classes)
print("Testing folders:", test_classes)

if train_classes != expected_classes:
    raise ValueError(
        f"Training classes do not match.\n"
        f"Expected: {expected_classes}\n"
        f"Found: {train_classes}"
    )

if test_classes != expected_classes:
    raise ValueError(
        f"Testing classes do not match.\n"
        f"Expected: {expected_classes}\n"
        f"Found: {test_classes}"
    )

# --------------------------------------------------
# 4. LOAD TRAINING DATA
# --------------------------------------------------

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    class_names=expected_classes,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="int"
)

# --------------------------------------------------
# 5. LOAD VALIDATION DATA
# --------------------------------------------------

validation_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    class_names=expected_classes,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="int"
)

# --------------------------------------------------
# 6. LOAD TEST DATA
# --------------------------------------------------

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    class_names=expected_classes,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    label_mode="int"
)

class_names = train_data.class_names

print("\nClass names:", class_names)
print("Number of classes:", len(class_names))

# --------------------------------------------------
# 7. IMPROVE DATA LOADING PERFORMANCE
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
validation_data = validation_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# --------------------------------------------------
# 8. DATA AUGMENTATION
# --------------------------------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
], name="data_augmentation")

# --------------------------------------------------
# 9. LOAD MOBILENETV2
# --------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers for initial training.
base_model.trainable = False

# --------------------------------------------------
# 10. BUILD MODEL
# --------------------------------------------------

inputs = tf.keras.Input(shape=(img_height, img_width, 3))

x = data_augmentation(inputs)

# MobileNetV2 expects pixel values scaled to [-1, 1].
x = tf.keras.layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# --------------------------------------------------
# 11. COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 12. CALLBACKS
# --------------------------------------------------

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_save_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

# --------------------------------------------------
# 13. PHASE 1: TRAIN CLASSIFICATION LAYERS
# --------------------------------------------------

print("\nPHASE 1: Training classification layers...")

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=15,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback
    ]
)

# --------------------------------------------------
# 14. PHASE 2: FINE-TUNE MOBILENETV2
# --------------------------------------------------

print("\nPHASE 2: Fine-tuning MobileNetV2...")

# Load the best model from phase 1.
model = tf.keras.models.load_model(model_save_path)

# Retrieve the actual nested MobileNetV2 layer.
base_model = model.get_layer("mobilenetv2_1.00_224")

# Unfreeze only the last 30 layers.
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Recompile with a low learning rate.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Use a fresh checkpoint callback for fine-tuning.
fine_tune_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_save_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

fine_tune_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=8,
    callbacks=[
        fine_tune_checkpoint,
        fine_tune_early_stopping
    ]
)

# --------------------------------------------------
# 15. LOAD BEST SAVED MODEL
# --------------------------------------------------

if not os.path.exists(model_save_path):
    raise FileNotFoundError("The trained model was not saved.")

model = tf.keras.models.load_model(model_save_path)

print("\nBest saved model loaded successfully.")

# --------------------------------------------------
# 16. EVALUATE ON TEST DATA
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(test_data, verbose=1)

print("\n------------------------------")
print("FINAL TEST RESULTS")
print("------------------------------")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# --------------------------------------------------
# 17. GENERATE PREDICTIONS
# --------------------------------------------------

true_labels = np.concatenate([
    labels.numpy()
    for _, labels in test_data
])

predictions = model.predict(test_data, verbose=1)

predicted_labels = np.argmax(predictions, axis=1)

# --------------------------------------------------
# 18. PRECISION, RECALL AND F1-SCORE
# --------------------------------------------------

print("\nCLASSIFICATION REPORT")

print(classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    digits=3
))

# --------------------------------------------------
# 19. CONFUSION MATRIX
# --------------------------------------------------

print("\nCONFUSION MATRIX")
print("Rows = actual class")
print("Columns = predicted class")

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names)))
)

print("\nClass order:", class_names)
print(matrix)

# --------------------------------------------------
# 20. SAVE CLASS NAMES
# --------------------------------------------------

with open("food_class_names.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(class_names))

print("\nClass names saved to food_class_names.txt")
print("New model saved as:", model_save_path)
print("TRAINING AND EVALUATION COMPLETED!")

