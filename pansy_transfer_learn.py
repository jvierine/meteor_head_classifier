import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.config.experimental.list_physical_devices()
import cv2

# Set directory path to dataset
dataset_path = "/data1/pansy/sorted_images"  # Change this to your actual dataset path

# Load images from directories, split into training and validation sets
batch_size = 32
img_size = (256, 256)

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names from folders
class_names = train_ds.class_names
print(f"Class names: {class_names}")

# Prefetch data for performance improvement
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# Load a pretrained model without the top classifier layers
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(256, 256, 3),
    include_top=False,  # Remove classification layers
    weights="imagenet"  # Use pretrained weights
)

# Freeze the base model (optional, for feature extraction only)
base_model.trainable = False

# Add classification head
global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(class_names), activation="softmax")

# Build the final model
model = tf.keras.Sequential([
    base_model,
    global_avg_layer,
    tf.keras.layers.Dropout(0.2),  # Prevent overfitting
    prediction_layer
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

base_model.trainable = True  # Unfreeze model for fine-tuning

# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train for a few more epochs
fine_tune_epochs = 5
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs
)

model.save("image_classifier.h5")
