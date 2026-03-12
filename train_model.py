import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# -----------------------------
# DATASET PATHS
# -----------------------------
train_dir = r"C:\Users\ELCOT\mypro\dataset\train"
val_dir   = r"C:\Users\ELCOT\mypro\dataset\val"

if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError("Train/Val folders not found.")

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
IMG_SIZE = 128
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7,1.3],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"
)

# -----------------------------
# BUILD MODEL (TRANSFER LEARNING)
# -----------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze backbone
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(train_data.num_classes, activation="softmax")(x)

model = models.Model(base_model.input, output)

# -----------------------------
# PHASE 1 TRAINING
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Phase 1 Training Started...")
model.fit(train_data, validation_data=val_data, epochs=5)

# -----------------------------
# PHASE 2 FINE TUNING
# -----------------------------
print("Fine Tuning Started...")

# Unfreeze last 40 layers
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=val_data, epochs=10)

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("model", exist_ok=True)
model_path = "model/Leaf_Disease_128x128.h5"
model.save(model_path)

print(f"Model saved successfully at {model_path}")