import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = 24
BATCH_SIZE = 32
EPOCHS = 15
DATASET_PATH = 'dataset'

# Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load datasets
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# ✅ REVERSE class labels manually in generators
# Create a wrapper to flip the labels
def reverse_labels(generator):
    for batch_x, batch_y in generator:
        yield batch_x, 1 - batch_y  # Flip 0↔1

# Print original class mapping
print("🔄 Reversing Labels — Original Mapping:", train_data.class_indices)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output: 0 = open, 1 = closed (after reversal)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with reversed labels
model.fit(
    reverse_labels(train_data),
    steps_per_epoch=len(train_data),
    validation_data=reverse_labels(val_data),
    validation_steps=len(val_data),
    epochs=EPOCHS
)

# Save model
os.makedirs('model', exist_ok=True)
model.save('model/cnn_model.h5')
print("✅ Model retrained with reversed labels and saved to model/cnn_model.h5")
