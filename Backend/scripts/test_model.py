import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model/cnn_model.h5')

# Test image
img_path = 'dataset/really_open_eyes/left_20250804104255909611.jpg'  # 👈 replace with real file

# Load and preprocess
img = cv2.imread(img_path)
if img is None:
    print(f"❌ Image not found at: {img_path}")
    exit()

img = cv2.resize(img, (24, 24))
img = np.expand_dims(img, axis=0) / 255.0

# Predict
prediction = model.predict(img)[0][0]
print(f"🔍 Raw prediction score: {prediction:.4f}")

if prediction > 0.5:
    print("🧠 Model thinks: CLOSED EYE")
else:
    print("🧠 Model thinks: OPEN EYE")
