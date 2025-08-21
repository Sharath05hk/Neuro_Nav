import cv2
import time
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the CNN model
model = load_model("model/cnn_model.h5")

# Constants
BLINK_HOLD_DURATION = 3.0  # seconds

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Eye indices
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]

# Webcam
cap = cv2.VideoCapture(0)

# Blink timers
left_blink_start = None
right_blink_start = None

def get_eye_roi(image, landmarks, eye_points):
    h, w, _ = image.shape
    x1 = int(landmarks[eye_points[0]].x * w)
    y1 = int(landmarks[eye_points[0]].y * h)
    x2 = int(landmarks[eye_points[1]].x * w)
    y2 = int(landmarks[eye_points[1]].y * h)

    margin = 10
    x1, x2 = max(0, x1 - margin), min(w, x2 + margin)
    y1, y2 = max(0, y1 - margin), min(h, y2 + margin)

    return image[y1:y2, x1:x2]

def preprocess_eye(eye_img):
    eye_img = cv2.resize(eye_img, (24, 24))
    eye_img = eye_img / 255.0
    return np.expand_dims(eye_img, axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the webcam
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    feedback = ""

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            landmark_list = landmarks.landmark

            left_eye_img = get_eye_roi(frame, landmark_list, LEFT_EYE_IDX)
            right_eye_img = get_eye_roi(frame, landmark_list, RIGHT_EYE_IDX)

            left_closed = False
            right_closed = False

            if left_eye_img.size != 0:
                left_pred = model.predict(preprocess_eye(left_eye_img))[0][0]
                left_closed = left_pred > 0.5
            if right_eye_img.size != 0:
                right_pred = model.predict(preprocess_eye(right_eye_img))[0][0]
                right_closed = right_pred > 0.5

            # LEFT BLINK
            if left_closed:
                if left_blink_start is None:
                    left_blink_start = time.time()
                elif time.time() - left_blink_start >= BLINK_HOLD_DURATION:
                    pyautogui.click(button='left')
                    feedback = "LEFT CLICK"
                    left_blink_start = None
            else:
                left_blink_start = None

            # RIGHT BLINK
            if right_closed:
                if right_blink_start is None:
                    right_blink_start = time.time()
                elif time.time() - right_blink_start >= BLINK_HOLD_DURATION:
                    pyautogui.click(button='right')
                    feedback = "RIGHT CLICK"
                    right_blink_start = None
            else:
                right_blink_start = None

    # 🪟 Display feedback on frame
    if feedback:
        cv2.putText(frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 4)

    cv2.imshow("Neuro Nav - Blink Click", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
