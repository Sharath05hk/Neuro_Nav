from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import time
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
import mediapipe as mp
import threading

# ---------- Flask app ----------
app = Flask(__name__)
# Allow React (localhost:3000) to call Flask (127.0.0.1:5000)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# ---------- Your original constants & setup ----------
pyautogui.FAILSAFE = False

# Load CNN model ONCE
model = load_model("model/cnn_model.h5")

# Constants
BLINK_HOLD_DURATION = 2.0
SCROLL_HOLD_THRESHOLD = 1.0
SMOOTHING = 0.4

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]
UPPER_LIPS = 13
LOWER_LIPS = 14

# Screen
screen_w, screen_h = pyautogui.size()
prev_mouse_x, prev_mouse_y = pyautogui.position()
ref_x, ref_y = None, None

# Blink/scroll timers
left_blink_start = None
right_blink_start = None
scroll_mode = False
scroll_toggle_start_time = None

# Shared frame for streaming
output_frame = None
frame_lock = threading.Lock()

# Camera thread control
camera_running = False
camera_thread = None


# ---------- Helper functions (from your main.py) ----------
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


def get_mouth_ratio(landmarks):
    top = np.array([landmarks[UPPER_LIPS].x, landmarks[UPPER_LIPS].y])
    bottom = np.array([landmarks[LOWER_LIPS].x, landmarks[LOWER_LIPS].y])
    return np.linalg.norm(top - bottom)


# ---------- Core loop moved from main.py ----------
def detection_loop():
    global output_frame, camera_running
    global prev_mouse_x, prev_mouse_y, ref_x, ref_y
    global left_blink_start, right_blink_start, scroll_mode, scroll_toggle_start_time

    print("🧠 Neuro Nav – detection loop started (Flask thread).")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows; safe elsewhere too

    try:
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            # Keep logic same as your main.py
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            feedback = ""

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Eyes
                left_eye_img = get_eye_roi(frame, landmarks, LEFT_EYE_IDX)
                right_eye_img = get_eye_roi(frame, landmarks, RIGHT_EYE_IDX)

                left_closed = False
                right_closed = False
                if left_eye_img.size != 0:
                    left_pred = model.predict(preprocess_eye(left_eye_img), verbose=0)[0][0]
                    left_closed = left_pred > 0.5
                if right_eye_img.size != 0:
                    right_pred = model.predict(preprocess_eye(right_eye_img), verbose=0)[0][0]
                    right_closed = right_pred > 0.5

                current_time = time.time()

                # Blink detection
                if left_closed:
                    if left_blink_start is None:
                        left_blink_start = current_time
                    elif current_time - left_blink_start >= BLINK_HOLD_DURATION:
                        pyautogui.click(button='left')
                        feedback = "LEFT CLICK"
                        left_blink_start = None
                else:
                    left_blink_start = None

                if right_closed:
                    if right_blink_start is None:
                        right_blink_start = current_time
                    elif current_time - right_blink_start >= BLINK_HOLD_DURATION:
                        pyautogui.click(button='right')
                        feedback = "RIGHT CLICK"
                        right_blink_start = None
                else:
                    right_blink_start = None

                # Nose cursor (use current frame size; no pre-read frame_w/h)
                h, w = frame.shape[:2]
                nose = landmarks[1]
                x = int(nose.x * w)
                y = int(nose.y * h)

                if ref_x is None:
                    ref_x, ref_y = x, y

                dx = x - ref_x
                dy = y - ref_y

                move_x = np.interp(dx, [-80, 80], [-screen_w // 2, screen_w // 2])
                move_y = np.interp(dy, [-80, 80], [-screen_h // 2, screen_h // 2])

                target_x = screen_w // 2 + move_x
                target_y = screen_h // 2 + move_y

                smooth_x = prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING
                smooth_y = prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING
                pyautogui.moveTo(smooth_x, smooth_y)
                prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

                # Mouth scroll toggle
                mouth_ratio = get_mouth_ratio(landmarks) * h
                if mouth_ratio > 20:
                    if scroll_toggle_start_time is None:
                        scroll_toggle_start_time = current_time
                    elif current_time - scroll_toggle_start_time > SCROLL_HOLD_THRESHOLD:
                        scroll_mode = not scroll_mode
                        feedback = "SCROLL ON" if scroll_mode else "SCROLL OFF"
                        scroll_toggle_start_time = None
                        time.sleep(0.5)
                else:
                    scroll_toggle_start_time = None

                # Scroll
                if scroll_mode:
                    if dy < -15:
                        pyautogui.scroll(20)
                    elif dy > 15:
                        pyautogui.scroll(-20)

                # Visual feedback
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                if scroll_mode:
                    cv2.putText(frame, "SCROLL MODE ON", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if feedback:
                    cv2.putText(frame, feedback, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

            # Update shared frame for Flask stream
            with frame_lock:
                output_frame = frame.copy()

            # small sleep to ease CPU
            time.sleep(0.001)
    finally:
        cap.release()
        print("🛑 Detection loop stopped, camera released.")


# ---------- Flask routes ----------
@app.route("/start_camera", methods=["GET"])
def start_camera():
    global camera_running, camera_thread
    if camera_running:
        return jsonify({"status": "already running"}), 200

    # reset reference & timers when starting fresh
    _reset_state()

    camera_running = True
    camera_thread = threading.Thread(target=detection_loop, daemon=True)
    camera_thread.start()
    return jsonify({"status": "camera started"}), 200


@app.route("/stop_camera", methods=["GET"])
def stop_camera():
    global camera_running
    if not camera_running:
        return jsonify({"status": "not running"}), 200

    camera_running = False
    return jsonify({"status": "camera stopped"}), 200


@app.route("/video_feed")
def video_feed():
    def generate():
        # Stream MJPEG
        while True:
            with frame_lock:
                frame = None if output_frame is None else output_frame.copy()
            if frame is None:
                time.sleep(0.02)
                continue
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            chunk = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return jsonify({"running": camera_running}), 200


def _reset_state():
    """Reset positions and timers when starting again."""
    global prev_mouse_x, prev_mouse_y, ref_x, ref_y
    global left_blink_start, right_blink_start, scroll_mode, scroll_toggle_start_time

    prev_mouse_x, prev_mouse_y = pyautogui.position()
    ref_x, ref_y = None, None
    left_blink_start = None
    right_blink_start = None
    scroll_mode = False
    scroll_toggle_start_time = None


if __name__ == "__main__":
    # run on 127.0.0.1:5000
    app.run(debug=True)
