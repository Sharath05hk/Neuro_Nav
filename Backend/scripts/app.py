from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import time
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
import mediapipe as mp
import threading
import os
import platform

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# ---------- Constants & setup ----------
pyautogui.FAILSAFE = False
model = load_model("model/cnn_model.h5")  # Load CNN once

# Blink & scroll config
BLINK_HOLD_DURATION = 0.35
SCROLL_HOLD_THRESHOLD = 1.0

# Cursor movement config (improved accuracy)
ALPHA = 0.35          # Higher smoothing for less jitter
DEADZONE = 8          # Ignore very tiny tremors
FRAME_DELAY = 1.0 / 30

# Stream config (preview)
STREAM_DELAY = 1.0 / 15
PREVIEW_W, PREVIEW_H = 640, 360
JPEG_QUALITY = 65     # slightly better quality

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]
UPPER_LIPS = 13
LOWER_LIPS = 14

# Screen & refs
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


# ---------- Helper functions ----------
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


# ---------- Core loop ----------
def detection_loop():
    global output_frame, camera_running
    global prev_mouse_x, prev_mouse_y, ref_x, ref_y
    global left_blink_start, right_blink_start, scroll_mode, scroll_toggle_start_time

    print("🧠 Neuro Nav – detection loop started (Flask thread).")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    last_time = time.time()

    try:
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            now = time.time()
            if now - last_time < FRAME_DELAY:
                continue
            last_time = now

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            feedback = ""

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # ---- Eye blink detection ----
                left_eye_img = get_eye_roi(frame, landmarks, LEFT_EYE_IDX)
                right_eye_img = get_eye_roi(frame, landmarks, RIGHT_EYE_IDX)

                left_closed = right_closed = False
                if left_eye_img.size != 0:
                    left_pred = model.predict(preprocess_eye(left_eye_img), verbose=0)[0][0]
                    left_closed = left_pred > 0.5
                if right_eye_img.size != 0:
                    right_pred = model.predict(preprocess_eye(right_eye_img), verbose=0)[0][0]
                    right_closed = right_pred > 0.5

                current_time = time.time()

                # Left click
                if left_closed:
                    if left_blink_start is None:
                        left_blink_start = current_time
                    elif current_time - left_blink_start >= BLINK_HOLD_DURATION:
                        pyautogui.click(button="left")
                        feedback = "LEFT CLICK"
                        left_blink_start = None
                else:
                    left_blink_start = None

                # Right click
                if right_closed:
                    if right_blink_start is None:
                        right_blink_start = current_time
                    elif current_time - right_blink_start >= BLINK_HOLD_DURATION:
                        pyautogui.click(button="right")
                        feedback = "RIGHT CLICK"
                        right_blink_start = None
                else:
                    right_blink_start = None

                # ---- Cursor movement ----
                h, w = frame.shape[:2]
                nose = landmarks[1]
                x = int(nose.x * w)
                y = int(nose.y * h)

                if ref_x is None:
                    ref_x, ref_y = x, y

                dx = (x - ref_x) / w
                dy = (y - ref_y) / h

                move_x = np.interp(dx, [-0.3, 0.3], [-screen_w // 2, screen_w // 2])
                move_y = np.interp(dy, [-0.3, 0.3], [-screen_h // 2, screen_h // 2])

                target_x = screen_w // 2 + move_x
                target_y = screen_h // 2 + move_y

                if abs(target_x - prev_mouse_x) < DEADZONE:
                    target_x = prev_mouse_x
                if abs(target_y - prev_mouse_y) < DEADZONE:
                    target_y = prev_mouse_y

                # Apply exponential smoothing
                smooth_x = (1 - ALPHA) * prev_mouse_x + ALPHA * target_x
                smooth_y = (1 - ALPHA) * prev_mouse_y + ALPHA * target_y
                pyautogui.moveTo(smooth_x, smooth_y)
                prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

                # ---- Mouth scroll toggle ----
                mouth_ratio = get_mouth_ratio(landmarks) * h
                if mouth_ratio > 20:
                    if scroll_toggle_start_time is None:
                        scroll_toggle_start_time = current_time
                    elif current_time - scroll_toggle_start_time > SCROLL_HOLD_THRESHOLD:
                        scroll_mode = not scroll_mode
                        feedback = "SCROLL ON" if scroll_mode else "SCROLL OFF"
                        scroll_toggle_start_time = None
                        time.sleep(0.3)
                else:
                    scroll_toggle_start_time = None

                if scroll_mode:
                    if dy < -0.05:
                        pyautogui.scroll(20)
                    elif dy > 0.05:
                        pyautogui.scroll(-20)

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                if scroll_mode:
                    cv2.putText(frame, "SCROLL MODE ON", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if feedback:
                    cv2.putText(frame, feedback, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

            preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
            with frame_lock:
                output_frame = preview

    finally:
        cap.release()
        print("🛑 Detection loop stopped, camera released.")


# ---------- Flask routes ----------
@app.route("/start_camera", methods=["GET"])
def start_camera():
    global camera_running, camera_thread
    if camera_running:
        return jsonify({"status": "already running"}), 200

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
        last_stream_time = 0.0
        while True:
            now = time.time()
            dt = now - last_stream_time
            if dt < STREAM_DELAY:
                time.sleep(STREAM_DELAY - dt)
                continue
            last_stream_time = time.time()

            with frame_lock:
                frame = None if output_frame is None else output_frame
            if frame is None:
                time.sleep(0.005)
                continue

            ret, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ret:
                continue

            chunk = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"
            )

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Accel-Buffering": "no",
    }
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
        direct_passthrough=True,
    )


@app.route("/health")
def health():
    return jsonify({"running": camera_running}), 200


@app.route("/open_pictures", methods=["GET"])
def open_pictures():
    path = os.path.expanduser("~/Pictures")
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            os.system(f"open {path}")
        else:
            os.system(f"xdg-open {path}")
        return jsonify({"status": "opened"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def _reset_state():
    global prev_mouse_x, prev_mouse_y, ref_x, ref_y
    global left_blink_start, right_blink_start, scroll_mode, scroll_toggle_start_time
    prev_mouse_x, prev_mouse_y = pyautogui.position()
    ref_x, ref_y = None, None
    left_blink_start = None
    right_blink_start = None
    scroll_mode = False
    scroll_toggle_start_time = None


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)
