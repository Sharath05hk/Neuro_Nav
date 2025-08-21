import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False  # Disable fail-safe

# Setup
screen_w, screen_h = pyautogui.size()
frame_center_x, frame_center_y = None, None
ref_x, ref_y = None, None

# Smoothing
prev_mouse_x, prev_mouse_y = pyautogui.position()
SMOOTHING = 0.2

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Webcam
cap = cv2.VideoCapture(0)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("🧠 Head tracking started — Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            x = int(nose.x * frame_w)
            y = int(nose.y * frame_h)

            if ref_x is None:
                ref_x, ref_y = x, y  # Initial head center

            dx = x - ref_x
            dy = y - ref_y

            # Interpolate nose movement to screen size
            move_x = np.interp(dx, [-60, 60], [-screen_w // 2, screen_w // 2])
            move_y = np.interp(dy, [-60, 60], [-screen_h // 2, screen_h // 2])

            target_x = screen_w // 2 + move_x
            target_y = screen_h // 2 + move_y

            # Smooth movement
            smooth_x = prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING
            smooth_y = prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING

            pyautogui.moveTo(smooth_x, smooth_y)
            prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

            # Draw on webcam for feedback
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"dx:{int(dx)} dy:{int(dy)}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("🧠 Neuro Nav – Final Cursor Tracker", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
