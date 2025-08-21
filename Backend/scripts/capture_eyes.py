import cv2
import os
import mediapipe as mp
from datetime import datetime

# Create dataset folders if they don't exist
for label in ['open_eyes', 'closed_eyes']:
    os.makedirs(f'dataset/{label}', exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Define eye landmarks
RIGHT_EYE = [33, 133]
LEFT_EYE = [362, 263]

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'o' to save OPEN eye, 'c' to save CLOSED eye, 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture image")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get eye coordinates
            l_eye = face_landmarks.landmark[LEFT_EYE[0]]
            r_eye = face_landmarks.landmark[RIGHT_EYE[0]]
            l_x, l_y = int(l_eye.x * w), int(l_eye.y * h)
            r_x, r_y = int(r_eye.x * w), int(r_eye.y * h)

            eye_width = 50
            eye_height = 30

            # Crop left and right eye
            left_eye_img = frame[l_y-eye_height//2:l_y+eye_height//2, l_x-eye_width//2:l_x+eye_width//2]
            right_eye_img = frame[r_y-eye_height//2:r_y+eye_height//2, r_x-eye_width//2:r_x+eye_width//2]

            # Show both cropped eyes
            if left_eye_img.size != 0:
                cv2.imshow("Left Eye", left_eye_img)
            if right_eye_img.size != 0:
                cv2.imshow("Right Eye", right_eye_img)

    # Display webcam feed
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    # Press 'o' to save image to open_eyes/
    if key == ord('o'):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        if left_eye_img.size != 0:
            cv2.imwrite(f'dataset/really_open_eyes/left_{timestamp}.jpg', left_eye_img)
        if right_eye_img.size != 0:
            cv2.imwrite(f'dataset/really_open_eyes/right_{timestamp}.jpg', right_eye_img)
        print("Saved open eye image.")

    # Press 'c' to save image to closed_eyes/
    elif key == ord('c'):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        if left_eye_img.size != 0:
            cv2.imwrite(f'dataset/really_closed_eyes/left_{timestamp}.jpg', left_eye_img)
        if right_eye_img.size != 0:
            cv2.imwrite(f'dataset/really_closed_eyes/right_{timestamp}.jpg', right_eye_img)
        print("Saved closed eye image.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
