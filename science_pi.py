import cv2
import mediapipe as mp
import random
import time

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Use static_image_mode=False and lower complexity for better Pi performance
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, 
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

body_parts = {
    "left eye": 33,
    "right eye": 263,
    "nose": 1,
    "mouth": 13,
    "left ear": 234,
    "right ear": 454
}

current_question = random.choice(list(body_parts.keys()))
last_switch_time = time.time()

# If using Pi Camera v2 with OpenCV
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # smaller resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def euclidean_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Resize for faster processing
    frame_small = cv2.resize(frame, (320, 240))
    h, w, _ = frame_small.shape
    rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    cv2.putText(frame_small, f"Show me your {current_question}!", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        target_idx = body_parts[current_question]
        target_lm = face_landmarks.landmark[target_idx]
        target_xy = (int(target_lm.x*w), int(target_lm.y*h))

        cv2.circle(frame_small, target_xy, 4, (0,0,255), -1)

        finger_tip_lm = hand_results.multi_hand_landmarks[0].landmark[8]
        finger_xy = (int(finger_tip_lm.x*w), int(finger_tip_lm.y*h))
        cv2.circle(frame_small, finger_xy, 4, (0,255,0), -1)

        if euclidean_distance(finger_xy, target_xy) < 30:
            cv2.putText(frame_small, "Correct!", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if time.time() - last_switch_time > 2:
                current_question = random.choice(list(body_parts.keys()))
                last_switch_time = time.time()
        else:
            cv2.putText(frame_small, "Wrong!", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Body Parts Quiz", frame_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
