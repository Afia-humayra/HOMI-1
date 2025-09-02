import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier('Nariz.xml')
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
left_ear_cascade  = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

if face_cascade.empty():
    print("ERROR: Couldn't load face cascade")
    sys.exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Face (blue)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Green

        noses = nose_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)  # Red

        lower_face_gray = roi_gray[int(h / 2):, :]
        lower_face_color = roi_color[int(h / 2):, :]
        mouths = mouth_cascade.detectMultiScale(lower_face_gray, 1.1, 20, minSize=(30, 30))
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(lower_face_color, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)  # Yellow

        ly = y + int(0.15 * h)
        ry = y + int(0.9 * h)

        # Left ear region
        lx1 = max(0, x - int(0.6 * w))
        lx2 = max(0, x + int(0.1 * w))
        if lx2 - lx1 > 10 and ry - ly > 10 and not left_ear_cascade.empty():
            roi_left_gray = gray[ly:ry, lx1:lx2]
            roi_left_color = frame[ly:ry, lx1:lx2]
            left_ears = left_ear_cascade.detectMultiScale(roi_left_gray, 1.05, 3, minSize=(15, 15))
            for (ex, ey, ew, eh) in left_ears:
                cv2.rectangle(roi_left_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Green

        rx1 = x + int(0.9 * w)
        rx2 = min(frame.shape[1], x + w + int(0.6 * w))
        if rx2 - rx1 > 10 and ry - ly > 10 and not right_ear_cascade.empty():
            roi_right_gray = gray[ly:ry, rx1:rx2]
            roi_right_color = frame[ly:ry, rx1:rx2]
            right_ears = right_ear_cascade.detectMultiScale(roi_right_gray, 1.05, 3, minSize=(15, 15))
            for (ex, ey, ew, eh) in right_ears:
                cv2.rectangle(roi_right_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Green

    cv2.imshow('Face, Eyes, Nose, Mouth, Ears Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
