import cv2
import dlib
import numpy as np

# Load face detection and facial landmark detection models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load input image
img = cv2.imread('input.jpg')

# Resize and crop image to focus on face
h, w = img.shape[:2]
face_rect = face_detector(img, 1)[0]
x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
face_img = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

# Detect facial landmarks
gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
landmarks = landmark_predictor(gray, face_rect)

# Extract facial features
left_eye_pts = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                         (landmarks.part(37).x, landmarks.part(37).y),
                         (landmarks.part(38).x, landmarks.part(38).y),
                         (landmarks.part(39).x, landmarks.part(39).y),
                         (landmarks.part(40).x, landmarks.part(40).y),
                         (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

right_eye_pts = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                          (landmarks.part(43).x, landmarks.part(43).y),
                          (landmarks.part(44).x, landmarks.part(44).y),
                          (landmarks.part(45).x, landmarks.part(45).y),
                          (landmarks.part(46).x, landmarks.part(46).y),
                          (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

nose_pts = np.array([(landmarks.part(27).x, landmarks.part(27).y),
                     (landmarks.part(31).x, landmarks.part(31).y),
                     (landmarks.part(35).x, landmarks.part(35).y)], np.int32)

mouth_pts = np.array([(landmarks.part(48).x, landmarks.part(48).y),
                      (landmarks.part(54).x, landmarks.part(54).y),
                      (landmarks.part(60).x, landmarks.part(60).y),
                      (landmarks.part(64).x, landmarks.part(64).y),
                      (landmarks.part(67).x, landmarks.part(67).y),
                      (landmarks.part(66).x, landmarks.part(66).y),
                      (landmarks.part(65).x, landmarks.part(65).y),
                      (landmarks.part(62).x, landmarks.part(62).y),
                      (landmarks.part(61).x, landmarks.part(61).y),
                      (landmarks.part(56).x, landmarks.part(56).y),
                      (landmarks.part(51).x, landmarks.part(51).y)], np.int32)

# Draw facial features on image
cv2.polylines(face_img, [left_eye_pts], True, (0, 0, 255), 2)
cv2.polylines(face_img, [right_eye_pts], True, (0, 0, 255), 2)
cv2
