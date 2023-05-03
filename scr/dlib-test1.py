# pip install dlib

import dlib
import cv2

# Load image
img = cv2.imread('../images/image.jpeg')

# Initialize face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Detect faces in the image
faces = detector(img)

# Loop over each detected face
for face in faces:
    # Detect facial landmarks
    landmarks = predictor(img, face)

    # Extract coordinates of facial features
    left_eye = landmarks.part(36).x, landmarks.part(36).y
    right_eye = landmarks.part(45).x, landmarks.part(45).y
    nose_tip = landmarks.part(30).x, landmarks.part(30).y
    mouth_left = landmarks.part(48).x, landmarks.part(48).y
    mouth_right = landmarks.part(54).x, landmarks.part(54).y

    # Draw circles around facial features
    cv2.circle(img, left_eye, 2, (0, 255, 0), -1)
    cv2.circle(img, right_eye, 2, (0, 255, 0), -1)
    cv2.circle(img, nose_tip, 2, (0, 255, 0), -1)
    cv2.circle(img, mouth_left, 2, (0, 255, 0), -1)
    cv2.circle(img, mouth_right, 2, (0, 255, 0), -1)

# Display image with facial features marked
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
