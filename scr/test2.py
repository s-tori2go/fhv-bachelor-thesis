# install libraries
# pip install opencv-python
# pip install opencv-contrib-python
# pip install dlib

# import the libraries
import cv2
import dlib
import numpy as np

# load the pre-trained models for face detection and facial landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load the input image
img = cv2.imread("input.jpg")

# preprocess the image by resizing it and converting it to grayscale
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect the face in the image using the pre-trained face detection model
faces = detector(gray)

# loop through each detected face and extract the facial landmarks
for face in faces:
    landmarks = predictor(gray, face)

# extract the coordinates of the facial features
left_eye = [(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y)]
right_eye = [(landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y)]
nose = [(landmarks.part(31).x, landmarks.part(31).y), (landmarks.part(35).x, landmarks.part(35).y)]
mouth = [(landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(54).x, landmarks.part(54).y)]

# extract the color of each feature using color segmentation and color space conversions:
def get_color(img, feature):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, feature[0], feature[1], 255, -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_hue = cv2.mean(h, mask=mask)[0]
    mean_saturation = cv2.mean(s, mask=mask)[0]
    mean_value = cv2.mean(v, mask=mask)[0]
    return (mean_hue, mean_saturation, mean_value)

left_eye_color = get_color(img, left_eye)
right_eye_color = get_color(img, right_eye)
nose_color = get_color(img, nose)
mouth_color = get_color(img, mouth)

# display the image with the detected features and their colors
cv2.rectangle(img, left_eye[0], left_eye[1], left_eye_color, 2)
cv2.rectangle(img, right_eye[0], right_eye[1], right_eye_color, 2)
cv2.rectangle(img, nose[0], nose[1], nose_color, 2)
cv2.rectangle(img, mouth[0], mouth[1], mouth_color, 2)
cv2.imshow("output", img)
cv2.waitKey(0)
