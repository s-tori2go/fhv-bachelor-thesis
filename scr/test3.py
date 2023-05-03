import cv2
import numpy as np

# Load the image
image = cv2.imread('face.jpg')

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the ranges of colors for skin, hair, eyes, and lips
skin_lower = np.array([0, 20, 70], dtype=np.uint8)
skin_upper = np.array([20, 255, 255], dtype=np.uint8)
hair_lower = np.array([5, 50, 50], dtype=np.uint8)
hair_upper = np.array([30, 255, 255], dtype=np.uint8)
eyes_lower = np.array([0, 0, 0], dtype=np.uint8)
eyes_upper = np.array([255, 255, 50], dtype=np.uint8)
lips_lower = np.array([150, 50, 50], dtype=np.uint8)
lips_upper = np.array([180, 255, 255], dtype=np.uint8)

# Threshold the image to get the regions of each color
skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
hair_mask = cv2.inRange(hsv, hair_lower, hair_upper)
eyes_mask = cv2.inRange(hsv, eyes_lower, eyes_upper)
lips_mask = cv2.inRange(hsv, lips_lower, lips_upper)

# Apply morphology operations to improve the masks
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
eyes_mask = cv2.morphologyEx(eyes_mask, cv2.MORPH_CLOSE, kernel)
lips_mask = cv2.morphologyEx(lips_mask, cv2.MORPH_CLOSE, kernel)

# Find the contours of each color region
skin_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hair_contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
eyes_contours, _ = cv2.findContours(eyes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
lips_contours, _ = cv2.findContours(lips_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
cv2.drawContours(image, skin_contours, -1, (0, 255, 0), 2)
cv2.drawContours(image, hair_contours, -1, (0, 0, 255), 2)
cv2.drawContours(image, eyes_contours, -1, (255, 0, 0), 2)
cv2.drawContours(image, lips_contours, -1, (0, 255, 255), 2)

# Display the image
cv2.imshow('Facial features color detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
