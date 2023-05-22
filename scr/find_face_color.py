import mediapipe as mp
import cv2
import dlib
import numpy as np

# Create a MediaPipe pipeline
face_detection = mp.solutions.face_detection.FaceDetection()
face_detection.min_detection_confidence = 0.5

# Initialize face detector and landmark detector
detector = dlib.get_frontal_face_detector()
shape_predictor_model = "/Users/viktoriiasimakova/Documents/GitHub/bachelor-thesis/models/shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(shape_predictor_model)

# Load and process the image
absolute_path_image = "/Users/viktoriiasimakova/Documents/GitHub/bachelor-thesis/images/faces/image.jpeg"
image = cv2.imread(absolute_path_image)
image_height, image_width, image_channel = image.shape

# Perform face detection using MediaPipe
results = face_detection.process(image)

# Process the detected faces
if results.detections:
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x, y, w, h = int(bbox.xmin * image_width), int(bbox.ymin * image_height), \
                     int(bbox.width * image_width), int(bbox.height * image_height)

        # Surround the facial area with a rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Use the shape predictor to perform facial feature localization
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = shape_predictor(image, rect)

        # Extract color palette from the facial area
        facial_area = image[y:y + h, x:x + w]
        colors = np.reshape(facial_area, (-1, 3))
        unique_colors, color_counts = np.unique(colors, axis=0, return_counts=True)
        dominant_colors = unique_colors[np.argsort(-color_counts)][:5]  # Extract top 5 dominant colors

        # Display the dominant colors
        for i, color in enumerate(dominant_colors):
            cv2.rectangle(image, (x + i * 50, y + h), (x + i * 50 + 50, y + h + 50), tuple(color.tolist()), -1)

# Display the image with facial areas and color palette
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
face_detection.close()
shape_predictor = None  # Release the shape predictor model
