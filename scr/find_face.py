import mediapipe as mp
import cv2
import dlib

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
        rect = dlib.rectangle(
            int(bbox.xmin * image_width),
            int(bbox.ymin * image_height),
            int((bbox.xmin + bbox.width) * image_width),
            int((bbox.ymin + bbox.height) * image_height)
        )

        # Use the shape predictor to perform facial feature localization
        landmarks = shape_predictor(image, rect)

        # Access the facial landmarks and perform further analysis
        # Extract coordinates of facial features
        left_eye = landmarks.part(36).x, landmarks.part(36).y
        right_eye = landmarks.part(45).x, landmarks.part(45).y
        nose_tip = landmarks.part(30).x, landmarks.part(30).y
        mouth_left = landmarks.part(48).x, landmarks.part(48).y
        mouth_right = landmarks.part(54).x, landmarks.part(54).y

        # Draw circles around facial features
        cv2.circle(image, left_eye, 2, (0, 255, 0), -1)
        cv2.circle(image, right_eye, 2, (0, 255, 0), -1)
        cv2.circle(image, nose_tip, 2, (0, 255, 0), -1)
        cv2.circle(image, mouth_left, 2, (0, 255, 0), -1)
        cv2.circle(image, mouth_right, 2, (0, 255, 0), -1)

# Display image with facial features marked
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
face_detection.close()
shape_predictor = None  # Release the shape predictor model
