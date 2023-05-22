import mediapipe as mp
import cv2
import dlib

from scr.color_harmonization import create_palettes
from scr.create_color_model import save_palette_as_lab, save_colors_as_rgb
from scr.optimize_colors import monochrome_stretch
from scr.show_palette import show_palette

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

        # Use the shape predictor to perform facial feature localization
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = shape_predictor(image, rect)

        # Extract color palettes from different facial parts
        facial_parts = {
            'eyes': [36, 37, 38, 39, 40, 41, 42, 43, 44, 45],
            'hair': list(range(0, 17)) + list(range(26, 36)),
            'mouth': list(range(48, 68)),
            'skin': list(range(1, 18)) + list(range(27, 36))
        }

        for part_name, part_landmarks in facial_parts.items():
            part_colors = []
            for landmark_id in part_landmarks:
                x, y = landmarks.part(landmark_id).x, landmarks.part(landmark_id).y
                color = image[y, x]
                part_colors.append(color)

            # Display the color palette for the facial part
            show_palette(part_name, part_colors)

            # Save the color palette as RGB values
            rgb_palette = save_colors_as_rgb(part_colors)
            #lab_palette = save_palette_as_lab(part_colors)
            print(f"{part_name} palette (LAB): {rgb_palette}")

            create_palettes(rgb_palette)

# Display the image with facial areas
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
face_detection.close()
shape_predictor = None  # Release the shape predictor model
