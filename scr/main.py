import mediapipe as mp
import cv2

from scr.color_harmonization import create_rgb_palettes
from scr.Dlib_Test2.show_palette import show_palette
from scr.convert_colors import rgb2bgr, bgr2rgb

# https://github.com/serengil/tensorflow-101/blob/master/python/Mediapipe-Face-Detector.ipynb
# https://python.plainenglish.io/face-mesh-detection-with-python-and-opencv-complete-project-359d81d6a712

mpDraw = mp.solutions.drawing_utils
faceModule = mp.solutions.face_mesh
face_mesh = faceModule.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Load and process the image
absolute_path_image = "/Users/viktoriiasimakova/Documents/GitHub/bachelor-thesis/images/faces/image2.JPG"
image = cv2.imread(absolute_path_image)
imageWithLandmarks = cv2.imread(absolute_path_image)
image_height, image_width, image_channel = image.shape
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face mesh detection using MediaPipe
results = face_mesh.process(image_rgb)

landmarks = results.multi_face_landmarks[0]

facial_areas = {
    'Contours': faceModule.FACEMESH_CONTOURS
    , 'Lips': faceModule.FACEMESH_LIPS
    , 'Face_oval': faceModule.FACEMESH_FACE_OVAL
    , 'Left_eye': faceModule.FACEMESH_LEFT_EYE
    , 'Left_eye_brow': faceModule.FACEMESH_LEFT_EYEBROW
    , 'Right_eye': faceModule.FACEMESH_RIGHT_EYE
    , 'Right_eye_brow': faceModule.FACEMESH_RIGHT_EYEBROW
    , 'Tesselation': faceModule.FACEMESH_TESSELATION
}

# Process the detected faces
if results.multi_face_landmarks:
    for facial_landmarks in results.multi_face_landmarks:
        # Draw landmarks
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * image_width)
            y = int(pt1.y * image_height)

            cv2.circle(imageWithLandmarks, (x, y), 2, (100, 100, 0), -1)

        # Extract color palettes from different facial parts
        facial_parts = {
            'eyes': [
                [33, 133, 155, 153, 154, 145, 144, 163, 173, 157, 158, 159, 160, 161, 246, 163, 144, 145, 153, 154]],
            'hair': [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]],
            'mouth': [[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61, 185, 40, 39, 37, 0, 267, 269]],
            'skin': [[10, 338, 297, 332, 284, 251], [389, 356, 454, 323], [361, 288, 397], [365, 379, 378, 400, 377]]
        }

        all_colors = []
        generated_palettes = []
        for part_name, part_landmarks in facial_parts.items():
            part_colors = []
            for landmark_group in part_landmarks:
                group_colors = []
                for landmark_id in landmark_group:
                    landmark = facial_landmarks.landmark[landmark_id]
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                    b, g, r = image[y, x]
                    color = (b, g, r)
                    group_colors.append(color)
                    all_colors.append(color)
                part_colors.extend(group_colors)
            show_palette(part_name, part_colors)

        show_palette('all colors', all_colors)
        all_colors = bgr2rgb(all_colors)

        generated_palettes = create_rgb_palettes(all_colors)
        generated_palettes = rgb2bgr(generated_palettes)
        show_palette('generated palettes', generated_palettes)

# Display the image with facial areas
cv2.imshow("Face Mesh detection", imageWithLandmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release resources
face_mesh.close()
