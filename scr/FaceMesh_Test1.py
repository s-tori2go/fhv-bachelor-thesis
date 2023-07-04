import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import numpy as np

from scr.color_harmonization import create_rgb_palettes
from scr.Dlib_Test2.show_palette import show_palette
from scr.convert_colors import rgb2bgr, bgr2rgb


# https://github.com/serengil/tensorflow-101/blob/master/python/Mediapipe-Face-Detector.ipynb
def plot_landmark(img_base, facial_area_name, facial_area_obj):
    print(facial_area_name, ":")

    img = img_base.copy()

    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

        cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness=2)

    fig = plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(img[:, :, ::-1])
    plt.show()


def analyze_area_color(img_base, facial_area_name, facial_area_obj):
    img = img_base.copy()
    area_mask = np.zeros_like(img[:, :, 0])  # Mask to extract the area of interest

    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

        cv2.line(area_mask, relative_source, relative_target, 255, thickness=2)

    # Extract the area of interest from the image using the mask
    area = cv2.bitwise_and(img, img, mask=area_mask)

    # Calculate the dominant color in the area, ignoring black
    pixels = area.reshape(-1, 3)  # Reshape to a flat array of pixels

    # Remove black pixels from the list of pixels
    pixels_without_black = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if len(pixels_without_black) > 0:
        unique_colors, color_counts = np.unique(pixels_without_black, axis=0, return_counts=True)
        dominant_color = unique_colors[np.argmax(color_counts)]
    else:
        # If there are no non-black pixels, return black as the dominant color
        dominant_color = [0, 0, 0]

    # Convert the dominant color from numpy array to RGB values
    rgb_dominant_color = [int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0])]
    bgr_dominant_color = (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2]))

    show_palette(facial_area_name, [bgr_dominant_color])

    return rgb_dominant_color


# https://github.com/serengil/tensorflow-101/blob/master/python/Mediapipe-Face-Detector.ipynb
# https://python.plainenglish.io/face-mesh-detection-with-python-and-opencv-complete-project-359d81d6a712

mpDraw = mp.solutions.drawing_utils
faceModule = mp.solutions.face_mesh
face_mesh = faceModule.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Load and process the image
absolute_path_image = "/Users/viktoriiasimakova/Documents/GitHub/bachelor-thesis/images/faces/image2.JPG"
image_base = cv2.imread(absolute_path_image)
image = image_base.copy()
imageWithLandmarks = cv2.imread(absolute_path_image)
image_height, image_width, image_channel = image_base.shape
image_rgb = cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB)

# Perform face mesh detection using MediaPipe
results = face_mesh.process(image_rgb)

landmarks = results.multi_face_landmarks[0]

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

        facial_areas = {
            'Contours': faceModule.FACEMESH_CONTOURS
            , 'Lips': faceModule.FACEMESH_LIPS
            , 'Face_oval': faceModule.FACEMESH_FACE_OVAL
            , 'Left_eye': faceModule.FACEMESH_LEFT_EYE
            , 'Left_eye_brow': faceModule.FACEMESH_LEFT_EYEBROW
            , 'Right_eye': faceModule.FACEMESH_RIGHT_EYE
            , 'Right_eye_brow': faceModule.FACEMESH_RIGHT_EYEBROW
            # , 'Tesselation': faceModule.FACEMESH_TESSELATION # this includes the whole landmark
        }

        for facial_area in facial_areas.keys():
            facial_area_obj = facial_areas[facial_area]
            analyze_area_color(image_base, facial_area, facial_area_obj)

        all_colors = []
        generated_palettes = []
        for part_name, part_landmarks in facial_areas.items():
            part_colors = []
            for landmark_group in part_landmarks:
                group_colors = []
                for landmark_id in landmark_group:
                    landmark = facial_landmarks.landmark[landmark_id]
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                    b, g, r = image_base[y, x]
                    color = (b, g, r)
                    group_colors.append(color)
                    all_colors.append(color)
                part_colors.extend(group_colors)
            # show_palette(part_name, part_colors)

        # show_palette('all colors', all_colors)
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
