# https://www.assemblyai.com/blog/mediapipe-for-dummies/

import cv2
import mediapipe as mp
import urllib.request
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
# import PyQt5
from PIL import Image

# from IPython.display import Video
# import nb_helpers

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Create a face mesh object
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    # Read image file with cv2 and process with face_mesh
    path_image = '../images/faces/image3.png'
    image = cv2.imread(path_image)
    cv2.imshow("Image", image)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Define boolean corresponding to whether or not a face was detected in the image
face_found = bool(results.multi_face_landmarks)

if face_found:
    # Create a copy of the image
    annotated_image = image.copy()

    # Draw landmarks on face
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.multi_face_landmarks[0],
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())

    # Save image
    cv2.imwrite('../images/processed/face_tesselation_only.png', annotated_image)

if face_found:
    # Create a copy of the image
    annotated_image = image.copy()

    # For each face in the image
    for face_landmarks in results.multi_face_landmarks:
        # Draw the facial contours of the face onto the image
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

        # Draw the iris location boxes of the face onto the image
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

    # Save the image
    cv2.imwrite('../images/processed/face_contours_and_irises.png', annotated_image)

if face_found:
    facial_areas = {
        'Contours': mp_face_mesh.FACEMESH_CONTOURS
        , 'Tesselation': mp_face_mesh.FACEMESH_TESSELATION
        , 'Lips': mp_face_mesh.FACEMESH_LIPS
        , 'Face_oval': mp_face_mesh.FACEMESH_FACE_OVAL
        , 'Left_eye': mp_face_mesh.FACEMESH_LEFT_EYE
        , 'Left_eye_brow': mp_face_mesh.FACEMESH_LEFT_EYEBROW
        , 'Right_eye': mp_face_mesh.FACEMESH_RIGHT_EYE
        , 'Right_eye_brow': mp_face_mesh.FACEMESH_RIGHT_EYEBROW
        , 'Irises': mp_face_mesh.FACEMESH_IRISES
        , 'Left_iris': mp_face_mesh.FACEMESH_LEFT_IRIS
        , 'Right_iris': mp_face_mesh.FACEMESH_RIGHT_IRIS
        , 'Nose': mp_face_mesh.FACEMESH_NOSE
    }

    # https://www.youtube.com/watch?v=vE3IKPnztek
    landmarks = results.multi_face_landmarks[0]

for area_name, area_indices in facial_areas.items():
    # Create a DataFrame for the current facial area
    df = pd.DataFrame(list(area_indices), columns=["p1", "p2"])

    # Initialize variables
    routes_idx = []
    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(df.shape[0]):
        obj = df[df["p1"] == p2]
        if not obj.empty:
            p1 = obj["p1"].values[0]
            p2 = obj["p2"].values[0]
        else:
            break

        current_route = [p1, p2]
        routes_idx.append(current_route)

    routes = []
    for source_idx, target_idx in routes_idx:
        source = landmarks.landmark[int(source_idx)]
        target = landmarks.landmark[int(target_idx)]

        relative_source = (int(source.x * image.shape[1]), int(source.y * image.shape[0]))
        relative_target = (int(target.x * image.shape[1]), int(target.y * image.shape[0]))

        routes.append(relative_source)
        routes.append(relative_target)

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)

    segmented_image = np.zeros_like(image)
    segmented_image[mask] = image[mask]

    cv2.imwrite(f'../images/processed/{area_name}_segmented.jpg', segmented_image)

# Open image
img = cv2.imread('../images/processed/face_oval_segmented.jpg')
cv2.imshow("face_oval_segmented", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
