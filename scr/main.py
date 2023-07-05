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
from mediapipe.tasks import python
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import extcolors
from colormap import rgb2hex
from scipy.spatial import KDTree
from webcolors import hex_to_name, hex_to_rgb, CSS3_HEX_TO_NAMES

from scr.Archive.color_extraction import color_to_df, palette_to_df
from scr.Archive.color_harmonization import create_rgb_palettes

# from IPython.display import Video
# import nb_helpers

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
path_image = '../images/faces/image3.png'

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face mesh object
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    # Read image file with cv2 and process with face_mesh
    image = cv2.imread(path_image)
    cv2.imshow("Image", image)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Define boolean corresponding to whether or not a face was detected in the image
face_found = bool(results.multi_face_landmarks)

# Create an image segmenter instance with the image mode
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='../models/selfie_multiclass_256x256.tflite'),
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True)

if face_found:
    facial_areas = {
        'Face_oval': mp_face_mesh.FACEMESH_FACE_OVAL
        # , 'Contours': mp_face_mesh.FACEMESH_CONTOURS
        # , 'Tesselation': mp_face_mesh.FACEMESH_TESSELATION
        # , 'Lips': mp_face_mesh.FACEMESH_LIPS
        # , 'Left_eye': mp_face_mesh.FACEMESH_LEFT_EYE
        # , 'Left_eye_brow': mp_face_mesh.FACEMESH_LEFT_EYEBROW
        # , 'Right_eye': mp_face_mesh.FACEMESH_RIGHT_EYE
        # , 'Right_eye_brow': mp_face_mesh.FACEMESH_RIGHT_EYEBROW
        # , 'Irises': mp_face_mesh.FACEMESH_IRISES
        # , 'Left_iris': mp_face_mesh.FACEMESH_LEFT_IRIS
        # , 'Right_iris': mp_face_mesh.FACEMESH_RIGHT_IRIS
        # , 'Nose': mp_face_mesh.FACEMESH_NOSE
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

    cv2.imwrite(f'../images/processed/{area_name}_segmented.png', segmented_image)

# Open image
img = cv2.imread('../images/processed/face_oval_segmented.png')
cv2.imshow("face_oval_segmented", img)

with ImageSegmenter.create_from_options(options) as segmenter:
    image = mp.Image.create_from_file(path_image)

    # Retrieve the category masks for the image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask
    hair_mask = (category_mask.numpy_view() == 1)  # 1 - hair
    body_skin_mask = (category_mask.numpy_view() == 2)  # 2 - body-skin
    face_skin_mask = (category_mask.numpy_view() == 3)  # 3 - face-skin
    # 0 - background, 4 - clothes, 5 - others (accessories)

    image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

    # blurred_image = cv2.GaussianBlur(image_data, (55,55), 0) # Apply effects
    condition = np.stack((hair_mask,) * 3, axis=-1)
    # output_image = np.where(condition, image_data, blurred_image)
    output_image = np.where(condition, image_data, [0, 0, 0])

    cv2.imwrite('../images/processed/Hair_segmented.png', output_image)

# Open image
img = cv2.imread('../images/processed/Hair_segmented.png')
cv2.imshow("segmentation", img)

# Read the images
img_url_face = cv2.imread('../images/processed/Face_oval_segmented.png')
img_url_hair = cv2.imread('../images/processed/hair_segmented.png')

# Combine the images horizontally
combined_image = cv2.hconcat([img_url_face, img_url_hair])

# Save the combined image to a file
combined_image_path = '../images/processed/Face_and_hair_segmented.png'
cv2.imwrite(combined_image_path, combined_image)

# Extract colors from the combined image
colors = extcolors.extract_from_path(combined_image_path, tolerance=12, limit=13)
df_color = color_to_df(colors)
rgb_array = np.array(df_color['c_rgb'])
rgb_list = [[int(val) for val in rgb.strip('()').split(',')] for rgb in rgb_array]

# Annotate text
list_color = list(df_color['c_code'])
list_color_name = list(df_color['c_name'])
list_precent = [int(i) for i in list(df_color['occurrence'])]
text_c = [c + ' ' + str(round(p * 100 / sum(list_precent), 1)) + '%' for c, p in zip(list_color, list_precent)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160, 120), dpi=10)

# Create donut plot
wedges, text = ax1.pie(list_precent,
                       labels=text_c,
                       labeldistance=1.05,
                       colors=list_color,
                       textprops={'fontsize': 150, 'color': 'black'})
plt.setp(wedges, width=0.3)

# color palette
x_posi, y_posi, y_posi2 = 160, -170, -170
for c in list_color:
    if list_color.index(c) <= 5:
        y_posi += 180
        rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
        ax2.add_patch(rect)
        ax2.text(x=x_posi + 400, y=y_posi + 100, s=list_color_name[list_color.index(c)], fontdict={'fontsize': 180})
    else:
        y_posi2 += 180
        rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
        ax2.add_artist(rect)
        ax2.text(x=x_posi + 1400, y=y_posi2 + 100, s=list_color_name[list_color.index(c)], fontdict={'fontsize': 180})

fig.set_facecolor('white')
ax2.axis('off')
bg = plt.imread('bg.png')
plt.imshow(bg)
plt.tight_layout()

# Save the figure as an image
output_filename = '../images/processed/color_extracted.png'
plt.savefig(output_filename)
plt.close(fig)
img = cv2.imread(output_filename)
cv2.imshow(output_filename, img)


generated_palettes = create_rgb_palettes(rgb_list)
df_generated_palettes = palette_to_df(generated_palettes)
print(df_generated_palettes)

# Annotate text
list_color = list(df_generated_palettes['c_code'])
list_color_name = list(df_generated_palettes['c_name'])

# Color palette
fig, ax = plt.subplots(figsize=(30, 20))
x_posi, y_posi = 160, 70
rect_width = 60
rect_height = 60
x_spacing = 200
y_spacing = 20
text_spacing = 10

column_count = 0
for c in list_color:
    rect = patches.Rectangle((x_posi, y_posi), rect_width, rect_height, facecolor=c)
    ax.add_patch(rect)
    ax.text(x=x_posi + rect_width + text_spacing, y=y_posi + rect_height / 2, s=list_color_name[list_color.index(c)],
            fontdict={'fontsize': 18})

    y_posi += rect_height + y_spacing
    column_count += 1

    if column_count >= 11:
        column_count = 0
        x_posi += rect_width + x_spacing
        y_posi = 70

# Customize the appearance
fig.set_facecolor('white')
ax.axis('off')
bg = plt.imread('bg.png')
plt.imshow(bg)
plt.tight_layout()


# Save the figure as an image
output_filename = '../images/processed/palettes_generated.png'
plt.savefig(output_filename)
plt.close(fig)
img = cv2.imread(output_filename)
cv2.imshow(output_filename, img)

cv2.waitKey(0)
cv2.destroyAllWindows()
