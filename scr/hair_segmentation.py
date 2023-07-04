import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python

# https://developers.google.com/mediapipe/solutions/vision/image_segmenter/index
# https://developers.google.com/mediapipe/solutions/vision/image_segmenter/python
# https://github.com/googlesamples/mediapipe/blob/main/examples/image_segmentation/python/image_segmentation.ipynb

path_image = '../images/faces/image3.png'

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create an image segmenter instance with the image mode
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='../models/selfie_multiclass_256x256.tflite'),
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True)
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

    print(f'Blurred background of {path_image}:')
    cv2.imwrite('../images/processed/hair_segmentation.png', output_image)

# Open image
img = cv2.imread('../images/processed/hair_segmentation.png')
cv2.imshow("segmentation", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
