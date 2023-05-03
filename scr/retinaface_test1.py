# https://github.com/hphuongdhsp/retinaface#installation

from retinaface import RetinafaceDetector
import cv2 as cv


### Mobinet backbone
detector  = RetinafaceDetector(net='mnet').detect_faces
img  = cv.imread('../images/image.jpeg')
bounding_boxes, landmarks = detector(img)
print(bounding_boxes)

### Resnet backbone
detector  = RetinafaceDetector(net='rnet').detect_faces
img  = cv.imread('../images/image.jpeg')
bounding_boxes, landmarks = detector(img)
print(bounding_boxes)
