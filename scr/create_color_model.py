import colorsys
import cv2
import numpy as np
from skimage import color


def save_colors_as_rgb(palette):
    # Convert LAB values to RGB
    lab_palette = np.array(palette)
    rgb_palette = color.lab2rgb(lab_palette)

    # Scale RGB values from [0, 1] to [0, 255]
    rgb_palette *= 255
    rgb_palette = rgb_palette.round().astype(int)

    # Return the RGB color palette
    return rgb_palette.tolist()


def save_palette_as_hsv(palette):
    hsv_palette = []

    for color in palette:
        hsv = colorsys.rgb_to_hsv(color[0] / 255, color[1] / 255, color[2] / 255)
        hsv_palette.append(hsv)

    return hsv_palette


def save_palette_as_lab(palette):
    lab_palette = []

    for color in palette:
        lab = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0]
        lab_palette.append(lab)

    return lab_palette