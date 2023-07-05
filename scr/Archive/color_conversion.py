import cv2
import numpy as np
import colorsys


def rgb2bgr(rgb_palette):
    bgr_palette = np.array(rgb_palette)
    bgr_palette[:, [0, 2]] = bgr_palette[:, [2, 0]]
    return bgr_palette


def bgr2rgb(bgr_palette):
    rgb_palette = np.array(bgr_palette)
    rgb_palette[:, [0, 2]] = rgb_palette[:, [2, 0]]
    return rgb_palette.tolist()


def rgb2hsv(rgb_palette):
    hsv_palette = []
    for rgb_color in rgb_palette:
        hsv_color = colorsys.rgb_to_hsv(rgb_color[0]/255.0, rgb_color[1]/255.0, rgb_color[2]/255.0)
        hsv_color = [hsv_color[0]*360.0, hsv_color[1]*100.0, hsv_color[2]*100.0]
        hsv_palette.append(hsv_color)
    return hsv_palette


def hsv2rgb(hsv_palette):
    rgb_palette = []
    for hsv_color in hsv_palette:
        hsv_color = [hsv_color[0]/360.0, hsv_color[1]/100.0, hsv_color[2]/100.0]
        rgb_color = colorsys.hsv_to_rgb(hsv_color[0], hsv_color[1], hsv_color[2])
        rgb_color = [int(rgb_color[0]*255.0), int(rgb_color[1]*255.0), int(rgb_color[2]*255.0)]
        rgb_palette.append(rgb_color)
    return rgb_palette


def bgr2hsl(bgr_color):
    r, g, b = bgr_color
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    h *= 360
    return h, s, l


# https://mexicoelectoral.wordpress.com/software/ryb2rgb-r/
# TEST: http://www.deathbysoftware.com/colors/index.html
def rgb2ryb(rgb_palette):
    ryb_palette = []
    for rgb_color in rgb_palette:
        r, g, b = rgb_color

        # Remove the whiteness from the color.
        w = min(r, g, b)
        r = r - w
        g = g - w
        b = b - w

        mg = max(r, g, b)

        # Get the yellow out of the red+green.
        y = min(r, g)
        r = r - y
        g = g - y

        # If this unfortunate conversion combines blue and green, then cut each in half to preserve the value's maximum range.
        if b + g != 0:
            b = b / 2
            g = g / 2

        # Redistribute the remaining green.
        y = y + g
        b = b + g

        # Normalize to values.
        my = max(r, y, b)
        if my != 0:
            n = mg / my
            r = r * n
            y = y * n
            b = b * n

        # Add the white back in.
        r = r + w
        y = y + w
        b = b + w

        # Return the color in RYB format.
        ryb_color = (r, y, b)
        ryb_palette.append(ryb_color)

    return ryb_palette


def ryb2rgb(ryb_palette):
    rgb_palette = []
    for ryb_color in ryb_palette:
        r, y, b = ryb_color

        # Remove the whiteness from the color.
        w = min(r, y, b)
        r = r - w
        y = y - w
        b = b - w

        my = max(r, y, b)

        # Get the green out of the yellow and blue.
        g = min(y, b)
        y = y - g
        b = b - g

        if b + g != 0:
            b = b * 2.0
            g = g * 2.0

        # Redistribute the remaining yellow.
        r = r + y
        g = g + y

        # Normalize to values.
        mg = max(r, g, b)
        if mg != 0:
            n = my / mg
            r = r * n
            g = g * n
            b = b * n

        # Add the white back in.
        r = r + w
        g = g + w
        b = b + w

        rgb_color = (r, g, b)
        rgb_palette.append(rgb_color)

    return rgb_palette
