import colorsys
import numpy as np
import math

from scr.convert_colors import rgb2hsv, hsv2rgb


def create_complementary_palette(rgb_palette):
    hsv_palette = rgb2hsv(rgb_palette)

    # Add 180° to the hue (h) for each color in the HSV palette
    complementary_palette = np.copy(hsv_palette)
    complementary_palette[:, 0] = (complementary_palette[:, 0] + 180) % 360
    # print(f"complementary_palette hsv: {complementary_palette}")
    complementary_palette = hsv2rgb(complementary_palette)
    print(f"complementary_palette rgb: {complementary_palette}")

    return complementary_palette


def create_palette(rgb_palette, degree):
    hsv_palette = rgb2hsv(rgb_palette)

    palette1 = np.copy(hsv_palette)
    palette1[:, 0] = (palette1[:, 0] + degree) % 360
    palette2 = np.copy(hsv_palette)
    palette2[:, 0] = (palette2[:, 0] - degree) % 360
    palette = np.concatenate((palette1, palette2), axis=0)
    palette = hsv2rgb(palette)

    return palette


def create_analogous_palette(rgb_palette):
    hsv_palette = rgb2hsv(rgb_palette)

    # Add 30° and subtract 30° from the hue (h) for each color in the HSV palette
    analogous_palette1 = np.copy(hsv_palette)
    analogous_palette1[:, 0] = (analogous_palette1[:, 0] + 30) % 360
    analogous_palette2 = np.copy(hsv_palette)
    analogous_palette2[:, 0] = (analogous_palette2[:, 0] - 30) % 360
    analogous_palette = np.concatenate((analogous_palette1, analogous_palette2), axis=0)
    # print(f"analogous_palette hsv: {analogous_palette}")
    analogous_palette = hsv2rgb(analogous_palette)
    print(f"analogous_palette rgb: {analogous_palette}")

    return analogous_palette


def create_triadic_palette(rgb_palette):
    hsv_palette = rgb2hsv(rgb_palette)

    # Add 120° and subtract 120° from the hue (h) for each color in the HSV palette
    triadic_palette1 = np.copy(hsv_palette)
    triadic_palette1[:, 0] = (triadic_palette1[:, 0] + 120) % 360
    triadic_palette2 = np.copy(hsv_palette)
    triadic_palette2[:, 0] = (triadic_palette2[:, 0] - 120) % 360
    triadic_palette = np.concatenate((triadic_palette1, triadic_palette2), axis=0)
    # print(f"triadic_palette hsv: {triadic_palette}")
    triadic_palette = hsv2rgb(triadic_palette)
    print(f"triadic_palette rgb: {triadic_palette}")

    return triadic_palette


def create_split_complementary_palette(rgb_palette):
    hsv_palette = rgb2hsv(rgb_palette)

    # Add 150° and add 210° from the hue (h) for each color in the HSV palette
    split_complementary_palette1 = np.copy(hsv_palette)
    split_complementary_palette1[:, 0] = (split_complementary_palette1[:, 0] + 150) % 360
    split_complementary_palette2 = np.copy(hsv_palette)
    split_complementary_palette2[:, 0] = (split_complementary_palette2[:, 0] + 210) % 360
    split_complementary_palette = np.concatenate((split_complementary_palette1, split_complementary_palette2), axis=0)
    # print(f"split_complementary_palette hsv: {split_complementary_palette}")
    split_complementary_palette = hsv2rgb(split_complementary_palette)
    print(f"split_complementary_palette rgb: {split_complementary_palette}")

    return split_complementary_palette


def create_square_palette(rgb_palette):
    hsv_palette = rgb2hsv(rgb_palette)

    # Add 90° and add 270° from the hue (h) for each color in the HSV palette
    square_palette1 = np.copy(hsv_palette)
    square_palette1[:, 0] = (square_palette1[:, 0] + 90) % 360
    square_palette2 = np.copy(hsv_palette)
    square_palette2[:, 0] = (square_palette2[:, 0] + 270) % 360
    square_palette = np.concatenate((square_palette1, square_palette2), axis=0)
    # print(f"square_palette hsv: {square_palette}")
    square_palette = hsv2rgb(square_palette)
    print(f"square_palette rgb: {square_palette}")

    return square_palette


def create_tetradic_palette(rgb_palette):
    hsv_palette = rgb2hsv(rgb_palette)

    # Add 60° and add 240° from the hue (h) for each color in the HSV palette
    tetradic_palette1 = np.copy(hsv_palette)
    tetradic_palette1[:, 0] = (tetradic_palette1[:, 0] + 60) % 360
    tetradic_palette2 = np.copy(hsv_palette)
    tetradic_palette2[:, 0] = (tetradic_palette2[:, 0] + 240) % 360
    tetradic_palette = np.concatenate((tetradic_palette1, tetradic_palette2), axis=0)
    # print(f"tetradic_palette hsv: {tetradic_palette}")
    tetradic_palette = hsv2rgb(tetradic_palette)
    print(f"tetradic_palette rgb: {tetradic_palette}")

    return tetradic_palette


# http://sputnik.freewisdom.org/lib/colors/#License
# https://dev.to/madsstoumann/colors-are-math-how-they-match-and-how-to-build-a-color-picker-4ei8
# http://bahamas10.github.io/ryb/
def create_rgb_palettes(rgb_palette):
    print(f"rgb_palette: {rgb_palette}")
    generated_palettes = []

    complementary_palette = create_complementary_palette(rgb_palette)
    analogous_palette = create_analogous_palette(rgb_palette)
    # triadic_palette = create_triadic_palette(rgb_palette)
    split_complementary_palette = create_split_complementary_palette(rgb_palette)
    # square_palette = create_square_palette(rgb_palette)
    # tetradic_palette = create_tetradic_palette(rgb_palette)

    generated_palettes.extend(complementary_palette)
    generated_palettes.extend(analogous_palette)
    # generated_palettes.extend(triadic_palette)
    generated_palettes.extend(split_complementary_palette)
    # generated_palettes.extend(tetradic_palette)
    # generated_palettes.extend(square_palette)
    print(f"generated palettes: {generated_palettes}")

    return generated_palettes
