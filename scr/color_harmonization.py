import numpy as np

from scr.Test1.show_palette import show_palette


def create_rgb_palettes(rgb_palette):
    # Convert RGB values to NumPy array
    rgb_palette = np.array(rgb_palette)
    palettes = []

    # Create complementary palette
    # Each RGB value is subtracted from 255 to get the complementary color
    complementary_palette = [255 - rgb for rgb in rgb_palette]

    # Create analogous palette
    analogous_palette = []
    for i, rgb in enumerate(rgb_palette):
        analogous_color1 = rgb_palette[(i - 1) % len(rgb_palette)]
        analogous_color2 = rgb_palette[(i + 1) % len(rgb_palette)]
        analogous_palette.append(rgb)
        analogous_palette.append(analogous_color1)
        analogous_palette.append(analogous_color2)
        palettes.append(rgb)
        palettes.append(analogous_color1)
        palettes.append(analogous_color2)

    # Create triadic palette
    triadic_palette = []
    for i, rgb in enumerate(rgb_palette):
        triadic_color1 = rgb_palette[(i + len(rgb_palette) // 3) % len(rgb_palette)]
        triadic_color2 = rgb_palette[(i + 2 * len(rgb_palette) // 3) % len(rgb_palette)]
        triadic_palette.append(rgb)
        triadic_palette.append(triadic_color1)
        triadic_palette.append(triadic_color2)
        palettes.append(rgb)
        palettes.append(triadic_color1)
        palettes.append(triadic_color2)

    # Create split-complementary palette
    split_complementary_palette = []
    for i, rgb in enumerate(rgb_palette):
        split_complementary_color1 = rgb_palette[(i + len(rgb_palette) // 2) % len(rgb_palette)]
        split_complementary_color2 = rgb_palette[(i + len(rgb_palette) * 3 // 4) % len(rgb_palette)]
        split_complementary_palette.append(rgb)
        split_complementary_palette.append(split_complementary_color1)
        split_complementary_palette.append(split_complementary_color2)
        palettes.append(rgb)
        palettes.append(split_complementary_color1)
        palettes.append(split_complementary_color2)

    # Create tetradic palette
    tetradic_palette = []
    for i, rgb in enumerate(rgb_palette):
        tetradic_color1 = rgb_palette[(i + len(rgb_palette) // 2) % len(rgb_palette)]
        tetradic_color2 = rgb_palette[(i + len(rgb_palette) // 2 + len(rgb_palette) // 4) % len(rgb_palette)]
        tetradic_palette.append(rgb)
        tetradic_palette.append(tetradic_color1)
        tetradic_palette.append(complementary_palette[i])
        tetradic_palette.append(tetradic_color2)
        palettes.append(rgb)
        palettes.append(tetradic_color1)
        palettes.append(complementary_palette[i])
        palettes.append(tetradic_color2)

    # Create square palette
    square_palette = []
    for i, rgb in enumerate(rgb_palette):
        square_color1 = rgb_palette[(i + len(rgb_palette) // 4) % len(rgb_palette)]
        square_color2 = rgb_palette[(i + len(rgb_palette) // 2) % len(rgb_palette)]
        square_color3 = rgb_palette[(i + 3 * len(rgb_palette) // 4) % len(rgb_palette)]
        square_palette.append(rgb)
        square_palette.append(square_color1)
        square_palette.append(square_color2)
        square_palette.append(square_color3)
        palettes.append(rgb)
        palettes.append(square_color1)
        palettes.append(square_color2)
        palettes.append(square_color3)

    # Display the palettes
    show_palette('Complementary Palette', complementary_palette)
    show_palette('Analogous Palette', analogous_palette)
    show_palette('Triadic Palette', triadic_palette)
    show_palette('Split-Complementary Palette', split_complementary_palette)
    show_palette('Tetradic Palette', tetradic_palette)
    show_palette('Square Palette', square_palette)
    show_palette('All Palettes', palettes)