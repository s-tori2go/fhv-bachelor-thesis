import numpy as np

from scr.Dlib_Test2.show_palette import show_palette


def create_lab_palettes(lab_palette):
    # Convert LAB values to NumPy array
    lab_palette = np.array(lab_palette)

    # Create complementary palette
    # A and B values are negated while keeping the L value unchanged
    complementary_palette = [(lab[0], -lab[1], -lab[2]) for lab in lab_palette]

    # Create analogous palette
    # contains the current color and its adjacent colors on the color wheel
    analogous_palette = []
    for i, lab in enumerate(lab_palette):
        analogous_color1 = lab_palette[(i - 1) % len(lab_palette)]
        analogous_color2 = lab_palette[(i + 1) % len(lab_palette)]
        analogous_palette.append(lab)
        analogous_palette.append(analogous_color1)
        analogous_palette.append(analogous_color2)

    # Create triadic palette
    # colors that are e.g. 2 steps and 4 steps ahead in a cyclic manner
    triadic_palette = []
    for i, lab in enumerate(lab_palette):
        triadic_color1 = lab_palette[(i + len(lab_palette) // 3) % len(lab_palette)]
        triadic_color2 = lab_palette[(i + 2 * len(lab_palette) // 3) % len(lab_palette)]
        triadic_palette.append(lab)
        triadic_palette.append(triadic_color1)
        triadic_palette.append(triadic_color2)

    # Create split-complementary palette
    split_complementary_palette = []
    for i, lab in enumerate(lab_palette):
        split_complementary_color1 = lab_palette[(i + len(lab_palette) // 2) % len(lab_palette)]
        split_complementary_color2 = lab_palette[(i + len(lab_palette) * 3 // 4) % len(lab_palette)]
        split_complementary_palette.append(lab)
        split_complementary_palette.append(split_complementary_color1)
        split_complementary_palette.append(split_complementary_color2)

    # Create tetradic palette
    tetradic_palette = []
    for i, lab in enumerate(lab_palette):
        tetradic_color1 = lab_palette[(i + len(lab_palette) // 2) % len(lab_palette)]
        tetradic_color2 = lab_palette[(i + len(lab_palette) // 2 + len(lab_palette) // 4) % len(lab_palette)]
        tetradic_palette.append(lab)
        tetradic_palette.append(tetradic_color1)
        tetradic_palette.append(complementary_palette[i])
        tetradic_palette.append(tetradic_color2)

    # Create square palette
    square_palette = []
    for i, lab in enumerate(lab_palette):
        square_color1 = lab_palette[(i + len(lab_palette) // 4) % len(lab_palette)]
        square_color2 = lab_palette[(i + len(lab_palette) // 2) % len(lab_palette)]
        square_color3 = lab_palette[(i + 3 * len(lab_palette) // 4) % len(lab_palette)]
        square_palette.append(lab)
        square_palette.append(square_color1)
        square_palette.append(square_color2)
        square_palette.append(square_color3)

    # Display the palettes
    show_palette('Complementary Palette', complementary_palette)
    show_palette('Analogous Palette', analogous_palette)
    show_palette('Triadic Palette', triadic_palette)
    show_palette('Split-Complementary Palette', split_complementary_palette)
    show_palette('Tetradic Palette', tetradic_palette)
    show_palette('Square Palette', square_palette)
