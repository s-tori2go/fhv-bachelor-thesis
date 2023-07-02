import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_palette(title, palette):
    # Determine the number of rows and columns in the palette grid
    num_colors = len(palette)
    max_columns = 20  # Maximum number of columns in the grid
    num_rows = (num_colors + max_columns - 1) // max_columns

    # Calculate the size of each color cell in the grid
    cell_size = 50
    grid_width = min(num_colors, max_columns) * cell_size
    grid_height = num_rows * cell_size

    # Create a blank palette image
    palette_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Fill the palette image with colors
    for i, color in enumerate(palette):
        row = i // max_columns
        col = i % max_columns
        x = col * cell_size
        y = row * cell_size
        palette_image[y:y + cell_size, x:x + cell_size, :] = color

    # Display the palette image
    cv2.imshow(title, palette_image)


def show_color_spectrum(title, palette):
    # Calculate the size of the spectrum image
    num_colors = len(palette)
    spectrum_width = 1000
    spectrum_height = 100

    # Create a blank spectrum image
    spectrum_image = np.zeros((spectrum_height, spectrum_width, 3), dtype=np.uint8)

    # Calculate the width of each color segment
    segment_width = spectrum_width / num_colors

    # Fill the spectrum image with colors
    for i, color in enumerate(palette):
        start_x = int(i * segment_width)
        end_x = int((i + 1) * segment_width)
        spectrum_image[:, start_x:end_x, :] = color

    # Display the spectrum image
    cv2.imshow(title, spectrum_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()