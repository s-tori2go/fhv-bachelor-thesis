import cv2
import numpy as np

import numpy as np

def monochrome_stretch(palette, channel):
    # Extract the specified channel from the LAB palette
    channel_values = np.array([color[channel] for color in palette])

    # Find the minimum and maximum values in the channel
    min_value = np.min(channel_values)
    max_value = np.max(channel_values)

    # Perform the monochrome stretch
    stretched_values = (channel_values - min_value) / (max_value - min_value)

    # Create the monochrome LAB palette using the stretched values
    monochrome_palette = np.copy(palette)
    for i in range(len(monochrome_palette)):
        monochrome_palette[i][channel] = stretched_values[i]

    return monochrome_palette


def organize_and_display_palettes(palettes):
    # Combine all colors from the palettes into a single list
    all_colors = [color for palette in palettes for color in palette]

    # Remove duplicates from the list of colors
    unique_colors = list(set(tuple(color) for color in all_colors))

    # Determine the number of rows and columns in the grid
    num_colors = len(unique_colors)
    max_columns = 5  # Maximum number of columns in the grid
    num_rows = (num_colors + max_columns - 1) // max_columns

    # Calculate the size of each color cell in the grid
    cell_size = 50
    grid_width = min(num_colors, max_columns) * cell_size
    grid_height = num_rows * cell_size

    # Create a blank image for the grid
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Fill the grid image with colors
    for i, color in enumerate(unique_colors):
        row = i // max_columns
        col = i % max_columns
        x = col * cell_size
        y = row * cell_size
        grid_image[y:y + cell_size, x:x + cell_size, :] = color

    # Display the grid image
    cv2.imshow('Combined Palettes', grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
