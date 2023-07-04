from scr.color_extraction import exact_color
from scr.color_harmonization import create_rgb_palettes

rgb_values = exact_color()
print(rgb_values)
generated_palettes = create_rgb_palettes(rgb_values)
print(rgb_values)