import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import extcolors
from colormap import rgb2hex
from scipy.spatial import KDTree
from webcolors import hex_to_name, hex_to_rgb, CSS3_HEX_TO_NAMES


def exact_color(input_image, resize, tolerance, zoom):
    # background
    bg = 'bg.png'
    fig, ax = plt.subplots(figsize=(192, 108), dpi=10)
    fig.set_facecolor('white')
    plt.savefig(bg)
    plt.close(fig)

    # resize
    output_width = resize
    img = Image.open(input_image)
    if img.size[0] >= resize:
        wpercent = (output_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((output_width, hsize), Image.LANCZOS)
        resize_name = '../images/processed/resized_color_extraction.png'
        img.save(resize_name)
    else:
        resize_name = input_image

    # Create dataframe
    img_url = input_image
    colors_x = extcolors.extract_from_path(img_url, tolerance=tolerance, limit=13)
    df_color = color_to_df(colors_x)

    # Annotate text
    list_color = list(df_color['c_code'])
    list_color_name = list(df_color['c_name'])
    list_precent = [int(i) for i in list(df_color['occurence'])]
    text_c = [c + ' ' + str(round(p * 100 / sum(list_precent), 1)) + '%' for c, p in zip(list_color, list_precent)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160, 120), dpi=10)

    # Create donut plot
    wedges, text = ax1.pie(list_precent,
                           labels=text_c,
                           labeldistance=1.05,
                           colors=list_color,
                           textprops={'fontsize': 150, 'color': 'black'})
    plt.setp(wedges, width=0.3)

    # add image in the center of donut plot
    img = mpimg.imread(resize_name)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0, 0))
    ax1.add_artist(ab)

    # color palette
    x_posi, y_posi, y_posi2 = 160, -170, -170
    for c in list_color:
        if list_color.index(c) <= 5:
            y_posi += 180
            rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor=c)
            ax2.add_patch(rect)
            ax2.text(x=x_posi + 400, y=y_posi + 100, s=list_color_name[list_color.index(c)], fontdict={'fontsize': 180})
        else:
            y_posi2 += 180
            rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor=c)
            ax2.add_artist(rect)
            ax2.text(x=x_posi + 1400, y=y_posi2 + 100, s=list_color_name[list_color.index(c)], fontdict={'fontsize': 180})

    fig.set_facecolor('white')
    ax2.axis('off')
    bg = plt.imread('bg.png')
    plt.imshow(bg)
    plt.tight_layout()

    # Save the figure as an image
    output_filename = '../images/processed/color_extraction.png'
    plt.savefig(output_filename)
    plt.close(fig)

    # Open the saved image
    img = cv2.imread(output_filename)

    # Display the image
    cv2.imshow("color_extraction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


def color_to_df(input):
    colors_pre_list = str(input).replace('([(', '').split(', (')[0:-1]
    df_rgb_tuples = [tuple(map(int, i.split('), ')[0].replace('(', '').split(', '))) for i in colors_pre_list]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')', '') for i in colors_pre_list]
    # Convert RGB to HEX code
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(", "")),
                           int(i.split(", ")[1]),
                           int(i.split(", ")[2].replace(")", ""))) for i in df_rgb]

    # Get color names
    color_names = [rgb_to_name(color) for color in df_rgb_tuples]
    print(color_names)

    df = pd.DataFrame(zip(df_color_up, color_names, df_percent), columns=['c_code', 'c_name', 'occurence'])
    return df


def rgb_to_name(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index] if index is not None else None

exact_color('../images/faces/image2.JPG', 900, 12, 0.9)
# exact_color('../images/faces/image2.JPG', 900, 24, 2.5)
# exact_color('../images/faces/image2.JPG', 900, 36, 2.5)
# exact_color('../images/faces/image2.JPG', 900, 24, 4.5)
