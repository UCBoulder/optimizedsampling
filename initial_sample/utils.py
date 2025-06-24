from PIL import Image, ImageDraw, ImageFont
from typing import List

from PIL import Image, ImageDraw, ImageFont
from typing import List

from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import math

def stack_and_label_images(images, labels, n_cols=None, font_size=40, padding=20, label_padding=10, bg_color="white"):
    """
    Stack images in a grid (horizontal/vertical layout) with labels underneath.
    
    Parameters:
        images (list of PIL.Image): List of images to stack.
        labels (list of str): List of labels for each image.
        n_cols (int): Number of columns in the output grid. If None, defaults to 1 row (horizontal stack).
        font_size (int): Font size for the labels.
        padding (int): Padding between images.
        label_padding (int): Space between image and label.
        bg_color (str): Background color for the canvas.

    Returns:
        PIL.Image: Combined image.
    """
    assert len(images) == len(labels), "Each image must have a label."

    # Load default matplotlib font
    font_path = fm.FontProperties(family="DejaVu Sans").get_file()
    font = ImageFont.truetype(font_path, font_size)

    # Determine grid size
    n = len(images)
    if n_cols is None:
        n_cols = n
    n_rows = math.ceil(n / n_cols)

    # Assume all images are the same size
    img_width, img_height = images[0].size

    # Measure label height
    dummy_img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy_img)
    label_heights = [draw.textbbox((0, 0), lbl, font=font)[3] for lbl in labels]
    max_label_height = max(label_heights)

    # Total size
    total_width = n_cols * img_width + (n_cols - 1) * padding
    total_height = n_rows * (img_height + label_padding + max_label_height) + (n_rows - 1) * padding

    # Create canvas
    combined = Image.new("RGB", (total_width, total_height), color=bg_color)
    draw = ImageDraw.Draw(combined)

    # Paste images and draw labels
    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // n_cols
        col = idx % n_cols

        x = col * (img_width + padding)
        y = row * (img_height + label_padding + max_label_height + padding)

        # Paste image
        combined.paste(img, (x, y))

        # Center label
        text_width = draw.textbbox((0, 0), label, font=font)[2]
        text_x = x + (img_width - text_width) // 2
        text_y = y + img_height + label_padding

        draw.text((text_x, text_y), label, font=font, fill="black")

    combined.save(output_path)
    print(f"Saved combined image to {output_path}")

def combine_images_with_labels(
    png_files: List[str],
    labels: List[str],
    output_path: str = 'combined_labeled.png',
    font_size: int = 75,
    label_padding: int = 100,
):
    """
    Combine PNG images horizontally with a large label under each one.

    Args:
        png_files (List[str]): List of file paths to PNG images.
        labels (List[str]): List of labels to display under each image.
        output_path (str): File path for the output image.
        font_size (int): Font size of the labels.
        label_padding (int): Vertical padding between images and labels.
    """
    if len(png_files) != len(labels):
        raise ValueError("Number of PNG files and labels must match.")

    # Load images
    images = [Image.open(f) for f in png_files]

    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        print("Cannot make correct font size")
        font = ImageFont.load_default()

    # Dimensions
    widths, heights = zip(*(img.size for img in images))
    label_heights = [font.getbbox(label)[3] - font.getbbox(label)[1] for label in labels]
    label_height = max(label_heights) + label_padding

    total_width = sum(widths)
    total_height = max(heights) + label_height

    combined = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(combined)

    # Paste images and draw labels
    x_offset = 0
    for img, label in zip(images, labels):
        combined.paste(img, (x_offset, 0))
        text_width = font.getlength(label)
        text_x = x_offset + (img.width - text_width) // 2
        text_y = img.height + label_padding // 2
        draw.text((text_x, text_y), label, font=font, fill="black")
        x_offset += img.width

    combined.save(output_path)
    print(f"Saved combined image to {output_path}")


if __name__ == '__main__':
    for num_counties in [25, 50, 75, 100]:
        for label in ["population", "treecover"]:
            pngs = [f"{label}/clustered/plots/100_counties_radius_10.png", f"{label}/density/plots/100_counties_radius_10.png"]
            sub_labels = ["By State Proportions", "By Population Density"]

            output_path = f"{label}/combined_{num_counties}.png"

            combine_images_with_labels(pngs, sub_labels, output_path)