"""produce_dataset - produce dataset for training and testing"""

import typing as tp
import glob
import os
import random
import itertools

import cv2
import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_images_from_folder(
    folder: str, extension: str = "jpg", provide_filepaths: bool = False
) -> tp.Iterable[np.ndarray]:
    """Reads images from folder in a lazy way"""
    for filepath in glob.glob(f"{folder}/*.{extension}"):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        # Convert to RGBA
        success, img = cv2.imencode(".png", img)
        if not success:
            raise RuntimeError(f"Failed to encode image {filepath} to png")
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

        if img.shape[-1] < 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        assert (
            img.shape[-1] == 4
        ), f"Expected 4 channels, got {img.shape[2]} ({filepath}))"

        if provide_filepaths:
            yield filepath, img
        else:
            yield img


def produce_watermarked_image(
    image: np.ndarray,
    watermark: np.ndarray,
    watermark_size: float,
    watermark_transparency: float,
    position: tp.Literal["random"] = "random",
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Applies watermark to an image

    Args:
        image (np.ndarray): an image on which watermark is to be applied on
        watermark (np.ndarray): watermark to be applied
        watermark_size (float): determines the size of watermark relative to image size
        watermark_transparency (float): determines the transparency of watermark
        position (tp.Literal['random'], optional): Method of positioning the watermark. Defaults to 'random'.

    Returns:
        np.ndarray: Image with applied watermark
        np.ndarray: Mask of the watermarked image
    """
    # 0. Assert that the watermark has 4 channels
    assert watermark.shape[-1] == 4, f"Expected 4 channels, got {watermark.shape[2]}"
    assert image.shape[-1] == 4, f"Expected 4 channels, got {image.shape[2]}"
    assert (
        watermark_transparency >= 0 and watermark_transparency <= 1
    ), f"Expected transparency to be in range [0, 1], got {watermark_transparency}"

    # 1. Resize watermark relative to the image size
    image_height, image_width, _ = image.shape
    watermark_height = int(image_height * watermark_size)
    watermark_width = int(image_width * watermark_size)
    watermark_patch = cv2.resize(watermark, (watermark_width, watermark_height))

    # 2. Determine the position to apply the watermark
    if position == "random":
        x_pos = random.randint(0, image_width - watermark_width)
        y_pos = random.randint(0, image_height - watermark_height)
    else:
        raise NotImplementedError(f"Position {position} is not implemented")

    # 3. Apply the watermark with transparency
    watermark_patch[:, :, 3] = np.ceil(
        watermark_patch[:, :, 3] * watermark_transparency
    )
    resized_watermark_patch = Image.fromarray(watermark_patch).convert("RGBA")

    watermarked_image = Image.fromarray(image).convert("RGBA")
    watermarked_image.paste(
        resized_watermark_patch, (x_pos, y_pos), resized_watermark_patch
    )
    watermarked_image = np.array(watermarked_image)

    # 4. Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y_pos : y_pos + watermark_height, x_pos : x_pos + watermark_width][
        watermark_patch[:, :, 3] > 0
    ] = 1

    # 5. Assert that the mask is of the same shape as the image for safety
    assert mask.shape == image.shape[:2]

    return watermarked_image, mask


def produce_subtitled_image(
    image: np.ndarray,
    subtitle_text: str,
    subtitle_size: float,  # relative to image size
    subtitle_color: tp.Tuple[int, int, int],
    subtitle_position: tp.Literal["top", "bottom"] = "bottom",
    subtitle_font: str = "data/source/fonts/Roboto-Regular.ttf",
):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    width, height = image.size
    subtitle_font_size = int(min(width, height) * subtitle_size)
    subtitle_font = ImageFont.truetype(subtitle_font, size=subtitle_font_size)
    text_width, text_height = draw.textsize(subtitle_text, font=subtitle_font)
    if subtitle_position == "bottom":
        position_x = (width - text_width) // 2
        position_y = height - text_height - 10  # Adjust the position as desired
    elif subtitle_position == "top":
        position_x = (width - text_width) // 2
        position_y = 10  # Adjust the position as desired
    else:
        raise ValueError(
            "Invalid subtitle position. Valid options are: 'top', 'bottom'."
        )

    # Draw the subtitle text on the image
    draw.text(
        (position_x, position_y), subtitle_text, font=subtitle_font, fill=subtitle_color
    )

    image = np.array(image)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.text((position_x, position_y), subtitle_text, font=subtitle_font, fill=1)
    mask = np.array(mask)

    assert mask.shape == image.shape[:2]

    return image, mask


def produce_dataset(
    source_images_dir: str,
    watermarks_dir: str,
    subtitles_file: str,
    output_dir: str,
    watermark_transparencies: tp.List[float] = [0.7, 0.9, 1.0],
    watermark_sizes: tp.List[float] = [0.3, 0.5, 0.7],  # relative to image size
    subtitles_sizes: tp.List[float] = [0.03, 0.05],  # relative to image size
    subtitles_colors: tp.List[tp.Tuple[int, int, int]] = [
        (255, 255, 255),
        (0, 255, 255),
        (255, 255, 0),
    ],
    produce_masks_jpg: bool = False,
):
    """Produce dataset for training and testing"""
    # 1. Create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Read watermarks and subtitles
    subtitles = open(subtitles_file).readlines()
    watermarks = []
    watermarks_gen = read_images_from_folder(watermarks_dir, extension="png")
    watermarks_count = len(os.listdir(watermarks_dir))
    for watermark in tqdm.tqdm(watermarks_gen, total=watermarks_count):
        watermarks.append(watermark)

    # 3. Read images from folder
    images_reader = read_images_from_folder(source_images_dir, provide_filepaths=True)
    images_count = len(os.listdir(source_images_dir))
    for image_filename, image in tqdm.tqdm(images_reader, total=images_count):
        # 4. Having read image, we can apply augmentations
        # 4.1 Apply watermarks
        for i, watermark in enumerate(watermarks):
            for transparency, size in itertools.product(
                watermark_transparencies, watermark_sizes
            ):
                watermarked_image, mask = produce_watermarked_image(
                    image=image,
                    watermark=watermark,
                    watermark_size=size,
                    watermark_transparency=transparency,
                )
                output_filename = f"w{i}_s{size}_t{transparency}_{os.path.basename(image_filename)}"  # blablabla.jpg
                output_filepath = os.path.join(output_dir, output_filename)

                cv2.imwrite(output_filepath, watermarked_image)
                np.savez_compressed(output_filepath.replace(".jpg", ".npz"), mask=mask)
                if produce_masks_jpg:
                    cv2.imwrite(
                        output_filepath.replace(".jpg", ".mask.jpg"), mask * 255
                    )

        # 4.2 Apply subtitles
        for i, subtitle in enumerate(subtitles):
            for size, color in itertools.product(subtitles_sizes, subtitles_colors):
                subtitled_image, mask = produce_subtitled_image(
                    image=image,
                    subtitle_text=subtitle,
                    subtitle_size=size,
                    subtitle_color=color,
                )

                output_filename = f"sub{i}_s{size}_c{color}_{os.path.basename(image_filename)}"  # blablabla.jpg
                output_filepath = os.path.join(output_dir, output_filename)

                cv2.imwrite(output_filepath, subtitled_image)
                np.savez_compressed(output_filepath.replace(".jpg", ".npz"), mask=mask)
                if produce_masks_jpg:
                    cv2.imwrite(
                        output_filepath.replace(".jpg", ".mask.jpg"), mask * 255
                    )
