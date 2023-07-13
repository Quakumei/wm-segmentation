"""Produce dataset for training and testing"""

import os

import click

from wm_segmentation.data.produce_dataset import produce_dataset


@click.command("produce_dataset")
@click.argument("source_images_dir", type=click.Path(exists=True))
@click.argument("watermarks_dir", type=click.Path(exists=True))
@click.argument("subtitles_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def produce_dataset_handler(
    source_images_dir: str, watermarks_dir: str, subtitles_file: str, output_dir: str
):
    """Produce dataset for training and testing"""

    # 1. Log inputs
    n_images = len(os.listdir(source_images_dir))
    n_watermarks = len(os.listdir(watermarks_dir))
    n_subtitles = len(open(subtitles_file).readlines())
    print("=== Inputs ===============================")
    print(f"\tn_images:\t{n_images}")
    print(f"\tn_watermarks:\t{n_watermarks}")
    print(f"\tn_subtitles:\t{n_subtitles}")
    print(f"\tinput_dir:\t{source_images_dir}")
    print(f"\toutput_dir:\t{output_dir}")
    print("==========================================")

    # 2. Produce dataset
    produce_dataset(
        source_images_dir=source_images_dir,
        watermarks_dir=watermarks_dir,
        subtitles_file=subtitles_file,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    produce_dataset_handler()
