import os

import click
import keras
import tensorflow as tf
import cv2
import numpy as np

DEFAULT_MODEL_PATH='checkpoints/watermark_subtitles_dataset_segmentation/deeplabv3+/best_model_v19.h5'

def load_model(h5_file: str):
    model = keras.models.load_model(h5_file)
    return model

@click.command("inference")
@click.argument("input_img", type=click.Path(exists=True))
@click.option("--model", "-m", help="Path to the model h5 file", type=click.Path(exists=True), default=DEFAULT_MODEL_PATH)
@click.option("--output", "-o", help="Path to the output image", type=str, default=None)
@click.option("--resize", "-r", help="Resize the image to this size", type=int, default=None)
@click.option("--from-folder", "-f", help="Flag to indicate if the input is a folder", is_flag=True, default=False)
def inference_handler(
    input_img: str,
    model: str,
    output: str,
    resize: int,
    from_folder: bool,
):
    model = load_model(model)
    if from_folder:
        for file in os.listdir(input_img):
            if file.endswith(".png") or file.endswith(".jpg") and \
                not file.endswith(".output.png") and not file.endswith(".output.npz"):
                process_image(
                    os.path.join(input_img, file),
                    model,
                    output=os.path.join(output, os.path.splitext(file)[0] + ".output"),
                    resize=resize,
                )
    else:
        process_image(input_img, model, output=output, resize=resize)

def process_image(
    input_img, model, output=None, resize=None
):
    image = cv2.imread(input_img) # (h, w, 3)
    source_shape = image.shape
    print(f"Processing image: {input_img} ({image.shape})")
    if output is None:
        output = os.path.splitext(input_img)[0] + ".output"
    if resize is not None:
        image = cv2.resize(image, (resize, resize))
    image = image / 255.0
    print(f"Image resized to: {image.shape}")
    prediction = model.predict(image[None, ...])[0]
    # prediction = tf.argmax(prediction, axis=-1)
    print(f"Prediction shape: {prediction.shape}")
    if resize is not None:
        prediction = cv2.resize(prediction, (source_shape[1], source_shape[0]))

    assert prediction.shape[0] == source_shape[0]
    assert prediction.shape[1] == source_shape[1]

    # Save
    for i in range(prediction.shape[-1]):
        cv2.imwrite(output + f"_{i}.png", prediction[..., i] * 255)
        np.savez_compressed(output + f"_{i}.npz", prediction[..., i])


if __name__ == "__main__":
    inference_handler()
