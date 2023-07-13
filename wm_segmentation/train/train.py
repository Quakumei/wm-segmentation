"""Model training script."""

import tensorflow as tf
import keras

import wm_segmentation.train.train_core as train_core
import wm_segmentation.tfds.WatermarkSubtitlesDatasetSegmentation as WatermarkSubtitlesDatasetSegmentation
from wm_segmentation.models.DeeplabV3Plus import DeeplabV3Plus

IMAGE_SIZE = 256

MODELS = {"deeplabv3+": DeeplabV3Plus(IMAGE_SIZE, 2)}


def main():
    model_name = "deeplabv3+"
    model = MODELS[model_name]
    dataset_name = "watermark_subtitles_dataset_segmentation"
    logs_dir = "logs"
    checkpoints_dir = "checkpoints"
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    data_dir = "~/tensorflow_dataset"
    learning_rate = 0.001
    prefetch = 2
    batch_size = 2
    epochs = 100
    metrics = ["accuracy"]

    # Get callbacks
    callbacks = train_core.get_default_callbacks(
        model_name, dataset_name, logs_dir, checkpoints_dir, loss
    )

    # Get data
    dataset_train, dataset_val = train_core.get_train_val_from_tfds(
        dataset_name=dataset_name,
        data_dir=data_dir,
        is_supervised=True,
    )

    def preprocess_dataset(image, mask):
        mask.set_shape([None, None, 1])
        mask = tf.image.resize(images=mask, size=[IMAGE_SIZE, IMAGE_SIZE])
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = tf.keras.applications.resnet50.preprocess_input(image)

        return (image, mask)

    # Set optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_core.train_model(
        model=model,
        dataset_train_val=(dataset_train, dataset_val),
        dataset_map=preprocess_dataset,
        batch_size=batch_size,
        prefetch=prefetch,
        callbacks=callbacks,
        loss=loss,
        optimizer=optimizer,
        epochs=epochs,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
