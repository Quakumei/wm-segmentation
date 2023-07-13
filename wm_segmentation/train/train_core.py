import datetime
import inspect
import typing as tp

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm.keras import TqdmCallback

# import logging # Doesn't work for some reason


def get_default_callbacks(
    model_name: str,
    dataset_name: str,
    logs_dir: str = "/logs",
    checkpoints_dir: str = "/models",
    loss: str = "val_mse",
) -> tp.List[tf.keras.callbacks.Callback]:
    print("Setting tensorboard callback")

    log_dir = (
        f"{logs_dir}/{dataset_name}/fit/"
        + model_name
        + "__"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    checkpoint_path = (
        f"{checkpoints_dir}/{dataset_name}/{model_name}/"
        + "best_model_{}.h5".format("v19")
    )

    print("log_dir: {}".format(log_dir))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        write_grads=True,
    )

    print("Setting early stopping callback")
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=10)

    print("Setting model checkpoint callback")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    print("Setting tqdm callback")
    tqdm_callback = TqdmCallback(verbose=2)

    print("Setting reduce LR callback")
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=0.00001
    )

    return [
        tensorboard_callback,
        early_stopping_callback,
        model_checkpoint_callback,
        tqdm_callback,
        reduce_lr_callback,
    ]


def get_train_val_from_tfds(
    dataset_name: str,
    data_dir: str,
    is_supervised: bool = True,
    train_slice: str = ":99%",
    validate_slice: str = ":99%",
):
    dataset_train = tfds.load(
        dataset_name,
        split=f"train[{train_slice}]",
        shuffle_files=True,
        as_supervised=is_supervised,
        data_dir=data_dir,
    )
    dataset_val = tfds.load(
        dataset_name,
        split=f"validate[{validate_slice}]",
        shuffle_files=False,
        as_supervised=is_supervised,
        data_dir=data_dir,
    )
    return dataset_train, dataset_val


def train_model(
    # Model settings
    model,
    # Dataset settings
    dataset_train_val: tp.Tuple[tf.data.Dataset, tf.data.Dataset],
    dataset_map: tp.Callable[
        [np.ndarray, np.ndarray], tp.Tuple[np.ndarray, np.ndarray]
    ] = lambda x: x,
    # Batch size
    batch_size: int = 16,
    prefetch: int = 1,
    # Callbacks
    callbacks: tp.List[tf.keras.callbacks.Callback] = [],
    # Train loop settings
    loss: str | tp.Any = "mse",
    metrics: tp.List[str] = ["mse", "mae"],
    epochs: int = 10,
    # Optimizer settings
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
):
    # Prepare dataset
    ds_train, ds_val = [
        ds.map(dataset_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
        .prefetch(prefetch)
        for ds in dataset_train_val
    ]

    # Compile and fit
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    # Log params
    log_string = "Training model with the following parameters:\n"
    interesting_params = {
        "model": model,
        "batch_size": batch_size,
        "prefetch": prefetch,
        "callbacks": callbacks,
        "loss": loss,
        "metrics": metrics,
        "epochs": epochs,
        "optimizer": optimizer,
    }
    try:
        interesting_params.update(("ds_map", inspect.getsource(dataset_map)))
    except:
        print("Couldn't get normalized function source code")

    for key, value in interesting_params.items():
        log_string += "{}: {}\n".format(key, value)

    print(log_string)

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
