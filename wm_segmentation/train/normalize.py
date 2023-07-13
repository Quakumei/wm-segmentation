import typing as tp

import numpy as np
import tensorflow as tf


def get_dataset_normalize(
    feature_f: tp.Callable[[np.ndarray], np.ndarray] = lambda x: x,
    target_f: tp.Callable[[np.ndarray], np.ndarray] = lambda x: x,
) -> tp.Callable[[np.ndarray, np.ndarray], tp.Tuple[np.ndarray, np.ndarray]]:
    def dataset_normalize(
        features: np.ndarray, target: np.ndarray
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        return feature_f(features), target_f(target)

    # Test func on dummy data
    # assert_on_dummy_data(dataset_normalize)

    return dataset_normalize
