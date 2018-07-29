import numpy as np


def get_constants():
    return {
        'background_cleaning': {
            'lower_white': np.array([0, 0, 0], dtype=np.uint8),
            'upper_white': np.array([180, 10, 255], dtype=np.uint8),
            'angles': list(range(-15, 15)),
            'left_book_start_threshold': 800,
            'right_book_start_threshold': 800,
            'donot_alter_angle_threshold': 1000,
            'ignore_fraction': 0.25,
            'gradient': 25,
            'mean_factor': 0.7,
            'score_multiplier': [1, 0.8, 0.5, 0.3, 0.1],
        },
        'split_image': {
            'bin_size': 5,
        }
    }
