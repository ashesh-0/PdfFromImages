import numpy as np


def get_constants():
    return {
        'background_cleaning': {
            'lower_white': np.array([0, 0, 0], dtype=np.uint8),
            'upper_white': np.array([180, 10, 255], dtype=np.uint8),
            'angles': list(range(-10, 10)),
            'left_book_start_threshold': 800,
            'right_book_start_threshold': 800,
            'donot_alter_angle_threshold': 1000,
            'ignore_fraction': 0.25,
            'gradient': 10,
        },
        'split_image': {
            'bin_size': 5,
        }
    }
