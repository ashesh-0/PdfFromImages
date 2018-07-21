"""
Given an image in which  two pages of a book are present, it returns two images where
each page contains one page of the book
"""
import math
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

from utils import rotate_image
from constants import get_constants


class SplitImage:
    def __init__(self, image_files):
        self._image_files = image_files
        self._origin = (0, 0)
        self._y_crop_fraction = 0.01
        self._x_crop_fraction = 0.01
        self._buffer = 40

        self._canny_apertureSize = 3
        self._vertical_text_pointcount_min = 400

        # This will be the directory relative to the location where files are present.
        self._data_directory = 'split_images'
        self._constants = get_constants()['split_image']
        self._max_angle_deviation = 5

    def _get_edges(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_cropped = self._rectangle_crop(gray)
        invert = 255 - gray_cropped
        edges = cv2.Canny(invert, 50, 150, apertureSize=self._canny_apertureSize)
        return edges

    def _rectangle_crop(self, img):
        skip_x = int(img.shape[0] / 2 * self._x_crop_fraction)
        skip_y = int(img.shape[1] / 2 * self._y_crop_fraction)
        img = img[skip_x:-skip_x, skip_y:-skip_y, ]
        return img

    def _get_bin_count(self, rotated_edges):
        non_zero = np.count_nonzero(rotated_edges, axis=0)
        df = pd.Series(non_zero, index=list(range(0, rotated_edges.shape[1])))
        df = df.groupby(np.arange(len(df)) // self._constants['bin_size'], axis=0).sum()
        return df

    def _get_y_from_bin(self, bin):
        return (bin + 0.5) * self._constants['bin_size']

    def _get_y_wrt_center(self, y_wrt_top_left_corner, img):
        return y_wrt_top_left_corner - img.shape[1] // 2

    def _get_y_wrt_left_top_corner(self, y_wrt_center, img):
        return y_wrt_center + img.shape[1] // 2

    def _get_start_end(self, df, index, vertical_text_pointcount_min=None):
        if vertical_text_pointcount_min is None:
            vertical_text_pointcount_min = self._vertical_text_pointcount_min

        start = index
        end = index
        while start > 0:
            if df.iloc[start] <= vertical_text_pointcount_min:
                start = start - 1
            else:
                break

        while end < df.shape[0] - 1:
            if df.iloc[end] <= vertical_text_pointcount_min:
                end = end + 1
            else:
                break
        return (start, end)

    def _near_center(self, index, size):
        mid = size // 2
        deviation = int(0.1 * size)
        return index in range(mid - deviation, mid + deviation)

    def _get_y_min_bin_index(self, bin_count):
        """
        Given the histogram (bin_count) of how many points lie in a y-axis range, it returns a middle bin
        """
        test_bin_count = bin_count.copy()
        while test_bin_count.shape[0] > 0:
            idx = test_bin_count.idxmin()
            if self._near_center(idx, bin_count.shape[0]):
                break
            test_bin_count = test_bin_count.drop([idx], axis=0)

        start, end = self._get_start_end(bin_count, idx, vertical_text_pointcount_min=test_bin_count.loc[idx])
        return (start + end) // 2

    def _get_score(self, rotated_edges):
        bin_count = self._get_bin_count(rotated_edges)
        y_min_bin_index = self._get_y_min_bin_index(bin_count)
        start, end = self._get_start_end(bin_count, y_min_bin_index)
        deriv = bin_count.diff().abs()
        der_left = deriv.iloc[start - 3:start + 3].max()
        der_right = deriv.iloc[end - 3:end + 3].max()
        return max(der_left, der_right)

    def _find_angle(self, edges):
        plt.figure(figsize=(20, 15))

        score_df = pd.DataFrame([], columns=['score'])
        # plot_index = 1
        for angle in range(-self._max_angle_deviation, self._max_angle_deviation):

            rotated_edges = rotate_image(edges, angle)
            score_df.loc[angle] = self._get_score(rotated_edges)

            # plt.subplot(self._max_angle_deviation, 2, plot_index)
            # plt.imshow(rotated_edges)

            # bin_count = self._get_bin_count(rotated_edges)
            # y_val = self._get_y_from_bin(self._get_y_min_bin_index(bin_count))

            # x_min = 0
            # x_max = rotated_edges.shape[0]

            # plt.plot([y_val, y_val], [x_min, x_max])
            # plt.title('Angle {} Score:{}'.format(angle * 180 / math.pi, score_df.loc[angle].values[0]))
            # plot_index += 1

        # plt.show()
        score_df = score_df.sort_values('score')
        return score_df.index[-1]

    def _splitted_filenames(self, image_file):
        tokens = image_file.split('.')
        name = '.'.join(tokens[:-1])
        fname1 = name + '_1.{extension}'.format(extension=tokens[-1])
        fname2 = name + '_2.{extension}'.format(extension=tokens[-1])

        # Create a diretory and move these filepaths inside that directory.
        new_directory = os.path.dirname(image_file) + '/{}/'.format(self._data_directory)
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)

        fname1 = new_directory + os.path.basename(fname1)
        fname2 = new_directory + os.path.basename(fname2)

        return (fname1, fname2)

    def _split_file(self, image_file):
        print('Splitting file {}'.format(image_file))
        img = cv2.imread(image_file)
        img1, img2 = self._split_image(img)
        fname1, fname2 = self._splitted_filenames(image_file)
        cv2.imwrite(fname1, img1)
        cv2.imwrite(fname2, img2)
        return (fname1, fname2)

    def _split_image(self, img):
        # First work is to find the optimal angle to rotate
        edges = self._get_edges(img)
        angle = self._find_angle(edges)

        # After angle is found
        rotated_edges = rotate_image(edges, angle)

        bin_count = self._get_bin_count(rotated_edges)
        y_min_bin_index = self._get_y_min_bin_index(bin_count)
        y_val_in_rectangle = int(self._get_y_from_bin(y_min_bin_index))

        y = self._get_y_wrt_center(y_val_in_rectangle, rotated_edges)

        dst = rotate_image(img, angle)
        y_val = self._get_y_wrt_left_top_corner(y, dst)

        first_image = dst[:, :(y_val + self._buffer), :]
        second_image = dst[:, (y_val - self._buffer):, :]

        return (first_image, second_image)

    def split(self):
        output = {}
        for i, image_file in enumerate(self._image_files):
            f1, f2 = self._split_file(image_file)
            output[image_file] = (f1, f2)
            if i % 10 == 1:
                print('[ SplitImage ]: {}% complete'.format((i * 100) // len(self._image_files)))

        print('[ SplitImage ]: 100% complete')
        return output
