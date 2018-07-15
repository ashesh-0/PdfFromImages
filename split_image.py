"""
Given an image in which  two pages of a book are present, it returns two images where
each page contains one page of the book
"""
import math
import os

import cv2
import numpy as np
import pandas as pd
from sklearn import linear_model

from utils import rotate_image
from constants import get_constants


class SplitImage:
    def __init__(self, image_files):
        self._image_files = image_files
        self._origin = (0, 0)
        self._y_crop_fraction = 0.7
        self._x_crop_fraction = 0.4
        self._buffer = 50

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

    # def _get_cordinates(self, edges):
    #     points = []
    #     for i in range(edges.shape[0]):
    #         for j in range(edges.shape[1]):
    #             if edges[i, j]:
    #                 points.append((i, j))
    #     return points

    def _rectangle_crop(self, img):
        skip_x = int(img.shape[0] / 2 * self._x_crop_fraction)
        skip_y = int(img.shape[1] / 2 * self._y_crop_fraction)
        img = img[skip_x:-skip_x, skip_y:-skip_y, ]
        return img

    # def _rotate(self, row, angle):
    #     ox, oy = self._origin

    #     px = row['x']
    #     py = row['y']

    #     qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    #     qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    #     return qx, qy

    def _get_bin_count(self, rotated_edges):
        non_zero = np.count_nonzero(rotated_edges, axis=0)
        df = pd.Series(non_zero, index=list(range(0, rotated_edges.shape[1])))
        df = df.groupby(np.arange(len(df)) // self._constants['bin_size'], axis=0).sum()
        return df

    def _get_y_from_bin(self, bin):
        return (bin + 0.5) * self._constants['bin_size']

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

    # def _find_y(self, img_df):
    #     bin_count = self._get_bin_count(img_df)

    #     idx = bin_count.idxmin()
    #     y_min_index = bin_count.index.tolist().index(idx)

    #     start, end = self._get_start_end(bin_count, y_min_index)

    #     return int((start + end) / 2)

    def _near_center(self, index, size):
        mid = size // 2
        deviation = int(0.1 * size)
        return index in range(mid - deviation, mid + deviation)

    def _get_y_min_bin_index(self, bin_count, y_max):
        """
        Given the histogram (bin_count) of how many points lie in a y-axis range, it returns a middle bin
        """
        test_bin_count = bin_count.copy()
        while test_bin_count.shape[0] > 0:
            idx = test_bin_count.idxmin()
            y_min_index = self._get_y_from_bin(idx)
            if self._near_center(y_min_index, y_max):
                break
            test_bin_count = test_bin_count.drop([idx], axis=0)

        start, end = self._get_start_end(bin_count, idx, vertical_text_pointcount_min=bin_count.min())
        return (start + end) // 2

    def _get_score(self, rotated_edges):
        bin_count = self._get_bin_count(rotated_edges)
        y_min_bin_index = self._get_y_min_bin_index(bin_count, rotated_edges.shape[1])
        start, end = self._get_start_end(bin_count, y_min_bin_index)
        deriv = bin_count.diff().abs()
        der_left = deriv.iloc[start - 3:start + 3].max()
        der_right = deriv.iloc[end - 3:end + 3].max()
        return max(der_left, der_right)

    def _find_angle(self, edges):
        # plt.figure(figsize=(20, 15))

        score_df = pd.DataFrame([], columns=['score'])
        # plot_index = 1
        for angle in range(-self._max_angle_deviation, self._max_angle_deviation):

            rotated_edges = rotate_image(edges, angle)
            score_df.loc[angle] = self._get_score(rotated_edges)

            # plt.subplot(self._max_angle_deviation, 2, plot_index)
            # plt.scatter(rotated_df['x'], rotated_df['y'], )

            # bin_count = self._get_bin_count(rotated_df)
            # y_val = bin_count.index[self._get_y_min_bin_index(bin_count)].mid

            # x_min = rotated_df.x.min()
            # x_max = rotated_df.x.max()

            # plt.plot([x_min, x_max], [y_val, y_val])
            # plt.title('Angle {} Score:{}'.format(angle *180/math.pi, score_df.loc[angle].values[0]))
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
        y_min_bin_index = self._get_y_min_bin_index(bin_count, rotated_edges.shape[1])
        y_val_in_rectangle = int(self._get_y_from_bin(y_min_bin_index))

        y_val_outside_rectangle_original = self._y_crop_fraction * img.shape[1] / 2
        x_val_outside_rectangle_original = self._x_crop_fraction * img.shape[0] / 2

        hypotenus = np.sqrt(
            np.power(y_val_outside_rectangle_original, 2) + np.power(x_val_outside_rectangle_original, 2))
        phi = math.atan(y_val_outside_rectangle_original / x_val_outside_rectangle_original)

        y_val_outside_rectangle = hypotenus * math.sin(angle + phi)
        if angle < 0:
            y_val_outside_rectangle = y_val_outside_rectangle - img.shape[0] * math.sin(angle)
        y_val = int(y_val_outside_rectangle + y_val_in_rectangle)

        angle_degree = angle / math.pi * 180

        M = cv2.getRotationMatrix2D((x_val_outside_rectangle_original, y_val_outside_rectangle_original), angle_degree,
                                    1)
        dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        first_image = dst[:, :(y_val + self._buffer), :]
        second_image = dst[:, (y_val - self._buffer):, :]

        return (first_image, second_image)

    def split(self):
        output = {}
        for image_file in self._image_files:
            f1, f2 = self._split_file(image_file)
            output[image_file] = (f1, f2)

        return output
