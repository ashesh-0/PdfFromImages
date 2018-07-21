import os

import cv2
import numpy as np

from constants import get_constants
from utils import rotate_image


class BoundaryScore:
    def __init__(self, angle, x, score):
        self.angle = angle
        self.x = x
        self.score = score


class BackgroundCleaning:
    def __init__(self, img_files):
        self._img_files = img_files
        self._constants = get_constants()['background_cleaning']
        self._new_files = {}

    def get_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = self._constants['lower_white']
        upper_white = self._constants['upper_white']
        mask = cv2.inRange(hsv, lower_white, upper_white)
        return mask

    def _get_left_score(self, angle, rotated_mask):
        non_zero = np.count_nonzero(rotated_mask, axis=0)
        non_zero_grad = np.diff(non_zero)
        if non_zero[0] > self._constants['left_book_start_threshold']:
            index = 0
            left_score = BoundaryScore(angle, index, abs(non_zero_grad[index]))
        else:
            for fac in [1, 0.8, 0.5, 0.3]:
                page_indices = np.argwhere(non_zero > self._constants['left_book_start_threshold'] * fac)
                if len(page_indices) > 0:
                    break
            if len(page_indices) == 0:
                raise Exception('Big gradient threshold {}. No index matched'.format(
                    self._constants['left_book_start_threshold']))
            index = page_indices[0][0]
            s_index = max(0, index - 5)
            e_index = min(len(non_zero_grad) - 1, index + 5)

            left_score = BoundaryScore(angle, index, max(np.abs(non_zero_grad[s_index:e_index])))

        rotated_mask = rotated_mask.copy()
        s_index = max(0, index - 3)
        e_index = min(rotated_mask.shape[1] - 1, index + 3)

        rotated_mask[:, s_index:e_index] = 0

        return left_score

    def _get_right_score(self, angle, rotated_mask):

        non_zero = np.count_nonzero(rotated_mask, axis=0)
        non_zero_grad = np.diff(non_zero)

        if non_zero[-1] > self._constants['right_book_start_threshold']:
            index = len(non_zero_grad) - 1
            right_score = BoundaryScore(angle, index, abs(non_zero_grad[index]))
        else:
            for fac in [1, 0.8, 0.5, 0.3]:
                page_indices = np.argwhere(non_zero > self._constants['right_book_start_threshold'] * fac)
                if len(page_indices) > 0:
                    break
            if len(page_indices) == 0:
                raise Exception('Big gradient threshold {}. No index matched'.format(
                    self._constants['right_book_start_threshold']))

            index = page_indices[-1][0]
            s_index = max(0, index - 5)
            e_index = min(len(non_zero_grad) - 1, index + 5)

            right_score = BoundaryScore(angle, index, max(np.abs(non_zero_grad[s_index:e_index])))

        return right_score

    def get_score(self, mask, degree):

        rotated_mask = rotate_image(mask, degree)
        left_score = self._get_left_score(degree, rotated_mask.copy())
        right_score = self._get_right_score(degree, rotated_mask.copy())

        return (left_score, right_score)

    def _find_orientation(self, mask):
        angles = self._constants['angles']

        best_left = None
        best_right = None

        for angle in angles:
            left, right = self.get_score(mask, angle)

            if best_left is None or left.score > best_left.score:
                best_left = left

            if best_right is None or right.score > best_right.score:
                best_right = right

        return best_left, best_right

    def get_cleaned_image(self, img_file):
        img = cv2.imread(img_file)
        mask = self.get_mask(img)

        left_boundary, right_boundary = self._find_orientation(mask)
        return self._remove_background(img, left_boundary, right_boundary)

    def _remove_zero_padding(self, rotated_img, orig_shape):
        # remove zero padding
        center = tuple(map(lambda x: x / 2, rotated_img.shape))

        w_start = int(center[0] - orig_shape[0] / 2)
        w_end = int(center[0] + orig_shape[0] / 2)

        h_start = int(center[1] - orig_shape[1] / 2)
        h_end = int(center[1] + orig_shape[1] / 2)

        rotated_img = rotated_img[w_start:w_end, h_start:h_end]
        return rotated_img

    def _remove_background_left(self, img, boundary):
        orig_shape = img.shape
        new_img = rotate_image(img, boundary.angle)
        new_img[:, :(boundary.x - 1)] = 0
        left_cropped = rotate_image(new_img, -1 * boundary.angle)

        return self._remove_zero_padding(left_cropped, orig_shape)

    def _remove_background_right(self, img, boundary):
        orig_shape = img.shape
        new_img = rotate_image(img, boundary.angle)
        new_img[:, (boundary.x + 1):] = 0
        right_cropped = rotate_image(new_img, -1 * boundary.angle)

        return self._remove_zero_padding(right_cropped, orig_shape)

    def _remove_background(self, img, left_boundary, right_boundary):
        shape = img.shape
        img = self._remove_background_right(img, right_boundary)
        assert img.shape == shape

        img = self._remove_background_left(img, left_boundary)
        assert img.shape == shape

        return img

    def _convert_to_file(self, img, img_file):
        direc = os.path.dirname(img_file)
        direc += '/bkgc/'
        if not os.path.exists(direc):
            os.mkdir(direc)

        tokens = os.path.basename(img_file).split('.')
        extension = tokens[-1]

        new_fname = direc + '.'.join(tokens[:-1]) + '_bkgc.' + extension
        cv2.imwrite(new_fname, img)
        self._new_files[img_file] = new_fname
        return new_fname

    def cleanup(self):
        for _, cleaned_file in self._new_files.items():
            if os.path.exists(cleaned_file):
                print('Removing:', cleaned_file)
                os.remove(cleaned_file)

    def _get_cleaned_image_file(self, img_file):
        img = self.get_cleaned_image(img_file)
        return self._convert_to_file(img, img_file)

    def run(self):
        for i, img_file in enumerate(self._img_files):
            self._get_cleaned_image_file(img_file)
            if i % 10 == 1:
                print('[ BackgroundCleaning ]: {}% complete'.format((i * 100) // len(self._img_files)))

        print('[ BackgroundCleaning ]: 100% complete')
        return self._new_files
