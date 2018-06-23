"""
Given a list of input image files, it concatenates it into a pdf.
"""

from fpdf import FPDF
import cv2


class ImagesToPdf(FPDF):
    def __init__(self, image_files, fname='Book.pdf'):
        super().__init__()
        self._image_files = image_files
        self._image_area_fraction = 0.9
        self._x = self.w * (1 - self._image_area_fraction) / 2
        self._y = self.h * (1 - self._image_area_fraction) / 2
        self._output_fname = fname

    def convert(self):
        # imagelist is the list with all image filenames
        for image in self._image_files:

            self.add_page()
            height, width, _ = cv2.imread(image).shape
            self.image(image, self._x, self._y, self.w * self._image_area_fraction, self.h * self._image_area_fraction)

        self.output(self._output_fname, "F")
