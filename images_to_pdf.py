"""
Given a list of input image files, it concatenates it into a pdf.
"""

from fpdf import FPDF
import cv2


class ImagesToPdf(FPDF):
    def __init__(self, image_files, fname='Book.pdf'):
        super().__init__()
        self._image_files = image_files
        self._x = 0
        self._y = 0
        self._output_fname = fname

    def convert(self):
        # imagelist is the list with all image filenames
        for image in self._image_files:

            self.add_page()
            height, width, _ = cv2.imread(image).shape
            # 1 px = 0.264583 mm (FPDF default is mm)
            # pdf.image(imageFile, 0, 0, float(width * 0.264583), float(height * 0.264583))
            self.image(image, self._x, self._y, width / 5, height / 5)

        self.output(self._output_fname, "F")
