"""
Given a list of input image files, it concatenates it into a pdf.
"""

from fpdf import FPDF


class ImagesToPdf(FPDF):
    def __init__(self, image_files, fname='Book.pdf'):
        self._image_files = image_files
        self._x = 0
        self._y = 0
        self._output_fname = fname

    def convert(self):
        # imagelist is the list with all image filenames
        for image in self._image_files:
            self.add_page()
            self.image(image, self._x, self._y)

        self.output(self._output_fname, "F")
