import argparse
import os

from images_to_pdf import ImagesToPdf
from split_image import SplitImage


def main(image_directory):
    fpaths = []
    fnames = os.listdir(image_directory)
    for fname in fnames:
        fpaths.append(image_directory + fname)

    # Assuming that naming of image files has the same order as the page numbers
    fpaths.sort()
    sp = SplitImage(fpaths)

    files_dict = sp.split()
    sorted_files = []
    for fpath in fpaths:
        first_page, second_page = files_dict[fpath]
        sorted_files.append([first_page, second_page])

    ImagesToPdf(sorted_files).convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Image directory')
    args = parser.parse_args()
    main(args.directory)
