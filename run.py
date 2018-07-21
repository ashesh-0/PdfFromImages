import argparse
import os

from images_to_pdf import ImagesToPdf
from split_image import SplitImage
from background_cleaning import BackgroundCleaning


def get_cleaned_files(fpaths):
    # Remove background
    cleaner = BackgroundCleaning(fpaths.copy())
    output = cleaner.run()
    cleaned_files = []
    for fpath in fpaths:
        cleaned_files.append(output[fpath])
    return cleaned_files


def get_splitted_files(cleaned_files):
    # Split image
    sp = SplitImage(cleaned_files)

    files_dict = sp.split()
    splitted_files = []
    for fpath in cleaned_files:
        first_page, second_page = files_dict[fpath]
        splitted_files += [first_page, second_page]

    return splitted_files


def get_page_ordered_files(image_directory):
    fpaths = []
    fnames = os.listdir(image_directory)
    title_file = None
    for fname in fnames:

        fpath = image_directory + fname

        # Skip directories
        if os.path.isdir(fpath):
            continue

        if 'title' in fname and title_file is not None:
            title_file = fpath
        else:
            fpaths.append(fpath)

    # Assuming that naming of image files has the same order as the page numbers
    fpaths.sort()
    return (title_file, fpaths)


def main(image_directory):
    if '/' != image_directory[-1]:
        image_directory += '/'

    # Get files ordered by page number.
    title_file, fpaths = get_page_ordered_files(image_directory)
    # Remove background from them.
    cleaned_files = get_cleaned_files(fpaths.copy())

    # Split the page into two.
    splitted_files = get_splitted_files(cleaned_files)

    if title_file is not None:
        splitted_files = [title_file] + splitted_files

    # Convert the files to pdf
    ImagesToPdf(splitted_files).convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Image directory')
    args = parser.parse_args()
    main(args.directory)
