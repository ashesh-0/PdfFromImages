# BookImageSplitter
Converts list of 2 page images into 2 images for one page each.

## Using CNNs.
![RawImage](https://drive.google.com/uc?export=view&id=15xw8gwcdf8fpCWu1sdubxRJhHtzCUmZE) ![](https://docs.microsoft.com/en-us/windows/win32/uxguide/images/vis-icons-image6.png) ![Image2](https://drive.google.com/uc?export=view&id=1xzbJepx5tv5n789UWOWkN3jfEjn6G4Ta) ![](https://drive.google.com/uc?export=view&id=1a4_a3u7e_pBRj0vzGQhFUhyP8his3jnX) ![Image3](https://drive.google.com/uc?export=view&id=1mG-M3CFGIoWd6hxUL7AwO_iDmBMrKJ8C)

1. Using transfer learning, CNN is trained to do pixel segmentation of book area vs the background. [Notebook](./PagePixelSegmentation.ipynb)
2. Using transfer learning, CNN is trained to find out the mid line of the two pages. Mid line is the line which divides the left page from the right page. [Notebook](./PageMidBoundaryPixelSegmentation.ipynb)
3. Using weighted linear regression (with huber loss so as to ignore outliers), mid line is finetuned. This line is used both to rotate the image so that it becomes vertical in resulting image. Also, this is used to divide the image into left page and right page. [Notebook](./BookSegmentation.ipynb)

## TODO
1. Handle rough page edges. Make them smoothe.

## Traditional Computer Vision Techniques.
<!-- :---------: | :-----------: | ----------: -->
![RawImage](https://drive.google.com/uc?export=view&id=1MHokX938oPe9COpjTKM4BesBd-td8RcT) ![](https://docs.microsoft.com/en-us/windows/win32/uxguide/images/vis-icons-image6.png) ![Image1](https://drive.google.com/uc?export=view&id=1YkB_b4mi4-CeOwJBjyBhMl5SjKx-FOPJ) ![](https://docs.microsoft.com/en-us/windows/win32/uxguide/images/vis-icons-image6.png) ![Image2](https://drive.google.com/uc?export=view&id=1MZrYKsL4_08OfPy6QD3SqGi7mLqyKae0) ![](https://drive.google.com/uc?export=view&id=1a4_a3u7e_pBRj0vzGQhFUhyP8his3jnX) ![Image3](https://drive.google.com/uc?export=view&id=1KTBCrRtjhbC2vII2ZquNgn1EjmG0tiQr)

### How to use
`python run.py IMAGE_DIRECTORY`

In IMAGE_DIRECTORY, all images which needs to be separated into two should be present.

# TODO
1. Affine transformation on final output image so as to make the text horizontal.
2. Automatic detection of page numbers.
3. High resolution Pdf. Currently, low resolution pdf is getting formed.

