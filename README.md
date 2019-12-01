# BookImageSplitter
Converts list of 2 page images into 2 images for one page each.

<!-- :---------: | :-----------: | ----------: -->
![RawImage](https://drive.google.com/uc?export=view&id=1MHokX938oPe9COpjTKM4BesBd-td8RcT) ![](https://docs.microsoft.com/en-us/windows/win32/uxguide/images/vis-icons-image6.png) ![Image1](https://drive.google.com/uc?export=view&id=1YkB_b4mi4-CeOwJBjyBhMl5SjKx-FOPJ) ![](https://docs.microsoft.com/en-us/windows/win32/uxguide/images/vis-icons-image6.png) ![Image2](https://drive.google.com/uc?export=view&id=1MZrYKsL4_08OfPy6QD3SqGi7mLqyKae0) ![](https://drive.google.com/uc?export=view&id=1a4_a3u7e_pBRj0vzGQhFUhyP8his3jnX) ![Image3](https://drive.google.com/uc?export=view&id=1KTBCrRtjhbC2vII2ZquNgn1EjmG0tiQr)

# How to use
`python run.py IMAGE_DIRECTORY`

In IMAGE_DIRECTORY, all images which needs to be separated into two should be present.

# TODO
1. High resolution Pdf. Currently, low resolution pdf is getting formed.
2. Affine transformation on final output image so as to make the text horizontal.
3. Automatic detection of page numbers.
