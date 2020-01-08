import cv2
import numpy as np


def find_midline(img, midline_mask):
    assert img.shape[:2] == midline_mask.shape[:2]
    mask = midline_mask.copy()
    mask[mask < 0.9] = 0
    mask[mask > 0] = 1
    points = np.argwhere(mask == 1)

    vy, vx, y_rand, x_rand = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).tolist()
    x_rand = x_rand[0]
    y_rand = y_rand[0]
    vx = vx[0]
    vy = vy[0]

    return {'x': x_rand, 'y': y_rand, 'vx': vx, 'vy': vy}


def rotate_and_split(img, midline_mask, page_mask):
    mid = find_midline(img, midline_mask)
    vx = mid['vx']
    vy = mid['vy']
    degree = np.arcsin(vy / np.sqrt(vx**2 + vy**2)) * 180 / np.pi

    degree = -1 * (90 - degree)
    # clockwise rotation. rottate in 4th quadrant.
    M = cv2.getRotationMatrix2D((mid['x'], mid['y']), degree, 1)

    img = cv2.warpAffine(img, M, img.shape[:2], cv2.BORDER_CONSTANT)
    page_mask = cv2.warpAffine(page_mask, M, page_mask.shape[:2], cv2.BORDER_CONSTANT)
    L = int(mid['x'])
    B = 10
    return {
        'left': (img[:, :L + B], page_mask[:, :L + B]),
        'right': (img[:, L - B:], page_mask[:, L - B:]),
    }


def apply_page_mask(img, mask):
    mask = mask[:, :, np.newaxis]
    mask[mask < 0.9] = 0
    mask[mask > 0] = 1
    return img * mask / 255


def get_splitted_img(img, midline_mask, page_mask):
    splitted = rotate_and_split(img, midline_mask, page_mask)
    left = apply_page_mask(*splitted['left'])
    right = apply_page_mask(*splitted['right'])
    return (left, right)
