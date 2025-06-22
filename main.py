import cv2 as cv
import numpy as np


img = cv.imread("cell.jpg")


#zooms in the image
def zoom_in(image, zoom_factor=1.5):

    h, w = image.shape[:2]

    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = image[y1:y2, x1:x2]
    zoomed_image = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)

    return zoomed_image


def zooming_out(img):
    pass


cv.imshow("Display window", zoom_in(img))


k = cv.waitKey(0)

