import cv2 as cv
import numpy as np
import random

def sobel_gra(image):
    im = image.copy()
    # 求X方向梯度
    sobelx = cv.Scharr(im, cv.CV_64F, 1, 0)
    gradx = cv.convertScaleAbs(sobelx)  # 由于算完的图像有正有负，所以对其取绝对值
    # 求Y方向梯度
    sobely = cv.Scharr(im, cv.CV_64F, 0, 1)
    grady = cv.convertScaleAbs(sobely)
    # 合并梯度(近似),计算两个图像的权值和，dst = src1*alpha + src2*beta + gamma
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    # show("sobel", gradxy)
    return gradxy


def lap_gra(image):
    im = image.copy()
    dst = cv.Laplacian(im, cv.CV_16S, ksize=3)
    dst = cv.convertScaleAbs(dst)
    # show('laplacian', dst)
    return dst


def enhance_lap(image, y, x):
    left = False
    right = False
    up = False
    down = False
    for rows in range(y - 2, y + 3):
        for cols in range(x - 2, x + 3):
            if image[rows, cols] == 255:
                if rows < y:
                    left = True
                if rows > y:
                    right = True
                if cols < x:
                    up = True
                if cols > x:
                    down = True
    return (left & right) | (up & down)

def cross(image, y, x, height, width):
    win_width = 8
    if image[y, x] > 250:
        return 0
    # left
    left = False
    leftedge = x - win_width if x - win_width >= 0 else 0
    for cols in range(x - 1, leftedge - 1, -1):
        if image[y, cols] == 255:
            left = True
            break
        else:
            left = False

    # right
    right = False
    rightedge = x + win_width if x + win_width < width else width - 1
    for cols in range(x + 1, rightedge + 1):
        if image[y, cols] == 255:
            right = True
            break
        else:
            right = False

    # up
    up = False
    upedge = y - win_width if y - win_width >= 0 else 0
    for rows in range(y - 1, upedge - 1, -1):
        if image[rows, x] == 255:
            up = True
            break
        else:
            up = False
    # down
    down = False
    downedge = y + win_width if y + win_width < height else height - 1
    for rows in range(y + 1, downedge + 1):
        if image[rows, x] == 255:
            down = True
            break
        else:
            down = False

    return (left & right) | (up & down)

def fetch_backgroundnoise(image, pre_thresh):
    totalcount = 50
    totalmax = 0
    compare = np.zeros(pre_thresh + 1, np.uint8)
    while totalcount > 0:
        y = random.randint(2, image.shape[0] - 3)
        x = random.randint(2, image.shape[1] - 3)
        flag = True
        count = 0
        max = 0
        for rows in range(y - 2, y + 3):
            if flag is False:
                break
            for cols in range(x - 2, x + 3):
                if flag is False:
                    break
                if image[rows, cols] > pre_thresh:
                    flag = False
                else:
                    if image[rows, cols] > max:
                        max = image[rows, cols]
                    # count += 1
        if flag is True:
            totalmax = totalmax if totalmax > max else max
            totalcount -= 1
            compare[max] += 1
    print("frequentset thresh", np.argwhere(compare == np.max(compare))[0][0])
    print("maxest thresh", totalmax)
    return np.argwhere(compare == np.max(compare))[0][0]


def coordinate(img, axis):
    if img.max() < 127:
        return -1
    if axis == 0:
        x_list = np.argwhere(img > 127)
        return np.mean(x_list[:, 1])
    else:
        y_list = np.argwhere(img > 127)
        return np.mean(y_list[:, 0])