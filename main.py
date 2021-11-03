
# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 14：46
# @Author  : kiem
# @Email   : jinhui__lu@163.com
# @File    : main.py
# @SoftwareName: PyCharm


import datetime
import shutil
import time
import cfg
import cv2
import os
import function
import numpy as np


def deBackground(image):
    temp = image
    img1 = temp[:, :, 0]
    t, rst = cv2.threshold(img1, 255, 255, cv2.THRESH_TOZERO)
    temp[:, :, 0] = rst
    img2 = temp[:, :, 1]
    t, rst = cv2.threshold(img2, 255, 255, cv2.THRESH_TOZERO)
    temp[:, :, 1] = rst
    img3 = temp[:, :, 2]
    t, rst = cv2.threshold(img3, 30, 255, cv2.THRESH_TOZERO)
    temp[:, :, 2] = rst


def cut_img(image, contours):
    n = 1
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if abs(w-180) > 18 or abs(h-180) > 18 or (h/w) >= 1.1 or (w/h) >= 1.1:
            # cutimg = image[y:y + h, x:x + w, :]
            # cv2.imwrite('data1/' + str(time.time()) + '.JPG', cutimg)
            continue
        cutimg = image[y:y+h, x:x+w, :]
        cv2.imwrite('data1/'+str(n)+'.JPG', cutimg)
        n += 1


def connect(img, height, width):
    result=img.copy()
    for y in range(height):
        for x in range(width):
            if 3 < y < (height - 4) and 3 < x < (width - 4):
                if img[y, x] >= 255:
                    continue
                else:
                    if function.cross(img, y, x, height, width):
                        result[y, x] = 255
    return result


def jurge(path):
    if os.path.exists(path):
        cut = cv2.imread(path)
        cut1 = cut.copy()
        cutG= cv2.cvtColor(cut1, cv2.COLOR_BGR2GRAY)
        th, a = cv2.threshold(cutG, 180, 255, cv2.THRESH_BINARY)
        c, hi = cv2.findContours(a, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(c) == 0:
            return 'GRAY', -3, -3
        if cv2.boundingRect(c[0])[0] < 90:
            cut = cv2.flip(cut, 0)
        cutGRAY = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        t, block = cv2.threshold(cutGRAY, cfg.BLACK_BLOCK_DETECTION_THRESHOLD, 255, cv2.THRESH_BINARY)  # save black_block, to calculate central point
        t, block = cv2.threshold(block, 127, 255, cv2.THRESH_BINARY_INV)
        contours1, hierarchy = cv2.findContours(block, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        frame = []
        j = 0
        for i in range(len(contours1)):
            x, y, w, h = cv2.boundingRect(contours1[i])
            if w * h < 1000 or w/h >= 2 or h/w >= 2:
                continue

            j += 1
            if j == 1 or j == 3:
                w, y, h = 40, y - abs(40 - h), 40  # low right regression
            else:
                x, h, w = x - abs(40 - w), 40, 40  # upper left regression
            # print(w * h)
            frame.append([x, y, w, h])
            # brunt = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
            # image = cv2.drawContours(cut, [brunt], contourIdx=-1, color=(0, 0, 255), thickness=1)
        if len(frame) == 4:
            if frame[1][0] < 90:
                frame[1], frame[2] = frame[2], frame[1]
            # bx1, by1 = (frame[0][0] + frame[2][0] + 40) / 2, (frame[0][1] + frame[2][1] + 40) / 2
            # bx2, by2 = (frame[1][0] + frame[3][0] + 40) / 2, (frame[1][1] + frame[3][1] + 40) / 2
            bx11, by11 = (frame[0][0] + frame[1][0] + 40) / 2, (frame[0][1] + frame[1][1] + 40) / 2
            bx21, by21 = (frame[2][0] + frame[3][0] + 40) / 2, (frame[2][1] + frame[3][1] + 40) / 2
            ########################################################################################
            line = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
            if line.max() < 255:
                line = line+(255-line.max())+40
            t, line = cv2.threshold(line, 200, 255, cv2.THRESH_BINARY)
            max_x, max_y = line.shape[1], line.shape[0]
            l1 = [100, max_y-30, 40, 30]
            l2 = [max_x-30, 103, 30, 40]
            lx1 = function.coordinate(line[max_y-30:max_y, 100:130], 0)+100
            ly1 = function.coordinate(line[103:140, max_x-30:max_x], 1)+103
            l3 = [38, 0, 40, 30]
            l4 = [0, 45, 30, 40]
            lx2 = function.coordinate(line[0:30, 38:78], 0) + 38
            ly2 = function.coordinate(line[45:85, 0:30], 1) + 45
            if lx1 <= 0 or lx2 <= 0 or ly1 <= 0 or ly2 <= 0:
                return 'GRAY', -3, -3
            dis1 = max(abs(bx11-lx1), abs(by11-ly1))
            dis2 = max(abs(bx21-lx2), abs(by21-ly2))
            grade = 'G'
            # cut[int(ly1),int(lx1),:] = 0
            # cut[int(by11),int(bx11),:]=0
            if dis1 >= ng or dis2 >= ng:
                grade = 'NG'
            # elif abs(dis1 - gray) < 0.5 or abs(dis2 - gray) < 0.5:
            #     grade = 'GRAY'
            elif dis1 > cfg.GRAY_THRESHOLD or dis2 > cfg.GRAY_THRESHOLD:
                grade = 'GRAY'
        else:
            grade = 'GRAY'
            dis1 = -2
            dis2 = -2
    else:  #
        grade = 'GRAY'
        dis1 = -1
        dis2 = -1
    # while True:
    #     cv2.imshow('imgGRA', cut)
    #     cv2.imshow('canny', line[45:85, 0:30])
    #     cv2.imshow('ca', line[0:30, 38:78])
    #     if cv2.waitKey(5) & 0xff == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    return grade, dis1, dis2


if __name__ == "__main__":
    for root, dirs, files in os.walk(cfg.FILE_PATH):
        for filename in files:
            if filename[-4:] != '.JPG' and filename[-4:] != '.jpg':
                continue
            t1 = time.time()
            ng = cfg.NG_THRESHOLD
            gray = cfg.GRAY_THRESHOLD
            shutil.rmtree('data1')
            os.mkdir('data1')
            img = cv2.imread(root + '/' + filename)
            imgBGR = cv2.imread(root + '/' + filename)
            deBackground(imgBGR)
            imgGRAY = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(imgGRAY, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            cut_img(img, contours)
            #######################################################

            grade, dis1, dis2 = jurge('data1/1.JPG')
            dis1 = max(dis1, dis2)

            if grade == 'G' and os.path.exists('data1/2.JPG'):
                grade, dis3, dis4 = jurge('data1/2.JPG')
                dis2 = max(dis3, dis4)
            t2 = time.time()

            # while True:
            #     cv2.imshow('imgGRA', cut)
            #     cv2.imshow('canny', line)
            #     if cv2.waitKey(5) & 0xff == ord('q'):
            #         break
            # cv2.destroyAllWindows()


            if not os.path.exists('result/pictures/' + str(datetime.date.today())):
                os.makedirs('result/pictures/' + str(datetime.date.today()) + '/NG/')
                os.makedirs('result/pictures/' + str(datetime.date.today()) + '/G/')
                os.makedirs('result/pictures/' + str(datetime.date.today()) + '/GRAY/')
            if not os.path.exists('result/logs/'):
                os.makedirs('result/logs/')
            if grade == 'NG':
                fp = open('result/logs/'+str(datetime.date.today())+'-NG.txt', 'a')  # 打印NG文件名称
                fp.write(root + filename + '|Dis:' + str(max(dis1, dis2)) +
                         ' |tact:' + str(t2-t1) + ' |time:' + str(datetime.datetime.now()) + '\n')  # 文件内容
                # cv2.imwrite('result/pictures/'+str(datetime.date.today())+'/NG/'+filename, img)  # 保存图片
            elif grade == 'GRAY':
                fp = open('result/logs/' + str(datetime.date.today()) + '-GRAY.txt', 'a')  # 打印GRAY文件名称
                fp.write(root + filename + '|Dis:' + str(max(dis1, dis2)) +
                         ' |tact:' + str(t2-t1) + ' |time:' + str(datetime.datetime.now()) + '\n')  # 文件内容
                # cv2.imwrite('result/pictures/' + str(datetime.date.today()) + '/GRAY/' + filename, img)  # 保存图片
            else:
                fp = open('result/logs/' + str(datetime.date.today()) + '-GOOD.txt', 'a')  # 打印G文件名称
                fp.write(root + filename + '|Dis:' + str(max(dis1, dis2)) +
                         ' |tact:' + str(t2-t1) + ' |time:' + str(datetime.datetime.now()) + '\n')  # 文件内容
                # cv2.imwrite('result/pictures/' + str(datetime.date.today()) + '/G/' + filename, img)  # 保存图片
