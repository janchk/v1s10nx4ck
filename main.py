import numpy as np
import cv2
import copy
from epsareas import epsareas_func

file = "C:/Users/janch/Desktop/validationset/akn.099.060.left.avi"
cap = cv2.VideoCapture(file)
# cap_lz = cv2.VideoCapture(file)
ret1, s_frame = cap.read()  # s_frame frame for show
frame_count = 0
kernel = np.ones((5, 5), np.uint8)
red_mask_mass = []
# red_mask1 = np.array([1])
# red_mask2 = np.array([1])
while True:

    frame_count += 1
    if frame_count % 1 == 0:
        ret1, s_frame = cap.read()
    frame = copy.copy(s_frame)[0:640, 0:1920]  # кадр для рисования, пусть будет

    # Покадровое считывание
    # ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # границы красного цвета BGR
    lower_red = np.array([0, 0, 110], dtype="uint8")
    upper_red = np.array([100, 100, 255], dtype="uint8")


    lower_black = np.array([0, 0, 0], dtype="uint16")
    upper_black = np.array([40, 40, 40], dtype="uint16")

    # покадровая маска
    mask_red = cv2.inRange(cv2.medianBlur(frame, 5), lower_red, upper_red)  # blured
    mask_black = cv2.inRange(frame, lower_black, upper_black)

    # морфологические преобразования
    mask_red1 = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red1 = cv2.morphologyEx(mask_red1, cv2.MORPH_CLOSE, kernel)
    # mask_red1 = cv2.dilate(mask_red1, kernel, iterations=1)

    # магия с последовательным изчезновением красного
    if len(red_mask_mass) < 6:
        red_mask_mass.append(mask_red1)
    else:
        red_mask_mass.pop(0)
        red_mask_mass.append(mask_red1)

    sum_rmask = mask_red1
    if len(red_mask_mass) > 0:
        for rmask in red_mask_mass:
            sum_rmask = cv2.addWeighted(rmask, 0.5, sum_rmask, 1, 1)
    cv2.imshow('summ_rmask', sum_rmask)

    # наложение масок на изображение. Совсем не обязательно, но наглядно.
    black = cv2.bitwise_and(frame, frame, mask=mask_black)
    red = cv2.bitwise_and(frame, frame, mask=sum_rmask)



    # порог
    ret, thresh = cv2.threshold(sum_rmask, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresh", cv2.resize(thresh, (800, 600)))
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # разобраться

    # поиск индекса с наибольшей площадью
    # areas_space = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas_space)
    # cnt = contours[max_index]
    #---------------------------------------------------

    if frame_count % 3 == 0:
        epsareas_func(contours, s_frame, frame_count)


    # for epsarea in contours:
    #     x, y, w, h = cv2.boundingRect(epsarea)
    #     # circle_rad = w/2



    # углы
    # cornerss = cv2.goodFeaturesToTrack(mask_black, 25, 0.01, 10)
    # corners = np.array(cornerss)


    # контур



    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(mask_black, (x, y), 35, 255, -1)

    # # Курги
    # # bgray = cv2.medianBlur(gray, 5)
    # bgray = cv2.resize(gray, (800, 600))
    # circles = cv2.HoughCircles(bgray, cv2.HOUGH_GRADIENT, 1.2, 5)
    # if circles is not None:
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         # draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv2.circle(bgray, (x, y), r, (0, 255, 0), 4)
    #         cv2.rectangle(bgray, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # cv2.imshow("Show", bgray)
    # Наложение маски

    # rs = cv2.resize(gray, (800, 600))
    # rs2 = cv2.resize(mask_red1, (800, 600))
    rs3 = cv2.resize(thresh, (800, 600))
    # rs4 = cv2.resize(red, (800, 600))
    # rs5 = cv2.resize(mask_black, (800, 600))

    # Вывод

    # cv2.imshow('gray', rs)
    # cv2.imshow('mask_red1', rs2)
    # cv2.imshow('thresh', rs3)
    # cv2.imshow('red', rs4)
    # cv2.imshow('black', rs5)

    # Delay между кадрами в мс
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()