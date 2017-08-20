import cv2
import numpy as np
import sys

green_up = np.array([147, 252, 101], dtype='uint8')
green_lw = np.array([47, 148, 3], dtype='uint8')

def epsareas_func(contours, s_frame, frame_number):

    black = np.zeros([1080, 1920, 1], dtype="uint8")  # для фильтра
    # Определяем окрестность
    epsareas = [cv2.boundingRect(a) for a in contours]

    # Убираем слишком маленькие области
    areas_space = [cv2.contourArea(c) for c in contours]
    ind_big_spaces = [i for i, x in enumerate(areas_space) if x > 40]
    epsareas = [epsareas[ind] for ind in ind_big_spaces]


    for epsarea in epsareas:

        x, y, w, h = epsarea
        adv_red_mask = cv2.rectangle(black, (x - 10, y - 10), (x + w + 10, y + 3 * h + 5), (255, 255, 255), -1)  # белая область
        # cv2.rectangle(s_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # зелёный квадрат
        # mask_white = cv2.inRange(smimg, np.array([254, 254, 254], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))
        # cv2.imshow("Show", cv2.resize(s_frame, (800, 600)))
        masked = cv2.bitwise_and(s_frame, s_frame, mask=adv_red_mask)
        grn_masked = cv2.inRange(masked, green_lw, green_up)


        cv2.imshow("Masked", cv2.resize(masked, (800,600)))
        # cv2.imshow("GRMasked", grn_masked)
        ret, thresh = cv2.threshold(grn_masked, 127, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            print(frame_number)
            sys.exit()

        # cv2.imshow("LZ_Show", cv2.resize(frame, (800, 600)))