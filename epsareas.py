import cv2
import numpy as np


def epsareas_func(contours, s_frame):
    # Определяем окрестность
    epsareas = [cv2.boundingRect(a) for a in contours]

    # Убираем слишком маленькие области
    areas_space = [cv2.contourArea(c) for c in contours]
    ind_big_spaces = [i for i, x in enumerate(areas_space) if x > 15]
    epsareas = [epsareas[ind] for ind in ind_big_spaces]

    # Чистим входящие маленькие области
    for i in epsareas:
        for j in epsareas:
            xi, yi, wi, hi = i
            xj, yj, wj, hj = j
            if (xi-10) < xj <= (xi+wi+10) and (yi-10) < yj <= (yi + 4*hi + 5):
                epsareas.remove(j)

    for epsarea in epsareas:
        x, y, w, h = epsarea
        cv2.rectangle(s_frame, (x - 10, y - 10), (x + w + 10, y + 4 * h + 5), (0, 0, 255), 2)  # красный
        cv2.rectangle(s_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # зелёный
        cv2.imshow("Show", cv2.resize(s_frame, (800, 600)))
        # cv2.imshow("LZ_Show", cv2.resize(frame, (800, 600)))