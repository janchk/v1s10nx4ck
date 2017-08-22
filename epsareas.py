import cv2
import numpy as np


def epsareas_func(contours, s_frame, frame_number, filename, black):

    green_up = np.array([166, 255, 45], dtype='uint8')
    green_lw = np.array([122, 213, 0], dtype='uint8') # 213 204 7


    # Определяем окрестность
    epsareas = [cv2.boundingRect(a) for a in contours]

    # Убираем слишком маленькие области
    areas_space = [cv2.contourArea(c) for c in contours]
    ind_big_spaces = [i for i, x in enumerate(areas_space) if x > 20] # площадь максимально маленькой области
    epsareas = [epsareas[ind] for ind in ind_big_spaces]


    for epsarea in epsareas:

        x, y, w, h = epsarea
        adv_red_mask = cv2.rectangle(black, (x - 7, y - 7), (x + w + 5, y + 3 * h + 2), (255, 255, 255), -1)  # белая область
        # cv2.rectangle(s_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # зелёный квадрат
        # mask_white = cv2.inRange(smimg, np.array([254, 254, 254], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))
        # cv2.imshow("Show", cv2.resize(s_frame, (800, 600)))
        masked = cv2.bitwise_and(s_frame, s_frame, mask=adv_red_mask)
        grn_masked = cv2.inRange(masked, green_lw, green_up)

        # показ картинки
        cv2.putText(masked, ("%s" % frame_number), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255])
        cv2.imshow(filename, masked)

        # cv2.imshow("GRMasked", grn_masked)
        ret, thresh = cv2.threshold(grn_masked, 127, 255, cv2.THRESH_BINARY)
        _, contours_grn, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("Contours", grn_masked)
        if [cv2.contourArea(cgrn) > 3 for cgrn in contours_grn]:
            return frame_number