import cv2
import numpy as np
kernel = np.ones((5, 5), np.uint8)


def epsareas_func(contours_red, s_frame, frame_number, filename, black):
    # todo go to hsv!!!!!
    # bgr green
    # green_up = np.array([121, 242, 0], dtype='uint8')
    # green_lw = np.array([210, 200, 18], dtype='uint8') # 213 204 7
    # hsv green
    green_up = np.array([99, 255, 255], dtype='uint8')
    green_lw = np.array([50, 100, 100], dtype='uint8')  # 87 100 100

    s_frame = cv2.cvtColor(s_frame, cv2.COLOR_BGR2HSV)

    # Определяем окрестность
    redareas = [cv2.boundingRect(crd) for crd in contours_red]  # массив окрестностей в которых искать зелёный

    # Убираем слишком маленькие области
    areas_space = [cv2.contourArea(c) for c in contours_red]
    ind_big_spaces = [i for i, x in enumerate(areas_space) if x > 30]  # площадь максимально маленькой области
    redareas = [redareas[ind] for ind in ind_big_spaces]

    adv_red_mask = black
    # masked = s_frame
    # grn_masked = cv2.inRange(s_frame, green_lw, green_up)

    for redarea in redareas:
        x, y, w, h = redarea
        # resized_img = thresh[y - int(h/3):y + 3 * h + int(h/3), x - int(w/3):x + w + int(w/3)]
        # cv2.imshow('rszd', resized_img)
        # _, contours_grn, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # greenareas = [cv2.boundingRect(cgrn) for cgrn in contours_grn]

        # s_red_light = w * h
        # adv_red_mask = cv2.rectangle(adv_red_mask, (x - int(w/3), y - int(h/3)),
        #                              (x + w + int(w/3), y + 3 * h + int(h/3)), (255, 255, 255), -1)  # белая область
        adv_red_mask = cv2.rectangle(adv_red_mask, (x - int(w/3), y + int(h)),
                                      (x + w + int(w / 3), y + 3 * h + int(h / 3)), (255, 255, 255), -1)
        cv2.rectangle(s_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # зелёный квадрат
        # mask_white = cv2.inRange(smimg, np.array([254, 254, 254], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))

        # cv2.imshow("Contours", grn_masked)
    masked = cv2.bitwise_and(s_frame, s_frame, mask=adv_red_mask)
    grn_masked = cv2.inRange(masked, green_lw, green_up)
    # grn_masked = cv2.medianBlur(grn_masked, 5)
    grn_masked = cv2.morphologyEx(grn_masked, cv2.MORPH_OPEN, kernel)
    grn_masked = cv2.morphologyEx(grn_masked, cv2.MORPH_CLOSE, kernel)
    ret, thresh = cv2.threshold(grn_masked, 127, 255, cv2.THRESH_BINARY)
    _, contours_grn, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # greenareas = cv2.boundingRect(contours_grn)
    # показ картинки
    # cv2.putText(masked, ("%s" % frame_number), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255])
    # cv2.imshow(filename, masked)
    # cv2.imshow("GRMasked", grn_masked)
    if np.any([contours_grn[i].size > 5 for i in range(0, len(contours_grn))]):
        return frame_number + 3