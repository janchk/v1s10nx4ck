import numpy as np
import cv2
import time
import os
import re


def epsareas_func(contours, s_frame, frame_number):

    green_up = np.array([147, 252, 101], dtype='uint8')
    green_lw = np.array([58, 148, 43], dtype='uint8')

    black = np.zeros([640, 1920, 1], dtype="uint8")  # для фильтра
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
        # cv2.imshow("adv", adv_red_mask)
        masked = cv2.bitwise_and(s_frame, s_frame, mask=adv_red_mask)
        grn_masked = cv2.inRange(masked, green_lw, green_up)


        cv2.imshow("Masked", masked)
        # cv2.imshow("GRMasked", grn_masked)
        ret, thresh = cv2.threshold(grn_masked, 127, 255, cv2.THRESH_BINARY)
        _, contours_grn, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("Contours", grn_masked)
        if [cv2.contourArea(cgrn) > 3 for cgrn in contours_grn]:
            cap.release()
            cv2.destroyAllWindows()
            return frame_number


def main(frame_count, kernel, red_mask_mass):
    while True:
        # границы красного цвета BGR
        lower_red = np.array([0, 0, 110], dtype="uint8") # 198
        upper_red = np.array([100, 100, 255], dtype="uint8")

        frame_count += 1
        if frame_count % 2 == 0:
            ret1, s_frame = cap.read()
            try:
                s_frame = s_frame[0:640, 0:1920]
            except: return -1

            # покадровая маска
            mask_red = cv2.inRange(cv2.medianBlur(s_frame, 5), lower_red, upper_red)  # blured

            # морфологические преобразования
            mask_red1 = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_red1 = cv2.morphologyEx(mask_red1, cv2.MORPH_CLOSE, kernel)

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
            # cv2.imshow('summ_rmask', sum_rmask)

            # порог
            ret, thresh = cv2.threshold(sum_rmask, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow("Thresh", cv2.resize(thresh, (800, 600)))
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # разобраться

            res = epsareas_func(contours, s_frame, frame_count)
            if res:
                return res/2

        # frame = copy.copy(s_frame)[0:640, 0:1920]  # кадр для рисования, пусть будет

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(-1)
            break

    cap.release()
    cv2.destroyAllWindows()
    return -1

start_time = time.time()
text_file = "C:/Users/janch/PycharmProjects/visionhack/output1.txt"
r = re.compile(".*avi")
files = [f for f in os.listdir('C:/Users/janch/Desktop/validationset')]
# files = filter(r.match, files)
for file in files:
    filepath = ('C:/Users/janch/Desktop/validationset/%s' % file)
    cap = cv2.VideoCapture(filepath)
    ret1, s_frame = cap.read()  # s_frame frame for show
    # cv2.imshow("s_frm", s_frame)
    frame_count = 0
    kernel = np.ones((5, 5), np.uint8)
    red_mask_mass = []
    out = main(frame_count, kernel, red_mask_mass)
    if out <= 30:
        out = -1
    with open(text_file, 'a') as text:
        text.write("%s %s \n" % (file, out))

# time
print("--- %s seconds ---" % (time.time() - start_time))




