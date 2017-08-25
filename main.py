import numpy as np
import cv2
import time
import os
from epsareas import epsareas_func



def main(frame_count, kernel, red_mask_mass, filename):
    frame_skip = 1 # не работает
    frame_size_mult = 0.8

    # границы красного цвета BGR
    lower_red = np.array([0, 0, 120], dtype="uint8")  # 198, 110
    upper_red = np.array([100, 100, 255], dtype="uint8")

    # границы красного цвета HSV
    # lower_red = np.array([0, 0, 120], dtype="uint8")  # 198, 110
    # upper_red = np.array([100, 100, 255], dtype="uint8")


    lower_black = 0
    upper_black = 0
    while True:
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        if frame_count % frame_skip == 0:
            try:
                ret1, s_frame = cap.read()
                # s_frame = cv2.resize(s_frame, (int(cap_w * frame_size_mult), int(cap_h * frame_size_mult)),
                #                      interpolation=cv2.INTER_CUBIC)
                s_frame = s_frame[0:int(cap_h*0.7), 0:int(cap_w)]
            except:
                return -1

            black = np.zeros([s_frame.shape[:2][0], s_frame.shape[:2][1], 1], dtype="uint8")  # для фильтра черный фон

            # покадровая маска
            mask_red = cv2.inRange(cv2.medianBlur(s_frame, 5), lower_red, upper_red)  # blured

            # морфологические преобразования
            mask_red1 = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_red1 = cv2.morphologyEx(mask_red1, cv2.MORPH_CLOSE, kernel)

            # магия с последовательным изчезновением красного
            if len(red_mask_mass) < 4:  # задержка в кадрах перед исчезновением красной области
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
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # разобраться

            res = epsareas_func(contours, s_frame, frame_count, filename, black)
            if res:
                return res
            frame_count += frame_skip  #
        else:
            # ret1, s_frame = cap.read()
            frame_count += 1  #

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(-1)
            break

    return -1

start_time = time.time()
text_file = "C:/Users/janch/PycharmProjects/visionhack/output1_rev3.txt"
# r = re.compile(".*avi")
files = [f for f in os.listdir('C:/Users/janch/Desktop/validationset')]
# files = filter(r.match, files)

for file in files:
    filepath = ('C:/Users/janch/Desktop/validationset/%s' % file)

    cap_timestart = time.time()

    cap = cv2.VideoCapture(filepath)
    cap_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    ret1, s_frame = cap.read()

    # cv2.imshow("s_frm", s_frame)
    frame_count = 0
    kernel = np.ones((5, 5), np.uint8)
    red_mask_mass = []

    out = main(frame_count, kernel, red_mask_mass, file)

    if out <= 15:
        out = -1
    with open(text_file, 'a') as text:
        text.write("%s %s \n" % (file, out))
    cap.release()
    cv2.destroyAllWindows()
    print("--- %s seconds ---  %s" % (time.time() - cap_timestart, file))

# time
print("--- %s seconds at all ---" % (time.time() - start_time))




