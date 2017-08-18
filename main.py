import numpy as np
import cv2

cap = cv2.VideoCapture("C:/Users/janch/Desktop/validationset/akn.014.044.left.avi")

while True:
    # Покадровое считывание
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # границы красного цвета BGR
    lower_red = np.array([0, 0, 110], dtype="uint8")
    upper_red = np.array([80, 80, 255], dtype="uint8")

    # границы красного цвета HSV
    # lower_red_hsv = np.array([150, 0, 0], np.uint8)
    # upper_red_hsv = np.array([250, 155, 255], np.uint8)

    lower_black = np.array([0, 0, 0], dtype="uint16")
    upper_black = np.array([40, 40, 40], dtype="uint16")
    # покадровая маска
    mask_red = cv2.inRange(cv2.medianBlur(frame, 5), lower_red, upper_red)  # blured
    mask_black = cv2.inRange(frame, lower_black, upper_black)

    black = cv2.bitwise_and(frame, frame, mask=mask_black)
    red = cv2.bitwise_and(frame, frame, mask=mask_red)

    # порог
    # b_frame_threshed = cv2.medianBlur(frame_threshed, 5)
    ret, thresh = cv2.threshold(mask_red, 127, 255, cv2.THRESH_TOZERO)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # разобраться

    # тут должен быть перебор по окрестностям
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    for light in contours:  # todo попробовать посмотреть area больше определённого размера
                            # todo исключить, включённые в большие, маленькие кусочки областей
        x, y, w, h = cv2.boundingRect(light)
        # circle_rad = w/2
        cv2.rectangle(frame, (x-10, y-10), (x + w+10, y + 4*h+5), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Show", cv2.resize(frame, (800, 600)))


    # углы
    cornerss = cv2.goodFeaturesToTrack(mask_black, 25, 0.01, 10)
    corners = np.array(cornerss)


    # контур



    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(mask_black, (x, y), 35, 255, -1)

    # Курги
    # bgray = cv2.medianBlur(mask_red, 5)
    # bgray = cv2.resize(bgray, (800, 600))
    # circles = cv2.HoughCircles(mask_red, cv2.HOUGH_GRADIENT, 2, 5)
    # if circles is not None:
    #     # convert the (x, y) coordinates and radius of the circles to integers
    #     circles = np.round(circles[0, :]).astype("int")
    #
    #     # loop over the (x, y) coordinates and radius of the circles
    #     for (x, y, r) in circles:
    #         # draw the circle in the output image, then draw a rectangle
    #         # corresponding to the center of the circle
    #         cv2.circle(mask_red, (x, y), r, (0, 255, 0), 4)
    #         cv2.rectangle(mask_red, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # cv2.imshow("Show", frame)
    # Наложение маски

    rs = cv2.resize(gray, (800, 600))
    rs2 = cv2.resize(mask_red, (800, 600))
    rs3 = cv2.resize(thresh, (800, 600))
    rs4 = cv2.resize(red, (800, 600))
    rs5 = cv2.resize(mask_black, (800, 600))

    # Вывод

    cv2.imshow('gray', rs)
    cv2.imshow('mask_red', rs2)
    cv2.imshow('thresh', rs3)
    cv2.imshow('red', rs4)
    cv2.imshow('black', rs5)

    # Delay между кадрами в мс
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()