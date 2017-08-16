import numpy as np
import cv2

cap = cv2.VideoCapture("C:/Users/janch/Desktop/validationset/akn.058.069.left.avi")

while(True):
    # Покадровое считывание
    ret, frame = cap.read()
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # границы цвета BGR
    lower_red = np.array([10, 10, 150], dtype="uint16")
    upper_red = np.array([135, 135, 250], dtype="uint16")

    # покадровая маска
    mask = cv2.inRange(frame, lower_red, upper_red)

    # Наложение маски
    res = cv2.bitwise_and(frame, frame, mask=mask)

    rs = cv2.resize(frame, (800, 600))
    rs2 = cv2.resize(mask, (800, 600))
    rs3 = cv2.resize(res, (800, 600))

    # Вывод
    cv2.imshow('frame', rs)
    cv2.imshow('mask', rs2)
    cv2.imshow('res', rs3)
    # Delay между кадрами в мс
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()