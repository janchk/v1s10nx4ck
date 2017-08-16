import numpy as np
import cv2

cap = cv2.VideoCapture("C:/Users/janch/Desktop/validationset/akn.058.069.left.avi")

while(True):
    # Покадровое считывание
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # границы красного цвета BGR
    lower_red = np.array([0, 0, 150], dtype="uint16")
    upper_red = np.array([135, 135, 250], dtype="uint16")

    lower_black = np.array([0, 0, 0], dtype="uint16")
    upper_black = np.array([110, 110, 110], dtype="uint16")
    # углы
    cornerss = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.array(cornerss)
    # покадровая маска
    mask_red = cv2.inRange(frame, lower_red, upper_red)
    mask_black = cv2.inRange(frame, lower_black, upper_black)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(mask_red, (x, y), 3, 255, -1)


        # Наложение маски
    black = cv2.bitwise_and(frame, frame, mask=mask_black)
    red = cv2.bitwise_and(frame, frame, mask=mask_red)

    rs = cv2.resize(frame, (800, 600))
    rs2 = cv2.resize(mask_red, (800, 600))
    rs3 = cv2.resize(black, (800, 600))
    rs4 = cv2.resize(red, (800, 600))


    # Вывод
    cv2.imshow('frame', rs)
    cv2.imshow('mask', rs2)
    cv2.imshow('black', rs3)
    cv2.imshow('red', rs4)
    # Delay между кадрами в мс
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()