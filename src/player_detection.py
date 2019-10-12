# Install imutils to have useful functions

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('../Videos/video_long.mp4')

while cap.isOpened():
    ret, res = cap.read()
    if ret:
        # resizing the Video frame by frame
        frame = imutils.resize(res, 960, 540)
    else:
        break

    # use grayscale of frame for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in rects:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("Player Detection", frame)

    if cv2.waitKey(30) != -1:
        break

cap.release()
cv2.destroyAllWindows()

# Fazit:
#
# SEHR unperformant, Ergebnisse dafür ziemlich gut
# Korb wird als Mensch erkannt
# Für den realen Anwendungsfall viel zu langsam und performance-aufwendig
# Echtzeit-Verarbeitung wahrscheinlich nicht möglich
#
# Mit vortrainierter KI (YOLO) auch nicht möglich. 30min zur Video-Analyse sind zu lang.
# Material höchstens danach verwendbar.
#
#









