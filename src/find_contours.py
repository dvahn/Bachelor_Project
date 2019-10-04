import numpy as np
import cv2

def find_countours(roi):

    im2, cnts, hierarchy = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3] # get largest three contour area
    rects = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h >= 5:
            # if height is enough
            # create rectangle for bounding
            rect = (x, y, w, h)
            rects.append(rect)
            #cv2.rectangle(roi_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)

    #return (roi_copy, rects)
    return rects

cap = cv2.VideoCapture('../Videos/video_long.mp4')
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
framesPerSecond = cap.get(cv2.CAP_PROP_FPS)

# Values for detection of red attachments on player
hueLowerThresholdRED = 174
hueUpperThresholdRED = 177
saturationThreshold = 120

# Values for detection of green sheet on basket
hueLowerThresholdGREEN = 69
hueUpperThresholdGREEN = 80

while cap.isOpened():
    ret, res = cap.read()
    if ret:
        # resizing the Video frame by frame
        frame = cv2.resize(res, (960, 540))
    else:
        break

    # Umwandlung in HSV
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Eingeschränkter Suchbereich für rot
    constrainedFrame = frame[120:300, 120:270]
    hsvFrameConstrained = cv2.cvtColor(constrainedFrame, cv2.COLOR_BGR2HSV)

    # Aufspaltung in 3 Graustufenbilder für H, S, V mit:
    h, s, v = cv2.split(hsvFrame)
    hConst, sConst, vConst = cv2.split(hsvFrameConstrained)

    # Segmentierung: threshold(), inRange()
    # Hue 170 < hue < 180
    # Saturation: 120 < saturation
    hMaskRed = cv2.inRange(hConst, hueLowerThresholdRED, hueUpperThresholdRED)

    # cv2.imshow("Hue-Maske", hmask)

    ret1, smask = cv2.threshold(s, saturationThreshold, 255,
        cv2.THRESH_BINARY)
    #cv2.imshow("Sat-Maske", smask)

    # Rauschen rausfiltern
    cv2.medianBlur(hMaskRed, 1, hMaskRed)

    # Ausschnittarray wieder auf FullSize, damit bitwise_and geht
    helperArray = np.zeros((540, 960), np.uint8)
    helperArray[120:300, 120:270] = hMaskRed
    hMaskRed = helperArray

    # Verknüpfung der beiden Masken (Hue, Saturation)
    maskPlayer = cv2.bitwise_and(hMaskRed, smask)

    # Armmarkierungen finden über Suche nach Konturen in S/W-Bild
    maskPlayerContours = find_countours(hMaskRed)
    for i in maskPlayerContours:
        print(i[0])
        pt1 = (i[0], i[1])
        pt2 = (i[0]+i[2], i[1]+i[3])
        cv2.rectangle(maskPlayer, pt1, pt2, (255, 255, 255))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0))

    cv2.imshow("Maske Spieler", maskPlayer)

    # Schwerpunkt Spieler bestimmen
    # (cx, cy) = centerOfMass(mask)
    MP = cv2.moments(maskPlayer)
    if MP["m00"]:
        cx = int(MP["m10"] / MP["m00"])
        cy = int(MP["m01"] / MP["m00"])
        # Schwerpunkt zeichnen
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    # Video darstellen
    cv2.imshow("Video", frame)

    if cv2.waitKey(30) != -1:
        break

cap.release()
cv2.destroyAllWindows()




