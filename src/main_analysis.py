import numpy as np
import cv2

cap = cv2.VideoCapture('Videos/video_long.mp4')
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
framesPerSecond = cap.get(cv2.CAP_PROP_FPS)
#print(frameWidth, frameHeight, framesPerSecond)

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


    # Mouse-Callback zur Farbfindung
    def mouseCallback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            colorBGR = frame[y, x, :]
            colorRGB = tuple(reversed(colorBGR))
            colorHSV = cv2.cvtColor(np.uint8([[colorBGR]]), cv2.COLOR_BGR2HSV)
            #print("HSV value at ({},{}):{}".format(x, y, colorHSV))

    cv2.setMouseCallback("Video", mouseCallback)

    # Umwandlung in HSV
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Eingeschränkter Suchbereich für rot
    constrainedFrame = frame[120:300, 120:270]
    hsvFrameConstrained = cv2.cvtColor(constrainedFrame, cv2.COLOR_BGR2HSV)

    # Aufspaltung in 3 Graustufenbilder für H, S, V mit:
    h, s, v = cv2.split(hsvFrame)
    hConst, sConst, vConst = cv2.split(hsvFrameConstrained)

    # print(h,s,v)

    # Segmentierung: threshold(), inRange()
    # Hue 170 < hue < 180
    # Saturation: 120 < saturation
    hMaskRed = cv2.inRange(hConst, hueLowerThresholdRED, hueUpperThresholdRED)
    hMaskGreen = cv2.inRange(h, hueLowerThresholdGREEN, hueUpperThresholdGREEN)

    # cv2.imshow("Hue-Maske", hmask)

    ret1, smask = cv2.threshold(s, saturationThreshold, 255,
        cv2.THRESH_BINARY)
    #cv2.imshow("Sat-Maske", smask)

    # Ausschnittarray wieder auf FullSize, damit bitwise_and geht
    helperArray = np.zeros((540, 960), np.uint8)
    helperArray[120:300, 120:270] = hMaskRed
    hMaskRed = helperArray

    # Verknüpfung der beiden Masken (Hue, Saturation)
    maskPlayer = cv2.bitwise_and(hMaskRed, smask)
    cv2.medianBlur(maskPlayer, 1 ,maskPlayer)
    cv2.imshow("Maske Spieler", maskPlayer)

    maskNet = cv2.bitwise_and(hMaskGreen, smask)
    cv2.imshow("Maske Netz", maskNet)

    # Schwerpunkt Spieler bestimmen
    # (cx, cy) = centerOfMass(mask)
    MP = cv2.moments(maskPlayer)
    if MP["m00"]:
        cx = int(MP["m10"] / MP["m00"])
        cy = int(MP["m01"] / MP["m00"])
        # Schwerpunkt zeichnen
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    # Schwerpunkt Netz bestimmen
    MN = cv2.moments(maskNet)
    if MN["m00"]:
        cx = int(MN["m10"] / MN["m00"])
        cy = int(MN["m01"] / MN["m00"])
        # Schwerpunkt zeichnen
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)



    # Verbessern: Maske filtern (MedianFilter = Störung wegfiltern), Regionengröße (Fläche bestimmen), Bereich einschränken
    # man muss Kompromisse eingehen


    # Video darstellen
    cv2.imshow("Video", frame)

    if cv2.waitKey(30) != -1:
        break

cap.release()
cv2.destroyAllWindows()



def registerHits(cx, cy):
    # if cx and cy move:
    donothing = 0
    # count score one up
    # register hit

def calculateAngles(com1, com2, com3):
    # draw lines between centers of mass'
    # calculate angle
    angle = 0
    return angle

def registerThrow(angle):
    # if angle expands fast, it might be a throw
    donothing = 0
    # zeitabhängig
    # typische Winkel als Referenz


