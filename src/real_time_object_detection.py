from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments if needed
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False,
				help="Path to 'deploy' prototxt file.")
ap.add_argument("-m", "--model", required=False,
				help="Path to pre-trained AI model.")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
				help="Minimum probability to filter weak detections.")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		   "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 200, size=(len(CLASSES), 3))

# load the serialized model from disk
print("[INFO] Loading model...")

if args["prototxt"] is not None and args["model"] is not None:
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

else:
	# run default config
	net = cv2.dnn.readNetFromCaffe("../realtime_detection/MobileNetSSD_deploy.prototxt.txt",
								   "../realtime_detection/MobileNetSSD_deploy.caffemodel")

# initialize the video stream, allow the stream to warmup,
# and initialize the FPS counter
print("[INFO] Loading video...")

lastKnownPositionStart = None
lastKnownPositionEnd = None
lastKnownIdx = None
label = None
searchFrame = None
hConst = 0
sConst = 0
vConst = 0

hueLowerThresholdRED = 174
hueUpperThresholdRED = 177
saturationThreshold = 120


def find_countours(roi):
	im2, cnts, hierarchy = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]  # get largest three contour areas
	rects = []
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		x, y, w, h = cv2.boundingRect(approx)
		if h >= 5:
			# if height is enough
			# create rectangle for bounding
			rect = (x, y, x + w, y + h)
			rects.append(rect)

	return rects


def calculateCenterOfRectangle(listOfCorners):
	# calculating the center of every rectangle to get lines between centers to calculate angles
	cx = 0
	cy = 0
	if len(listOfCorners) >= 4:
		cx = int((listOfCorners[0] + listOfCorners[2]) / 2)
		cy = int((listOfCorners[1] + listOfCorners[3]) / 2)

	return cx, cy

# bubble sort for detected markers, to get the lines drawn correctly
def bubbleSort(arr):
	n = len(arr)

	if n < 3:
		return arr

	for i in range(n):

		for j in range(0, n - i - 1):

			if arr[j][0] > arr[j + 1][0]:
				arr[j], arr[j + 1] = arr[j + 1], arr[j]

	if calculateCenterOfRectangle(arr[1])[1] < calculateCenterOfRectangle(arr[2])[1]:
		# check if "hand" is between "shoulder" and "elbow"
		arr[1], arr[2] = arr[2], arr[1]


cap = cv2.VideoCapture('../Videos/video_long.mp4')
# vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while cap.isOpened():
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, res = cap.read()
	if ret:
		# resizing the Video frame by frame
		frame = imutils.resize(res, 960, 540)
	else:
		break

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
								 0.007843, (300, 300), 127.5)
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
										 confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
						  COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			lastKnownPositionStart = (startX - 10, startY - 50)
			lastKnownPositionEnd = (endX + 10, endY + 10)
			lastKnownIdx = idx

		else:
			# print rectangle if detection is lost
			cv2.rectangle(frame, lastKnownPositionStart, lastKnownPositionEnd,
						  COLORS[lastKnownIdx], 2)

			y = lastKnownPositionStart[1] - 15 if lastKnownPositionStart[1] - 15 > 15 else lastKnownPositionStart[
																							   1] + 15
			cv2.putText(frame, label, (lastKnownPositionStart[0], y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[lastKnownIdx], 2)

	lowerSearchBorder = int((lastKnownPositionEnd[1] - lastKnownPositionStart[1]) / 2)
	searchFrame = frame[lastKnownPositionStart[1]:lastKnownPositionEnd[1] - lowerSearchBorder,
				  lastKnownPositionStart[0]:lastKnownPositionEnd[0]]

	hsvFrameConstrained = cv2.cvtColor(searchFrame, cv2.COLOR_BGR2HSV)
	if hsvFrameConstrained is not np.zeros((540, 960), np.uint8):
		hConst, sConst, vConst = cv2.split(hsvFrameConstrained)

	_, smask = cv2.threshold(sConst, saturationThreshold, 255,
							 cv2.THRESH_BINARY)

	hMaskRed = cv2.inRange(hConst, hueLowerThresholdRED, hueUpperThresholdRED)
	cv2.medianBlur(hMaskRed, 1, hMaskRed)

	# Verknüpfung der beiden Masken (Hue, Saturation)
	maskPlayer = cv2.bitwise_and(hMaskRed, smask)

	# Ausschnittarray wieder auf FullSize, damit bitwise_and geht
	helperArray = np.zeros((540, 960), np.uint8)
	helperArray[lastKnownPositionStart[1]:lastKnownPositionEnd[1] - lowerSearchBorder,
	lastKnownPositionStart[0]:lastKnownPositionEnd[0]] = maskPlayer
	maskPlayer = helperArray

	# Morphological Filters
	kernel = np.ones((3, 3), np.uint8)
	cv2.medianBlur(maskPlayer, 3, maskPlayer)
	maskPlayer = cv2.dilate(maskPlayer, kernel, iterations=1)
	maskPlayer = cv2.morphologyEx(maskPlayer, cv2.MORPH_OPEN, kernel)

	# Armmarkierungen finden über Suche nach Konturen in S/W-Bild
	maskPlayerContours = find_countours(maskPlayer)
	print("Unsorted:", maskPlayerContours)
	bubbleSort(maskPlayerContours)
	print("Sorted:", maskPlayerContours)
	for i in maskPlayerContours:
		center = calculateCenterOfRectangle(i)
		cv2.circle(frame, center, 7, (0, 255, 0), 2)

	# draw lines between rectangles
	# stick lines to rectangles?
	if len(maskPlayerContours) >= 3:
		cv2.line(frame, calculateCenterOfRectangle(maskPlayerContours[0]),
				calculateCenterOfRectangle(maskPlayerContours[1]), (255, 255, 255))
		cv2.line(frame, calculateCenterOfRectangle(maskPlayerContours[1]),
				calculateCenterOfRectangle(maskPlayerContours[2]), (255, 255, 255))

	# show the output frame
	cv2.imshow("Maske Spieler", maskPlayer)
	cv2.imshow("Frame", frame)
	cv2.imshow("Searchframe", searchFrame)

	if cv2.waitKey(30) != -1:
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cap.release()
cv2.destroyAllWindows()
