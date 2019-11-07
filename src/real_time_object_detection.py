from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import math
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

'''constants and parameters'''
last_known_position_start = None
last_known_position_end = None
last_known_idx = None
label = None
search_frame = None
h_const = 0
s_const = 0
v_const = 0

hue_lower_threshold_red = 174
hue_upper_threshold_red = 177
hue_lower_threshold_green = 69
hue_upper_threshold_green = 80
saturation_threshold = 120

last_frame_area = 1000

hit_count = 0
counter = 0

min_angle = 360
max_angle = 0
current_min_angle = 360
current_max_angle = 0

attempts = 0

motion_line = []

ready = True
ready_label = "Throwing!"

def find_contours(roi, count):
	im2, cnts, hierarchy = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:count]  # get largest *count* contour areas
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

def calculate_center_of_rectangle(list_of_corners):
	# calculating the center of every rectangle to get lines between centers to calculate angles
	cx = 0
	cy = 0
	if len(list_of_corners) >= 4:
		cx = int((list_of_corners[0] + list_of_corners[2]) / 2)
		cy = int((list_of_corners[1] + list_of_corners[3]) / 2)

	return cx, cy

def calculate_area_of_rectangle(list_of_corners):

	area = (list_of_corners[2]-list_of_corners[0]) * (list_of_corners[3]-list_of_corners[1])

	return area

# bubble sort for detected markers, to get the lines drawn correctly
# Order is: shoulder, elbow, hand
def bubble_sort(arr):
	n = len(arr)

	if n < 3:
		return arr

	for i in range(n):

		for j in range(0, n - i - 1):

			if arr[j][0] > arr[j + 1][0]:
				arr[j], arr[j + 1] = arr[j + 1], arr[j]

	# check if "hand" is between "shoulder" and "elbow"
	if calculate_center_of_rectangle(arr[1])[1] < calculate_center_of_rectangle(arr[2])[1] and calculate_center_of_rectangle(arr[0])[0] < calculate_center_of_rectangle(arr[1])[0] < calculate_center_of_rectangle(arr[2])[0]:
		arr[1], arr[2] = arr[2], arr[1]

def get_angle(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	dY = y2 - y1
	dX = x2 - x1
	rads = math.atan2(-dY, dX)
	return math.degrees(rads)

def track_scoring(fr):
	global last_frame_area
	global hit_count
	hsv_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
	hue, sat, val = cv2.split(hsv_frame)
	h_mask_green = cv2.inRange(hue, hue_lower_threshold_green, hue_upper_threshold_green)
	ret1, sat_mask = cv2.threshold(sat, saturation_threshold, 255,
								   cv2.THRESH_BINARY)
	mask_net = cv2.bitwise_and(h_mask_green, sat_mask)
	cv2.medianBlur(mask_net, 3, mask_net)
	net_contours = find_contours(mask_net, 1)

	flat_net_contours = []
	if len(net_contours) > 0:
		for j in range(1):
			for number in net_contours[j]:
				flat_net_contours.append(number)

		current_area = calculate_area_of_rectangle(flat_net_contours)

		if current_area < last_frame_area * (7/10):
			print("Hit")
			hit_count += 1

		last_frame_area = current_area

def check_ready(marker):
	global ready
	global counter
	if len(marker) == 3:
		counter += 1
	if len(marker) < 3:
		ready = False
		counter = 0
	if counter >= 8:
		ready = True

def calculate_current_min_max_angles(ang):
	global min_angle
	global max_angle

	if max_angle < ang < 200:
		max_angle = ang
	if min_angle > ang >= 0:
		min_angle = ang

	return min_angle, max_angle

def reset_angles():
	global min_angle
	global max_angle
	min_angle = 360
	max_angle = 0

scaleX = 960
scaleY = 540
cap = cv2.VideoCapture('../Videos/video_3.mp4')
# vs = VideoStream(src=0).start()
time.sleep(1.0)
fps = FPS().start()

# loop over the frames from the video stream
while cap.isOpened():
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, res = cap.read()
	if ret:
		# resizing the Video frame by frame
		frame = imutils.resize(res, scaleX, scaleY)
	else:
		break

	'''Player detection with neural network'''
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
								0.007843, (300, 300), 127.5)
	# pass the blob through the network and obtain the detections and predictions
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
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
						COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			'''Define area in which markers are tracked'''
			last_known_position_start = (startX - 10, startY - 50)
			last_known_position_end = (endX + 10, endY + 10)
			last_known_idx = idx

		else:
			# print rectangle if detection is lost
			cv2.rectangle(frame, last_known_position_start, last_known_position_end, COLORS[last_known_idx], 2)

			y = last_known_position_start[1] - 15 if last_known_position_start[1] - 15 > 15 else last_known_position_start[1] + 15
			cv2.putText(frame, label, (last_known_position_start[0], y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[last_known_idx], 2)

	'''Tracking of markers'''
	lowerSearchBorder = int((last_known_position_end[1] - last_known_position_start[1]) / 2)
	search_frame = frame[last_known_position_start[1]:last_known_position_end[1] - lowerSearchBorder,
				  last_known_position_start[0]:last_known_position_end[0]]

	hsvFrameConstrained = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
	if hsvFrameConstrained is not np.zeros((scaleY, scaleX), np.uint8):
		h_const, s_const, v_const = cv2.split(hsvFrameConstrained)

	_, s_mask = cv2.threshold(s_const, saturation_threshold, 255,
							cv2.THRESH_BINARY)

	hMaskRed = cv2.inRange(h_const, hue_lower_threshold_red, hue_upper_threshold_red)
	cv2.medianBlur(hMaskRed, 1, hMaskRed)

	# Verknüpfung der beiden Masken (Hue, Saturation)
	mask_player = cv2.bitwise_and(hMaskRed, s_mask)

	# Ausschnittarray wieder auf FullSize, damit bitwise_and geht
	helper_array = np.zeros((scaleY, scaleX), np.uint8)
	helper_array[last_known_position_start[1]:last_known_position_end[1] - lowerSearchBorder,
	last_known_position_start[0]:last_known_position_end[0]] = mask_player
	mask_player = helper_array

	# Morphological Filters
	kernel = np.ones((3, 3), np.uint8)
	cv2.medianBlur(mask_player, 3, mask_player)
	mask_player = cv2.dilate(mask_player, kernel, iterations=1)
	mask_player = cv2.morphologyEx(mask_player, cv2.MORPH_OPEN, kernel)

	# Armmarkierungen finden über Suche nach Konturen in S/W-Bild
	mask_player_contours = find_contours(mask_player, 3)

	'''Sorting the markers'''
	bubble_sort(mask_player_contours)

	for i in mask_player_contours:
		center = calculate_center_of_rectangle(i)
		cv2.circle(frame, center, 7, (0, 255, 0), 2)


	'''Checking if throwing motion is in progress'''
	check_ready(mask_player_contours)
	if ready:
		cv2.putText(frame, ready_label, (40, scaleY - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
		# draw lines between rectangles
		if len(mask_player_contours) >= 3:
			# shoulder to elbow
			cv2.line(frame, calculate_center_of_rectangle(mask_player_contours[0]),
					calculate_center_of_rectangle(mask_player_contours[1]), (255, 255, 255))
			# elbow to hand
			cv2.line(frame, calculate_center_of_rectangle(mask_player_contours[1]),
					calculate_center_of_rectangle(mask_player_contours[2]), (255, 255, 255))

			'''drawing the motion as a line'''
			motion_line.append(calculate_center_of_rectangle(mask_player_contours[2]))
			for p in motion_line:
				cv2.circle(frame, p, 1, (0, 0, 255), 2)

			'''Calculate angle and put a label with it on screen'''
			a = calculate_center_of_rectangle(mask_player_contours[0])
			b = calculate_center_of_rectangle(mask_player_contours[1])
			c = calculate_center_of_rectangle(mask_player_contours[2])

			angle = int((180-(get_angle(b, c)-get_angle(a, b))))
			label_angle = '{}: {} degree'.format('Angle', angle)
			cv2.putText(frame, label_angle, (40, 40),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

			current_min_angle, current_max_angle = calculate_current_min_max_angles(angle)

	else:
		motion_line = []
		reset_angles()

	print(current_min_angle, current_max_angle)

	'''show the output'''
	track_scoring(frame)
	score_label = "{}: {}".format("Score", hit_count)
	cv2.putText(frame, score_label, (scaleX - 170, scaleY - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	current_min_angle_label = "{}: {}".format("Angle MIN", current_min_angle)
	current_max_angle_label = "{}: {}".format("Angle MAX", current_max_angle)
	cv2.putText(frame, current_min_angle_label, (scaleX-300, scaleY - 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	cv2.putText(frame, current_max_angle_label, (scaleX-300, scaleY - 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	cv2.imshow("Frame", frame)

	# update the FPS counter
	fps.update()

	if cv2.waitKey(30) != -1:
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cap.release()
cv2.destroyAllWindows()
