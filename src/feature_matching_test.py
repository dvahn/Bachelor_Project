import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Videos/video_long.mp4')

MIN_MATCH_COUNT = 10

img1 = cv2.imread('Bilder/basketball.jpg', 0)

# initiating SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()

# keypoints finden für img1


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)


while cap.isOpened():
    ret, res = cap.read()
    if ret:
        # resizing the Video frame by frame
        frame = cv2.resize(res, (960, 540))
    else:
        break

    kp1, des1 = sift.detectAndCompute(img1, None)
    img2 = frame
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    img1 = cv2.resize(img1, (240, 200))

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # plt.imshow(img3, 'gray'), plt.show()

    cv2.imshow("Video", img1)
    cv2.imshow("Video", frame)
    cv2.imshow("Video", img3)

    if cv2.waitKey(30) != -1:
        break

cap.release()
cv2.destroyAllWindows()