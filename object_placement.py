import cv2
import sys
import time
import numpy as np

# This is for marker detection
CAMERA_NUMBER = 0  
MIN_MATCHES_FOR_DETECTION = 10  

# This part is used for pose detection
MARKER_HEIGHT = 8.125       
MARKER_WIDTH = 10.625

def draw_coordinate_frame(bgr_image, K, rvec, tvec):
    s = 1.0         # length of coordinate frame axes

    Paxes = np.array([  # Each column is a unit vector coordinate axis
        [0, s, 0, 0],
        [0, 0, s, 0],
        [0, 0, 0, s]
    ])

    # Pose of object wrt camera.
    R_o_c,_ = cv2.Rodrigues(src=rvec)
    to_c = tvec

    P_c = np.matmul(R_o_c, Paxes) + np.reshape(to_c, (3, 1))
    p = np.matmul(K, P_c)
    p = (p / p[2, :]).astype(int)
    cv2.line(img=bgr_image, pt1=(p[0, 0], p[1, 0]), pt2=(p[0, 1], p[1, 1]), color=(0, 0, 255), thickness=2)
    cv2.line(img=bgr_image, pt1=(p[0, 0], p[1, 0]), pt2=(p[0, 2], p[1, 2]), color=(0, 255, 0), thickness=2)
    cv2.line(img=bgr_image, pt1=(p[0, 0], p[1, 0]), pt2=(p[0, 3], p[1, 3]), color=(255, 0, 0), thickness=2)

# This function draws a prism
def draw_prism(img, length, width, height, rvec, tvec, K, dist_coeff):

    l = length
    w = width
    h = height

    line_type = 8

    P_prism = np.array([[-w, -l, 0], [-w, +l, 0], [+w, +l, 0], 
                        [+w, -l, 0], [0, 0, h] ])

    p, J = cv2.projectPoints(P_prism, rvec, tvec, K, dist_coeff)
    p = p.reshape(-1,2)
    p = p.astype(np.int32)

    edges = [(0,1), (1,2), (2,3), (3,0),
             (0,4), (1,4), (2,4), (3,4)]

    for i,j in edges:
        cv2.line(img, tuple(p[i]), tuple(p[j]), color=(0, 0, 255), thickness=2)

def find_object(desc1, kp1, detector, marker_image):
    # Initialize image capture from camera.
    video_capture = cv2.VideoCapture(CAMERA_NUMBER)  # Open video capture object
    is_ok, bgr_image_input = video_capture.read()  # Make sure we can read video
    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    # Define camera intrinsics (ideally should do a calibration prior to this).
    f = 0.8 * bgr_image_input.shape[1]  # Guess focal length in pixels
    cx = bgr_image_input.shape[1] / 2  # Assume principal point is in center of image
    cy = bgr_image_input.shape[0] / 2
    dist_coeff = np.zeros(4)  # Assume no lens distortion
    K = np.float64([[f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1.0]])

    while True:
        is_ok, bgr_image_input = video_capture.read()
        if not is_ok:
            break  # no camera, or reached end of video file

        # Set this to true if at any point in the processing we can't find the marker.
        unable_to_find_marker = False

        # Detect keypoints and compute descriptors for input image.
        image_input_gray = cv2.cvtColor(src=bgr_image_input, code=cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(image=image_input_gray, mask=None)
        if len(kp2) < MIN_MATCHES_FOR_DETECTION:  # Higher threshold - fewer false detections
            unable_to_find_marker = True

        # Match descriptors to marker image.
        if not unable_to_find_marker:
            matcher = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=False)
            matches = matcher.knnMatch(desc1, desc2, k=2)  # Find closest 2
            good_matches = []
            for m in matches:
                if m[0].distance < 0.8 * m[1].distance:  # Ratio test
                    good_matches.append(m[0])
            if len(good_matches) < MIN_MATCHES_FOR_DETECTION:  # Higher threshold - fewer false detections
                unable_to_find_marker = True

        # Fit homography.
        if not unable_to_find_marker:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            Hmat, mask = cv2.findHomography(
                srcPoints=src_pts, dstPoints=dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=5.0,  # default is 3.0
                maxIters=2000  # default is 2000
            )
            num_inliers = sum(mask)  # mask[i] is 1 if point i is an inlier, else 0
            if num_inliers < MIN_MATCHES_FOR_DETECTION:
                unable_to_find_marker = True

        # Draw marker border on the image.
        if not unable_to_find_marker:
            # Project the marker border lines to the image using the computed homography.
            h, w = marker_image.shape
            marker_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
            warped_corners = cv2.perspectiveTransform(marker_corners.reshape(-1, 1, 2), Hmat)

            cv2.polylines(img=bgr_image_input, pts=[np.int32(warped_corners)], isClosed=True,
                          color=[0, 255, 0], thickness=4, lineType=cv2.LINE_AA)

        # Find pose.
        if not unable_to_find_marker:
            marker_3d = np.float32([[-MARKER_WIDTH / 2, -MARKER_HEIGHT / 2, 0],
                                    [-MARKER_WIDTH / 2, MARKER_HEIGHT / 2, 0],
                                    [MARKER_WIDTH / 2, MARKER_HEIGHT / 2, 0],
                                    [MARKER_WIDTH / 2, -MARKER_HEIGHT / 2, 0]])
            pose_ok, rvec, tvec = cv2.solvePnP(marker_3d, warped_corners, K, dist_coeff)
            if pose_ok:
                draw_prism(bgr_image_input, 1 , 0.75 , -2, rvec, tvec, K, dist_coeff)
                #draw_coordinate_frame(bgr_image=bgr_image_input, K=K, rvec=rvec, tvec=tvec)

        # Show image and wait for xx msec (0 = wait till keypress).
        cv2.imshow("Input image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break  # Quit on ESC or q

    video_capture.release()
    cv2.destroyAllWindows()


# Detects the object and returns the key points and descriptors to main
def detect_object():
    # Read in an image, and convert it to grayscale.
    marker_image = cv2.imread("Collection_shell.jpg", cv2.IMREAD_GRAYSCALE)
    if marker_image is None:
        print("Can't read marker image from file")
        sys.exit()
    
    # Optionally reduce the size of the marker image.
    h = int(0.5 * marker_image.shape[0])
    w = int(0.5 * marker_image.shape[1])

    # Initialize feature detector.
    detector = cv2.ORB_create(nfeatures=2500,  # default = 500
                              edgeThreshold=16)  # default = 31

    # Detect keypoints in marker image and compute descriptors.
    kp, desc = detector.detectAndCompute(image=marker_image, mask=None)

    return [kp, desc, detector, marker_image]

    
def main():
    
    # Get image detection done.
    [kp, desc, detector, marker_image] = detect_object()
    find_object(desc, kp, detector, marker_image)

if __name__ == "__main__":
    main()
