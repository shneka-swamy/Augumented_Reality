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


def incremental_tracking(bgr_image_input, gray_image, image_pts):
    new_gray = cv2.cvtColor(src=bgr_image_input, code=cv2.COLOR_BGR2GRAY)
                
    new_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prevImg=gray_image, nextImg=new_gray,
            prevPts=image_pts, nextPts=None,
            winSize=(32,32), maxLevel=8,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

    gray_image = new_gray.copy()


    return new_pts


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

    set_incremental_tracking = False
    
    while True:
        is_ok, bgr_image_input = video_capture.read()
        if not is_ok:
            break  # no camera, or reached end of video file

        # Set this to true if at any point in the processing we can't find the marker.
        unable_to_find_marker = False


        if set_incremental_tracking == False:
            image_input_gray = cv2.cvtColor(src=bgr_image_input, code=cv2.COLOR_BGR2GRAY)
            # Detect keypoints and compute descriptors for input image.
            kp2, desc2 = detector.detectAndCompute(image=image_input_gray, mask=None)
            if len(kp2) < MIN_MATCHES_FOR_DETECTION:  # Higher threshold - fewer false detections
                unable_to_find_marker = True
        else:
            new_pts = incremental_tracking(bgr_image_input, gray_image, image_pts)

        # Match descriptors to marker image.
        if not unable_to_find_marker and set_incremental_tracking == False:
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
            if set_incremental_tracking == False:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            else:
                dst_pts = new_pts

            Hmat, mask = cv2.findHomography(
                srcPoints=src_pts, dstPoints=dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=5.0,  # default is 3.0
                maxIters=2000  # default is 2000
            )
            num_inliers = sum(mask)  # mask[i] is 1 if point i is an inlier, else 0
            if num_inliers < MIN_MATCHES_FOR_DETECTION:
                
                if set_incremental_tracking == True:
                    set_incremental_tracking = False

                unable_to_find_marker = True

        # Draw marker border on the image.
        if not unable_to_find_marker:

            if set_incremental_tracking == True:
                src_pts = np.array([src_pts[i,:] for i in range(len(mask)) if mask[i] == 1])
                image_pts = np.array([image_pts[i,:] for i in range(len(mask)) if mask[i] == 1])

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
                if set_incremental_tracking == False:
                    
                    set_incremental_tracking = True
                    inverse_Hmat = np.linalg.inv(Hmat)
                
                    gray_image = image_input_gray
                    image_pts = cv2.goodFeaturesToTrack(image=gray_image, maxCorners=100,
                            qualityLevel=0.01, minDistance=20, blockSize=11)
                    image_pts = np.squeeze(image_pts)       

                    src_pts = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), inverse_Hmat)

                else:
                    draw_prism(bgr_image_input, 1 , 0.75 , -2, rvec, tvec, K, dist_coeff)

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