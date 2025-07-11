# feature_extraction.py
import cv2
import numpy as np
import math

def extract_centerline(frame):
    """
    extracts the center of mass (centerline) of the largest contour in a binary image.
    used to estimate the jet's central axis in a frame.

    parameters:
        frame (np.ndarray): input image (BGR)

    returns:
        tuple: (cx, cy) coordinates of centerline, or (np.nan, np.nan) if not found
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return np.nan, np.nan

def extract_geometry_features(frame):
    """
    extracts geometric features from the largest contour in a frame.
    features include: jet angle (from ellipse fit), width (bounding box), symmetry (left-right mass), and area.

    parameters:
        frame (np.ndarray): input image (BGR)

    returns:
        list: [angle, width, symmetry, area], np.nan for any missing values
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # use Otsu's thresholding for adaptive segmentation
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [np.nan] * 4

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)

    # only fit ellipse if contour is large enough
    angle = np.nan
    if len(c) >= 5 and area > 100:  # adjust area threshold as needed
        try:
            ellipse = cv2.fitEllipse(c)
            angle = ellipse[2]
        except:
            pass

    # Cone width = width at base of bounding box
    # Jet symmetry = horizontal symmetry: diff between left and right mass
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    left_mass = float(np.sum(mask[:, :mask.shape[1] // 2]))
    right_mass = float(np.sum(mask[:, mask.shape[1] // 2:]))
    symmetry = abs(left_mass - right_mass) / (left_mass + right_mass + 1e-5)


    return [angle, w, symmetry, area]

def extract_features(frame_side, frame_top):
    """
    extracts all features from both the side and top camera frames.
    combines centerline and geometry features for both views into a single feature vector.

    parameters:
        frame_side (np.ndarray): side view frame (BGR)
        frame_top (np.ndarray): top view frame (BGR)

    returns:
        list: [csx, csy, ctx, cty, side_angle, side_width, side_symmetry, side_area, top_angle, top_width, top_symmetry, top_area]
    """
    csx, csy = extract_centerline(frame_side)
    ctx, cty = extract_centerline(frame_top)
    side_features = extract_geometry_features(frame_side)
    top_features = extract_geometry_features(frame_top)
    return [csx, csy, ctx, cty] + side_features + top_features

def process_video(video_path, resize=(128, 128)):
    """
    processes the input video file, extracting features from each frame.
    splits each frame into side and top views, resizes, and extracts features.

    parameters:
        video_path (str): path to the video file
        resize (tuple): size to resize each view (default (128, 128))

    returns:
        np.ndarray: array of shape (frames, num_features) with all extracted features
    """
    
    cap = cv2.VideoCapture(video_path)
    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        mid = w // 2
        frame_side = cv2.resize(frame[:, :mid], resize)
        frame_top = cv2.resize(frame[:, mid:], resize)
        f = extract_features(frame_side, frame_top)
        features.append(f)
    cap.release()
    return np.array(features)  # shape: (frames, num_features)
