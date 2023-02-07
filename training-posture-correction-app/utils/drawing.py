import cv2 as cv
import numpy as np
from utils import constant_list as cl


def drawing_landmark_points(cap_img, landmarks, train_mode, side) -> np.ndarray:
    """
    drawing detected points
    detected points are reversed on the left and right

    input:
        cap_img = video capture image
        landmarks = points detected by mediapipe
        train_mode = training mode
        detect_side = detected side

    output: video capture image
    """

    if train_mode is None:
        # upper body points
        points = len(cl.LANDMARK_INDEXES) // 2
    else:
        # all body points
        points = len(cl.LANDMARK_INDEXES)

    for index in range(points):

        point = cl.LANDMARK_INDEXES[index] % 2
        target = True if point == 0 else False

        if side is None:
            # drawing all points
            cv.circle(cap_img, landmarks[index], 7, cl.COLOR_BLUE, 3)
        elif side == cl.RIGHT_SIDE and target:
            # drawing left side points
            cv.circle(cap_img, landmarks[index], 7, cl.COLOR_BLUE, 3)
        elif side == cl.LEFT_SIDE and not target:
            # drawing left side points
            cv.circle(cap_img, landmarks[index], 7, cl.COLOR_BLUE, 3)

    return cap_img


def draw_lines_between_landmarks(
    cap_img, landmarks, train_mode, side, color
) -> np.ndarray:
    """
    drawing landmark connect lines
    detected points are reversed on the left and right

    input:
        cap_img = video capture image
        landmarks = points detected by mediapipe
        train_mode = training mode
        detect_side = detected side
        color = line color

    output: video capture image
    """

    if train_mode is None:
        if side == cl.RIGHT_SIDE:
            points = len(cl.RIGHT_SIDE_UPPER_CONNECT)
            connect_list = cl.RIGHT_SIDE_UPPER_CONNECT
        else:
            points = len(cl.LEFT_SIDE_UPPER_CONNECT)
            connect_list = cl.LEFT_SIDE_UPPER_CONNECT
    else:
        points = len(cl.FULL_BODY_CONNECT)
        connect_list = cl.FULL_BODY_CONNECT

    for index in range(points):

        point1 = landmarks[connect_list[index][0]]
        point2 = landmarks[connect_list[index][1]]

        cv.line(cap_img, point1, point2, color, 3, cv.LINE_4, 0)

    return cap_img


def drawing_text(
    cap_img, txt, point=cl.VIDEO_FRAME_CENTER, color=cl.COLOR_WHITE
) -> np.ndarray:
    """
    inserting text into a captured image

    input:
        cap_img = video capture image
        txt = put text
        point = points detected by mediapipe
        color = text color

    output: video capture image
    """

    cv.putText(cap_img, txt, point, cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)

    return cap_img
