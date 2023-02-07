import math
from typing import Tuple

from utils import constant_list as cl


def using_landmark_lists(landmarks) -> list:
    """
    Convert the landmarks to be used into
    video capture coordinates and store them

    input: points detected by mediapipe
    output: using landmark list
    """

    using_landmark = []

    for index in cl.LANDMARK_INDEXES:
        landmark = calculate_landmark2video_coords(landmarks[index])
        using_landmark.append(landmark)

    center_l_shoulder = calculate_center_landmark(using_landmark[2])
    center_r_shoulder = calculate_center_landmark(using_landmark[3])
    center_shoulder_x = center_l_shoulder[cl.X] + center_r_shoulder[cl.X]
    center_shoulder_y = center_l_shoulder[cl.Y] + center_r_shoulder[cl.Y]

    center_l_waist = calculate_center_landmark(using_landmark[8])
    center_r_waist = calculate_center_landmark(using_landmark[9])
    center_waist_x = center_l_waist[cl.X] + center_r_waist[cl.X]
    center_waist_y = center_l_waist[cl.Y] + center_r_waist[cl.Y]

    using_landmark.append((center_shoulder_x, center_shoulder_y))
    using_landmark.append((center_waist_x, center_waist_y))

    return using_landmark


def calculate_cosine(point1, point2, point3) -> float:
    """
    calculate neck_angle
    input:
        point1:target angle
        point2:edge side
        point3:edge side

    output: calculated angle
    """

    d1_x = point1[cl.X] - point3[cl.X]
    d1_y = point1[cl.Y] - point3[cl.Y]
    d2_x = point1[cl.X] - point2[cl.X]
    d2_y = point1[cl.Y] - point2[cl.Y]
    d3_x = point3[cl.X] - point2[cl.X]
    d3_y = point3[cl.Y] - point2[cl.Y]

    d1 = math.sqrt(d1_x**2 + d1_y**2)
    d2 = math.sqrt(d2_x**2 + d2_y**2)
    d3 = math.sqrt(d3_x**2 + d3_y**2)

    cos_theta = (d1**2 + d2**2 - d3**2) / (2 * d1 * d2)
    angle = math.degrees(math.acos(cos_theta))

    return angle


def calculate_center_landmark(landmark) -> Tuple[int, int]:
    """
    landmark to half coords

    input: mediapipe landmark lists

    output: transform coords to half video frame
    """

    transform_landmark_x = int(landmark[cl.X] / 2)
    transform_landmark_y = int(landmark[cl.Y] / 2)

    return transform_landmark_x, transform_landmark_y


def calculate_landmark2video_coords(landmark) -> Tuple[int, int]:
    """
    landmark to video capture coords

    input: mediapipe landmark lists

    output: transform coords to video frame
    """

    transform_landmark_x = int(landmark.x * cl.VIDEO_FRAME_WIDTH)
    transform_landmark_y = int(landmark.y * cl.VIDEO_FRAME_HEIGHT)

    return transform_landmark_x, transform_landmark_y


def calculate_neck_angle(landmark, side) -> float:
    """
    landmark list for calculating neck angle

    input:
        landmark = mediapipe landmark lists
        side = detection side

    output: calculated angle
    """

    if side == cl.RIGHT_SIDE:
        ear = landmark[0]
        shoulder = landmark[2]
        ex_point = (shoulder[cl.X], shoulder[cl.Y] - cl.EXTENTOIN_CIE)
    else:
        ear = landmark[1]
        shoulder = landmark[3]
        ex_point = (shoulder[cl.X], shoulder[cl.Y] - cl.EXTENTOIN_CIE)

    return calculate_cosine(shoulder, ex_point, ear)


def calculate_knee_angle(landmark) -> float:
    """
    landmark list for calculating knee angle

    input:
        landmark = mediapipe landmark lists

    output: calculated angle
    """

    right_knee = landmark[10]
    right_waist = landmark[8]
    right_ankle = landmark[12]
    left_knee = landmark[11]
    left_waist = landmark[9]
    left_ankle = landmark[13]

    sum_right_knee = sum(right_knee)
    sum_right_waist = sum(right_waist)
    sum_right_ankle = sum(right_ankle)
    sum_left_knee = sum(left_knee)
    sum_left_waist = sum(left_waist)
    sum_left_ankle = sum(left_ankle)

    right = sum([sum_right_knee, sum_right_waist, sum_right_ankle])
    left = sum([sum_left_knee, sum_left_waist, sum_left_ankle])

    if right > left:
        angle = calculate_cosine(right_knee, right_ankle, right_waist)
    else:
        angle = calculate_cosine(left_knee, left_ankle, left_waist)

    return angle


def calculate_elbow_angle(landmark) -> float:
    """
    landmark list for calculating elbow angle

    input:
        landmark = mediapipe landmark lists

    output: calculated angle
    """

    right_shoulder = landmark[2]
    right_elbow = landmark[4]
    right_wrist = landmark[6]
    left_shoulder = landmark[3]
    left_elbow = landmark[5]
    left_wrist = landmark[7]

    sum_right_shoulder = sum(right_shoulder)
    sum_right_elbow = sum(right_elbow)
    sum_right_wrist = sum(right_wrist)
    sum_left_shoulder = sum(left_shoulder)
    sum_left_elbow = sum(left_elbow)
    sum_left_wrist = sum(left_wrist)

    right = sum([sum_right_shoulder, sum_right_elbow, sum_right_wrist])
    left = sum([sum_left_shoulder, sum_left_elbow, sum_left_wrist])

    if right > left:
        angle = calculate_cosine(right_elbow, right_shoulder, right_wrist)
    else:
        angle = calculate_cosine(left_elbow, left_shoulder, left_wrist)

    return angle


def calculate_armpits_angle(landmark) -> float:
    """
    landmark list for calculating neck angle

    input:
        landmark = mediapipe landmark lists

    output:calculated angle
    """
    right_shoulder = landmark[2]
    right_elbow = landmark[4]
    right_waist = landmark[8]
    left_shoulder = landmark[3]
    left_elbow = landmark[5]
    left_waist = landmark[9]

    sum_right_shoulder = sum(right_shoulder)
    sum_right_elbow = sum(right_elbow)
    sum_right_waist = sum(right_waist)
    sum_left_shoulder = sum(left_shoulder)
    sum_left_elbow = sum(left_elbow)
    sum_left_waist = sum(left_waist)

    right = sum([sum_right_shoulder, sum_right_elbow, sum_right_waist])
    left = sum([sum_left_shoulder, sum_left_elbow, sum_left_waist])

    if right > left:
        angle = calculate_cosine(right_shoulder, right_elbow, right_waist)
    else:
        angle = calculate_cosine(left_shoulder, left_elbow, left_waist)

    return angle
