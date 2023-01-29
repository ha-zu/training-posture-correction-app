import math
import os
import uuid
from collections import namedtuple
from typing import Any, Tuple

import cv2 as cv
import mediapipe as mp
import numpy as np
from constant import constant_list as cl


def main_training(train_mode: str = cl.DESK_WORK):
    """
    the right side of body should face the camera and 1m away.
    """

    # load pose model
    mp_pose = mp.solutions.pose

    # setting capture
    cap = cv.VideoCapture(cl.VIDEO_CAMERA_MAC)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cl.VIDEO_FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cl.VIDEO_FRAME_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, 15)
    writer = recording_format()

    # setting holistic
    with mp_pose.Pose(
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("【Error】can not opend video capture.")
                break

            # convert color for use mediapipe
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            # Inverted with respect to Y axis
            # Detected inverted right and left
            image = cv.flip(image, 1)
            image.flags.writeable = False
            # get detections
            results = pose.process(image)
            image.flags.writeable = True

            # convert color for use opencv
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # get landmarks
            landmarks = results.pose_landmarks
            # get landmarks index
            landmarks_idx = mp_pose.PoseLandmark

            # masking captured image(demo mode)
            condition = np.stack((results.segmentation_mask,) * 3, axis=2) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (229, 229, 229)
            image = np.where(condition, image, bg_image)

            # drawing base posture points
            if landmarks is not None:
                image = drawing_base_landmarks(image, landmarks, landmarks_idx)

                # train mode statement
                match train_mode:
                    # drawing landmarks and checking posture
                    case cl.DESK_WORK:
                        check_angle = check_straight_neck(landmarks, landmarks_idx)
                        # instruct voice for neck angle over 30 degree
                        if cl.STRAIGHT_NECK_ANGLE < check_angle:
                            cv.putText(
                                image,
                                str(check_angle),
                                (10, 30),
                                cv.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2,
                                cv.LINE_AA,
                            )
                    case cl.SQUAT:
                        image = drawing_squad_landmarks(image, landmarks, landmarks_idx)
                    case cl.PLANK:
                        image = drawing_plank_landmarks(image, landmarks, landmarks_idx)
                    case cl.PUSH_UP:
                        image = drawing_push_up_landmarks(
                            image, landmarks, landmarks_idx
                        )

            cv.imshow("video capture image", image)
            writer.write(image)

            # key esc pressed stop capture
            if cv.waitKey(1) == 27:
                break

    writer.release()
    cap.release()
    cv.destroyAllWindows()


def cie_landmark2videosize(landmark_x, landmark_y) -> Tuple[int, int]:
    """landmark coordinates to video coordinates"""

    x = int(landmark_x * cl.VIDEO_FRAME_WIDTH)
    y = int(landmark_y * cl.VIDEO_FRAME_HEIGHT)

    return x, y


def recording_format() -> Any:
    """recording video format"""

    fmt = cv.VideoWriter_fourcc("m", "p", "4", "v")
    video_out = os.path.join(
        os.getcwd(), cl.VIDEO_OUT_DIR, "{}.mp4".format(uuid.uuid1())
    )
    writer = cv.VideoWriter(
        video_out,
        fmt,
        cl.VIDEO_FRAME_RATE10,
        (cl.VIDEO_FRAME_WIDTH, cl.VIDEO_FRAME_HEIGHT),
    )

    return writer


def drawing_base_landmarks(image, landmarks, land_idx) -> np.ndarray:
    """drawing a base landmark"""

    # center points coordinates
    cie = namedtuple("Coordinates", ["x", "y"])
    landmark = landmarks.landmark

    # drawing ear points
    r_ear_x, r_ear_y = cie_landmark2videosize(
        landmark[land_idx.RIGHT_EAR].x, landmark[land_idx.RIGHT_EAR].y
    )
    l_ear_x, l_ear_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_EAR].x, landmark[land_idx.LEFT_EAR].y
    )
    # cv.circle(image, (r_ear_x, r_ear_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_ear_x, l_ear_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing sholder, neck(c_sholder) points
    r_sholder_x, r_sholder_y = cie_landmark2videosize(
        landmark[land_idx.RIGHT_SHOULDER].x, landmark[land_idx.RIGHT_SHOULDER].y
    )
    l_sholder_x, l_sholder_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_SHOULDER].x, landmark[land_idx.LEFT_SHOULDER].y
    )
    # c_sholder = cie(int((r_sholder_x + l_sholder_x) / 2), int((r_sholder_y + l_sholder_y) / 2))
    ex_l_sholder = cie(l_sholder_x, l_sholder_y - cl.EXTENTOIN_CIE)
    # cv.circle(image, (r_sholder_x, r_sholder_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image,
        (l_sholder_x, l_sholder_y),
        cl.RADIUS_SIZE7,
        cl.COLOR_BLUE,
        cl.THICKNES_SIZE3,
    )
    # cv.circle(image, (c_sholder.x, c_sholder.y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image,
        (ex_l_sholder.x, ex_l_sholder.y),
        cl.RADIUS_SIZE7,
        cl.COLOR_BLUE,
        cl.THICKNES_SIZE3,
    )

    # drawing waist points
    # r_waist_x, r_waist_y = cie_landmark2videosize(landmark[land_idx.RIGHT_HIP].x, landmark[land_idx.RIGHT_HIP].y)
    l_waist_x, l_waist_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_HIP].x, landmark[land_idx.LEFT_HIP].y
    )
    # c_waist = cie(int((r_waist_x + l_waist_x) / 2), int((r_waist_y + l_waist_y) / 2))
    # cv.circle(image, (r_waist_x, r_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_waist_x, l_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )
    # cv.circle(image, (c_waist.x, c_waist.y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)

    # drawing line connections
    # cv.line(image, (r_ear_x, r_ear_y), (r_sholder_x, r_sholder_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_ear_x, l_ear_y),
        (l_sholder_x, l_sholder_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_sholder_x, r_sholder_y), (c_sholder.x, c_sholder.y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    # cv.line(image, (l_sholder_x, l_sholder_y), (c_sholder.x, c_sholder.y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_sholder_x, l_sholder_y),
        (ex_l_sholder.x, ex_l_sholder.y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_sholder_x, r_sholder_y), (r_waist_x, r_waist_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_sholder_x, l_sholder_y),
        (l_waist_x, l_waist_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_waist_x, r_waist_y), (c_waist.x, c_waist.y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    # cv.line(image, (l_waist_x, l_waist_y), (c_waist.x, c_waist.y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)

    return image


def drawing_squad_landmarks(image, landmarks, land_idx) -> np.ndarray:
    """drawing a squat training landmark"""

    landmark = landmarks.landmark

    # drawing waist points
    # r_waist_x, r_waist_y = cie_landmark2videosize(landmark[land_idx.RIGHT_HIP].x, landmark[land_idx.RIGHT_HIP].y)
    l_waist_x, l_waist_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_HIP].x, landmark[land_idx.LEFT_HIP].y
    )
    # cv.circle(image, (r_waist_x, r_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_waist_x, l_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing knee points
    # r_knee_x, r_knee_y = cie_landmark2videosize(landmark[land_idx.RIGHT_KNEE].x, landmark[land_idx.RIGHT_KNEE].y)
    l_knee_x, l_knee_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_KNEE].x, landmark[land_idx.LEFT_KNEE].y
    )
    # cv.circle(image, (r_knee_x, r_knee_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_knee_x, l_knee_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing ankle points
    # r_ankle_x, r_ankle_y = cie_landmark2videosize(landmark[land_idx.RIGHT_ANKLE].x, landmark[land_idx.RIGHT_ANKLE].y)
    l_ankle_x, l_ankle_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_ANKLE].x, landmark[land_idx.LEFT_ANKLE].y
    )
    # cv.circle(image, (r_ankle_x, r_ankle_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_ankle_x, l_ankle_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing heel points
    # r_heel_x, r_heel_y = cie_landmark2videosize(landmark[land_idx.RIGHT_HEEL].x, landmark[land_idx.RIGHT_HEEL].y)
    l_heel_x, l_heel_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_HEEL].x, landmark[land_idx.LEFT_HEEL].y
    )
    # cv.circle(image, (r_heel_x, r_heel_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_heel_x, l_heel_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing toe points
    # r_toe_x, r_toe_y = cie_landmark2videosize(landmark[land_idx.RIGHT_FOOT_INDEX].x, landmark[land_idx.RIGHT_FOOT_INDEX].y)
    l_toe_x, l_toe_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_FOOT_INDEX].x, landmark[land_idx.LEFT_FOOT_INDEX].y
    )
    # cv.circle(image, (r_toe_x, r_toe_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_toe_x, l_toe_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing line connections
    # cv.line(image, (r_waist_x, r_waist_y), (r_knee_x, r_knee_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_waist_x, l_waist_y),
        (l_knee_x, l_knee_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_knee_x, r_knee_y), (r_ankle_x, r_ankle_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_knee_x, l_knee_y),
        (l_ankle_x, l_ankle_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_ankle_x, r_ankle_y), (r_heel_x, r_heel_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_ankle_x, l_ankle_y),
        (l_heel_x, l_heel_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_heel_x, r_heel_y), (r_toe_x, r_toe_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_heel_x, l_heel_y),
        (l_toe_x, l_toe_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )

    return image


def drawing_plank_landmarks(image, landmarks, land_idx) -> np.ndarray:
    """drawing a plank training landmark"""

    landmark = landmarks.landmark

    # drawing sholder, neck(c_sholder) points
    # r_sholder_x, r_sholder_y = cie_landmark2videosize(landmark[land_idx.RIGHT_SHOULDER].x, landmark[land_idx.RIGHT_SHOULDER].y)
    l_sholder_x, l_sholder_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_SHOULDER].x, landmark[land_idx.LEFT_SHOULDER].y
    )
    # cv.circle(image, (r_sholder_x, r_sholder_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image,
        (l_sholder_x, l_sholder_y),
        cl.RADIUS_SIZE7,
        cl.COLOR_BLUE,
        cl.THICKNES_SIZE3,
    )

    # drawing elbow points
    # r_elbow_x, r_elbow_y = cie_landmark2videosize(landmark[land_idx.RIGHT_ELBOW].x, landmark[land_idx.RIGHT_ELBOW].y)
    l_elbow_x, l_elbow_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_ELBOW].x, landmark[land_idx.LEFT_ELBOW].y
    )
    # cv.circle(image, (r_elbow_x, r_elbow_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image,
        (l_elbow_x, l_elbow_y),
        cl.RADIUS_SIZE7,
        cl.COLOR_BLUE,
        cl.THICKNES_SIZE3,
    )

    # drawing waist points
    # r_waist_x, r_waist_y = cie_landmark2videosize(landmark[land_idx.RIGHT_HIP].x, landmark[land_idx.RIGHT_HIP].y)
    l_waist_x, l_waist_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_HIP].x, landmark[land_idx.LEFT_HIP].y
    )
    # cv.circle(image, (r_waist_x, r_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_waist_x, l_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing knee points
    # r_knee_x, r_knee_y = cie_landmark2videosize(landmark[land_idx.RIGHT_KNEE].x, landmark[land_idx.RIGHT_KNEE].y)
    l_knee_x, l_knee_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_KNEE].x, landmark[land_idx.LEFT_KNEE].y
    )
    # cv.circle(image, (r_knee_x, r_knee_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_knee_x, l_knee_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing ankle points
    # r_ankle_x, r_ankle_y = cie_landmark2videosize(landmark[land_idx.RIGHT_ANKLE].x, landmark[land_idx.RIGHT_ANKLE].y)
    l_ankle_x, l_ankle_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_ANKLE].x, landmark[land_idx.LEFT_ANKLE].y
    )
    # cv.circle(image, (r_ankle_x, r_ankle_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_ankle_x, l_ankle_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing heel points
    # r_heel_x, r_heel_y = cie_landmark2videosize(landmark[land_idx.RIGHT_HEEL].x, landmark[land_idx.RIGHT_HEEL].y)
    l_heel_x, l_heel_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_HEEL].x, landmark[land_idx.LEFT_HEEL].y
    )
    # cv.circle(image, (r_heel_x, r_heel_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_heel_x, l_heel_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing toe points
    # r_toe_x, r_toe_y = cie_landmark2videosize(landmark[land_idx.RIGHT_FOOT_INDEX].x, landmark[land_idx.RIGHT_FOOT_INDEX].y)
    l_toe_x, l_toe_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_FOOT_INDEX].x, landmark[land_idx.LEFT_FOOT_INDEX].y
    )
    # cv.circle(image, (r_toe_x, r_toe_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_toe_x, l_toe_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing line connections
    # cv.line(image, (r_sholder_x, r_sholder_y), (r_elbow_x, r_elbow_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_sholder_x, l_sholder_y),
        (l_elbow_x, l_elbow_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )

    # cv.line(image, (r_waist_x, r_waist_y), (r_knee_x, r_knee_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_waist_x, l_waist_y),
        (l_knee_x, l_knee_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_knee_x, r_knee_y), (r_ankle_x, r_ankle_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_knee_x, l_knee_y),
        (l_ankle_x, l_ankle_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_ankle_x, r_ankle_y), (r_heel_x, r_heel_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_ankle_x, l_ankle_y),
        (l_heel_x, l_heel_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_heel_x, r_heel_y), (r_toe_x, r_toe_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_heel_x, l_heel_y),
        (l_toe_x, l_toe_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )

    return image


def drawing_push_up_landmarks(image, landmarks, land_idx) -> np.ndarray:
    """drawing a push up training landmark"""
    # toe, heel, knee, waist, sholder, elbow, wrist

    landmark = landmarks.landmark

    # drawing sholder, neck(c_sholder) points
    # r_sholder_x, r_sholder_y = cie_landmark2videosize(landmark[land_idx.RIGHT_SHOULDER].x, landmark[land_idx.RIGHT_SHOULDER].y)
    l_sholder_x, l_sholder_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_SHOULDER].x, landmark[land_idx.LEFT_SHOULDER].y
    )
    # cv.circle(image, (r_sholder_x, r_sholder_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image,
        (l_sholder_x, l_sholder_y),
        cl.RADIUS_SIZE7,
        cl.COLOR_BLUE,
        cl.THICKNES_SIZE3,
    )

    # drawing elbow points
    # r_elbow_x, r_elbow_y = cie_landmark2videosize(landmark[land_idx.RIGHT_ELBOW].x, landmark[land_idx.RIGHT_ELBOW].y)
    l_elbow_x, l_elbow_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_ELBOW].x, landmark[land_idx.LEFT_ELBOW].y
    )
    # cv.circle(image, (r_elbow_x, r_elbow_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image,
        (l_elbow_x, l_elbow_y),
        cl.RADIUS_SIZE7,
        cl.COLOR_BLUE,
        cl.THICKNES_SIZE3,
    )

    # drawing wrist points
    # r_wrist_x, r_wrist_y = cie_landmark2videosize(landmark[land_idx.RIGHT_WRIST].x, landmark[land_idx.RIGHT_WRIST].y)
    l_wrist_x, l_wrist_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_WRIST].x, landmark[land_idx.LEFT_WRIST].y
    )
    # cv.circle(image, (r_wrist_x, r_wrist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image,
        (l_wrist_x, l_wrist_y),
        cl.RADIUS_SIZE7,
        cl.COLOR_BLUE,
        cl.THICKNES_SIZE3,
    )

    # drawing waist points
    # r_waist_x, r_waist_y = cie_landmark2videosize(landmark[land_idx.RIGHT_HIP].x, landmark[land_idx.RIGHT_HIP].y)
    l_waist_x, l_waist_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_HIP].x, landmark[land_idx.LEFT_HIP].y
    )
    # cv.circle(image, (r_waist_x, r_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_waist_x, l_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing knee points
    # r_knee_x, r_knee_y = cie_landmark2videosize(landmark[land_idx.RIGHT_KNEE].x, landmark[land_idx.RIGHT_KNEE].y)
    l_knee_x, l_knee_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_KNEE].x, landmark[land_idx.LEFT_KNEE].y
    )
    # cv.circle(image, (r_knee_x, r_knee_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_knee_x, l_knee_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing ankle points
    # r_ankle_x, r_ankle_y = cie_landmark2videosize(landmark[land_idx.RIGHT_ANKLE].x, landmark[land_idx.RIGHT_ANKLE].y)
    l_ankle_x, l_ankle_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_ANKLE].x, landmark[land_idx.LEFT_ANKLE].y
    )
    # cv.circle(image, (r_ankle_x, r_ankle_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_ankle_x, l_ankle_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing heel points
    # r_heel_x, r_heel_y = cie_landmark2videosize(landmark[land_idx.RIGHT_HEEL].x, landmark[land_idx.RIGHT_HEEL].y)
    l_heel_x, l_heel_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_HEEL].x, landmark[land_idx.LEFT_HEEL].y
    )
    # cv.circle(image, (r_heel_x, r_heel_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_heel_x, l_heel_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing toe points
    # r_toe_x, r_toe_y = cie_landmark2videosize(landmark[land_idx.RIGHT_FOOT_INDEX].x, landmark[land_idx.RIGHT_FOOT_INDEX].y)
    l_toe_x, l_toe_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_FOOT_INDEX].x, landmark[land_idx.LEFT_FOOT_INDEX].y
    )
    # cv.circle(image, (r_toe_x, r_toe_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(
        image, (l_toe_x, l_toe_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3
    )

    # drawing line connections
    # cv.line(image, (r_sholder_x, r_sholder_y), (r_elbow_x, r_elbow_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_sholder_x, l_sholder_y),
        (l_elbow_x, l_elbow_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )

    # cv.line(image, (r_elbow_x, r_elbow_y), (r_wrist_x, r_wrist_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_elbow_x, l_elbow_y),
        (l_wrist_x, l_wrist_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )

    # cv.line(image, (r_waist_x, r_waist_y), (r_knee_x, r_knee_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_waist_x, l_waist_y),
        (l_knee_x, l_knee_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_knee_x, r_knee_y), (r_ankle_x, r_ankle_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_knee_x, l_knee_y),
        (l_ankle_x, l_ankle_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_ankle_x, r_ankle_y), (r_heel_x, r_heel_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_ankle_x, l_ankle_y),
        (l_heel_x, l_heel_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )
    # cv.line(image, (r_heel_x, r_heel_y), (r_toe_x, r_toe_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(
        image,
        (l_heel_x, l_heel_y),
        (l_toe_x, l_toe_y),
        cl.COLOR_GREEN,
        cl.THICKNES_SIZE3,
        cv.LINE_4,
        0,
    )

    return image


def check_straight_neck(landmarks, land_idx) -> float:
    """
    check straight neck
    out angle: over 30 degree
    """

    cie = namedtuple("Coordinates", ["x", "y"])
    landmark = landmarks.landmark

    l_ear_x, l_ear_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_EAR].x, landmark[land_idx.LEFT_EAR].y
    )

    l_sholder_x, l_sholder_y = cie_landmark2videosize(
        landmark[land_idx.LEFT_SHOULDER].x, landmark[land_idx.LEFT_SHOULDER].y
    )

    ex_l_sholder = cie(l_sholder_x, l_sholder_y - cl.EXTENTOIN_CIE)

    d1 = math.sqrt(
        (l_sholder_x - ex_l_sholder.x) ** 2 + (l_sholder_y - ex_l_sholder.y) ** 2
    )
    d2 = math.sqrt((l_sholder_x - l_ear_x) ** 2 + (l_sholder_y - l_ear_y) ** 2)
    d3 = math.sqrt((ex_l_sholder.x - l_ear_x) ** 2 + (ex_l_sholder.y - l_ear_y) ** 2)

    cos_theta = (d1**2 + d2**2 - d3**2) / (2 * d1 * d2)
    straight_neck_angle = math.degrees(math.acos(cos_theta))

    return straight_neck_angle


if __name__ == "__main__":
    main_training()
