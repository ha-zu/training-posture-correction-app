from collections import namedtuple
import cv2 as cv
import mediapipe as mp
import numpy as np
import os
from typing import Tuple
from typing import Any
import uuid

from constant import constant_list as cl

def main_training(train_mode):
    # print test
    print(train_mode)
    # load pose model
    mp_pose = mp.solutions.pose

    # setting capture
    cap = cv.VideoCapture(cl.VIDEO_CAMERA_MAC)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cl.VIDEO_FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cl.VIDEO_FRAME_HEIGHT)
    writer = recording_format()

    # setting holistic
    with mp_pose.Pose(
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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

            # masking captured image
            condition = np.stack((results.segmentation_mask,) * 3, axis=2) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (229, 229, 229)
            image = np.where(condition, image, bg_image)

            # drawing base posture points
            image = drawing_base_landmarks(image, landmarks, landmarks_idx)

            cv.imshow("video capture image", image)
            writer.write(image)

            # key esc pressed stop capture
            if cv.waitKey(1) == 27:
                break

    writer.release()
    cap.release()
    cv.destroyAllWindows()


def cie_landmark2videosize(landmark_x, landmark_y)->Tuple[int, int]:
    """landmark coordinates to video coordinates"""

    x = int(landmark_x * cl.VIDEO_FRAME_WIDTH)
    y = int(landmark_y * cl.VIDEO_FRAME_HEIGHT)

    return x, y

def recording_format()->Any:
    """recording video format"""

    fmt = cv.VideoWriter_fourcc("m", "p", "4", "v")
    video_out = os.path.join(os.getcwd(), cl.VIDEO_OUT_DIR, '{}.mp4'.format(uuid.uuid1()))
    writer = cv.VideoWriter(video_out, fmt, cl.VIDEO_FRAME_RATE, (cl.VIDEO_FRAME_WIDTH, cl.VIDEO_FRAME_HEIGHT))

    return writer


def drawing_base_landmarks(image, landmarks, land_idx)->np.ndarray:
    """drawing a base landmark"""

    # center points coordinates
    cie = namedtuple('Coordinates', ['x', 'y'])
    landmark = landmarks.landmark

    # drawing ear points
    r_ear_x, r_ear_y = cie_landmark2videosize(landmark[land_idx.RIGHT_EAR].x, landmark[land_idx.RIGHT_EAR].y)
    l_ear_x, l_ear_y = cie_landmark2videosize(landmark[land_idx.LEFT_EAR].x, landmark[land_idx.LEFT_EAR].y)
    cv.circle(image, (r_ear_x, r_ear_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(image, (l_ear_x, l_ear_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)

    # drawing sholder, neck(c_sholder) points
    r_sholder_x, r_sholder_y = cie_landmark2videosize(landmark[land_idx.RIGHT_SHOULDER].x, landmark[land_idx.RIGHT_SHOULDER].y)
    l_sholder_x, l_sholder_y = cie_landmark2videosize(landmark[land_idx.LEFT_SHOULDER].x, landmark[land_idx.LEFT_SHOULDER].y)
    c_sholder = cie(int((r_sholder_x + l_sholder_x) / 2), int((r_sholder_y + l_sholder_y) / 2))
    cv.circle(image, (r_sholder_x, r_sholder_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(image, (l_sholder_x, l_sholder_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(image, (c_sholder.x, c_sholder.y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)

    # drawing waist points
    r_waist_x, r_waist_y = cie_landmark2videosize(landmark[land_idx.RIGHT_HIP].x, landmark[land_idx.RIGHT_HIP].y)
    l_waist_x, l_waist_y = cie_landmark2videosize(landmark[land_idx.LEFT_HIP].x, landmark[land_idx.LEFT_HIP].y)
    c_waist = cie(int((r_waist_x + l_waist_x) / 2), int((r_waist_y + l_waist_y) / 2))
    cv.circle(image, (r_waist_x, r_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(image, (l_waist_x, l_waist_y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)
    cv.circle(image, (c_waist.x, c_waist.y), cl.RADIUS_SIZE7, cl.COLOR_BLUE, cl.THICKNES_SIZE3)

    # drawing line connections
    cv.line(image, (r_ear_x, r_ear_y), (r_sholder_x, r_sholder_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(image, (l_ear_x, l_ear_y), (l_sholder_x, l_sholder_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(image, (r_sholder_x, r_sholder_y), (c_sholder.x, c_sholder.y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(image, (l_sholder_x, l_sholder_y), (c_sholder.x, c_sholder.y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(image, (r_sholder_x, r_sholder_y), (r_waist_x, r_waist_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(image, (l_sholder_x, l_sholder_y), (l_waist_x, l_waist_y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(image, (r_waist_x, r_waist_y), (c_waist.x, c_waist.y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)
    cv.line(image, (l_waist_x, l_waist_y), (c_waist.x, c_waist.y), cl.COLOR_GREEN, cl.THICKNES_SIZE3, cv.LINE_4, 0)

    return image


def drawing_squad_landmarks(image, landmarks, landmark_idx)->np.ndarray:
    """drawing a squat training landmark"""
    # toe, heel, knee, waist, sholder, neck, head
    pass


def drawing_plank_landmarks(image, landmarks, landmark_idx)->np.ndarray:
    """drawing a plank training landmark"""
    # toe, heel, knee, waist, sholder, elbow, neck, head
    pass


def drawing_push_up_landmarks(image, landmarks, landmark_idx)->np.ndarray:
    """drawing a push up training landmark"""
    # toe, heel, knee, waist, sholder, elbow, neck, head
    pass


if __name__ == "__main__":
    main_training("test")
