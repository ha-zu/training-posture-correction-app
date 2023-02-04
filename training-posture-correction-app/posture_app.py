import time

import cv2 as cv
import mediapipe as mp
import numpy as np
from utils import constant_list as cl
from utils import drawing as dw
from utils import landmark_calculator as lc
from utils import saving as sv


def main_training(train_mode: str, side: str):
    """
    the right side of body should face the camera and 1m away.
    """

    # # load pose model
    mp_pose = mp.solutions.pose

    # setting capture
    cap = cv.VideoCapture(cl.VIDEO_CAMERA_MAC)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cl.VIDEO_FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cl.VIDEO_FRAME_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, 15)
    writer = sv.recording_format()

    # setting holistic
    with mp_pose.Pose(
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            success, cap_img = cap.read()
            if not success:
                print("【Error】can not opend video capture.")
                break

            # convert color for use mediapipe
            cap_img = cv.cvtColor(cap_img, cv.COLOR_BGR2RGB)

            # Inverted with respect to Y axis
            # Detected inverted right and left
            cap_img = cv.flip(cap_img, 1)
            cap_img.flags.writeable = False
            # get detections
            results = pose.process(cap_img)
            cap_img.flags.writeable = True

            # convert color for use opencv
            cap_img = cv.cvtColor(cap_img, cv.COLOR_RGB2BGR)

            # get landmarks
            landmarks = results.pose_landmarks

            # masking captured image(demo mode)
            try:
                condition = np.stack((results.segmentation_mask,) * 3, axis=2) > 0.1
                bg_image = np.zeros(cap_img.shape, dtype=np.uint8)
                bg_image[:] = (229, 229, 229)
                cap_img = np.where(condition, cap_img, bg_image)
            except np.AxisError:
                p1 = (0, 0)
                p2 = (cl.VIDEO_FRAME_WIDTH, cl.VIDEO_FRAME_HEIGHT)
                cap_img = cv.rectangle(cap_img, p1, p2, cl.COLOR_BLACK, -1)
                cap_img = dw.drawing_text(cap_img)

            # drawing base posture points
            if landmarks is not None:
                # get using landmark lists
                landmarks = landmarks.landmark
                land_lists = using_landmark_lists(landmarks)

                # set landmark points
                cap_img = dw.drawing_landmark_points(
                    cap_img, land_lists, train_mode, side
                )

                if train_mode is None:
                    # check_straight_neck
                    neck_angle = lc.calculate_neck_angle(land_lists, side)
                    print(neck_angle)
                    if neck_angle > cl.STRAIGHT_NECK_ANGLE:
                        color = cl.COLOR_RED
                        start = time.time()
                        print(start)
                    else:
                        color = cl.COLOR_GREEN
                        end = time.time() - start
                        print(end)
                elif train_mode == cl.SQUAT:
                    # knee degree
                    knee_angle = lc.calculate_knee_angle(land_lists)
                    if knee_angle < cl.SQUAT_ANGLE:
                        count = +1
                        print(count)
                elif train_mode == cl.PLANK:
                    # arm degree
                    elbow_angle = lc.calculate_elbow_angle(land_lists)
                    armpits_angle = lc.calculate_armpits_angle(land_lists)
                    check_plank_posture = lc.calculate_plank_posture(land_lists)
                    if (
                        elbow_angle == cl.PLANK_ANGLE
                        and armpits_angle == cl.PLANK_ANGLE
                        and check_plank_posture
                    ):
                        pass
                elif train_mode == cl.PUSH_UP:
                    # arm degree
                    elbow_angle = lc.calculate_elbow_angle(land_lists)
                    if elbow_angle < cl.PUSH_UP_ANGLE:
                        count = +1

                # set connect between landmark points
                cap_img = dw.draw_lines_between_landmarks(
                    cap_img, land_lists, train_mode, side, color
                )

            cv.imshow("video capture image", cap_img)
            writer.write(cap_img)

            # key esc pressed stop capture
            if cv.waitKey(1) == 27:
                break

    writer.release()
    cap.release()
    cv.destroyAllWindows()


def using_landmark_lists(landmarks) -> list:
    """
    Convert the landmarks to be used into
    video capture coordinates and store them

    input: points detected by mediapipe
    output: using landmark list
    """

    using_landmark = []

    for index in cl.LANDMARK_INDEXES:
        landmark = lc.calculate_landmark2video_coords(landmarks[index])
        using_landmark.append(landmark)

    center_l_shoulder = lc.calculate_center_landmark(using_landmark[2])
    center_r_shoulder = lc.calculate_center_landmark(using_landmark[3])
    center_shoulder_x = center_l_shoulder[cl.X] + center_r_shoulder[cl.X]
    center_shoulder_y = center_l_shoulder[cl.Y] + center_r_shoulder[cl.Y]

    center_l_waist = lc.calculate_center_landmark(using_landmark[8])
    center_r_waist = lc.calculate_center_landmark(using_landmark[9])
    center_waist_x = center_l_waist[cl.X] + center_r_waist[cl.X]
    center_waist_y = center_l_waist[cl.Y] + center_r_waist[cl.Y]

    using_landmark.append((center_shoulder_x, center_shoulder_y))
    using_landmark.append((center_waist_x, center_waist_y))

    return using_landmark


if __name__ == "__main__":
    main_training(train_mode=None, side=cl.LEFT_SIDE)
