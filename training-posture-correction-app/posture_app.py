import cv2 as cv
import mediapipe as mp
import numpy as np
from utils import check_posture as cp
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
    rep_count = 0
    base_posture = False

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
                cap_img = dw.drawing_text(cap_img, cl.LOADING)

            # drawing base posture points
            if landmarks is not None:
                # get using landmark lists
                landmarks = landmarks.landmark
                land_lists = lc.using_landmark_lists(landmarks)

                # set landmark points
                cap_img = dw.drawing_landmark_points(
                    cap_img, land_lists, train_mode, side
                )

                color = cl.COLOR_GREEN
                if train_mode is None:
                    # check_straight_neck
                    neck_angle = lc.calculate_neck_angle(land_lists, side)

                    if neck_angle > cl.STRAIGHT_NECK_ANGLE:
                        color = cl.COLOR_RED
                    else:
                        color = cl.COLOR_GREEN

                elif train_mode == cl.SQUAT:
                    # knee degree
                    knee_angle = lc.calculate_knee_angle(land_lists)
                    position = cp.calculate_knee_toe_position(land_lists)

                    if not position:
                        # if knees are in front of your toe
                        color = cl.COLOR_RED

                    if knee_angle > cl.SQUAT_ANGLE:
                        # standing posture
                        base_posture = True

                    if knee_angle < cl.SQUAT_ANGLE and base_posture and position:
                        # angle and before standing position and if knees are not in front of your toe
                        base_posture = False
                        rep_count += 1

                elif train_mode == cl.PLANK:
                    # arm degree
                    elbow_angle = lc.calculate_elbow_angle(land_lists)
                    armpits_angle = lc.calculate_armpits_angle(land_lists)
                    if (
                        (
                            elbow_angle <= cl.PLANK_ANGLE_UPPER
                            and elbow_angle >= cl.PLANK_ANGLE_LOWER
                        )
                        and (
                            armpits_angle <= cl.PLANK_ANGLE_UPPER
                            and armpits_angle >= cl.PLANK_ANGLE_LOWER
                        )
                    ):
                        # Todo timer
                        pass
                    else:
                        color = cl.COLOR_RED

                elif train_mode == cl.PUSH_UP:
                    # arm degree
                    elbow_angle = lc.calculate_elbow_angle(land_lists)

                    if elbow_angle > cl.PUSH_UP_ANGLE_UPPER:
                        base_posture = True

                    if elbow_angle < cl.PUSH_UP_ANGLE_LOWER and base_posture:
                        base_posture = False
                        rep_count += 1

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


if __name__ == "__main__":
    main_training(train_mode=cl.SQUAT, side=cl.LEFT_SIDE)
