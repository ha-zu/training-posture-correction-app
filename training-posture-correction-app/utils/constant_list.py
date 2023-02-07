"""constants list"""

# use constants
RADIUS_SIZE7 = 7
THICKNES_SIZE3 = 3
EXTENTOIN_CIE = 100
LOADING = "Loading..."

# color set is B, G, R
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 89, 89)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 204, 0)
VIDEO_FRAME_WIDTH = 1280
VIDEO_FRAME_HEIGHT = 720
VIDEO_FRAME_CENTER = (590, 310)
VIDEO_CAMERA_MAC = 0
VIDEO_CAMERA_USB = 1
VIDEO_OUT_DIR = "data"
IMAGE_OUT_DIR = "image"
VIDEO_FRAME_RATE10 = 10

# training menu
DESK_WORK = "desk work"
PLANK = "plank"
PUSH_UP = "push up"
SQUAT = "squat"

# detection side
RIGHT_SIDE = 1
LEFT_SIDE = 2

# base angle
STRAIGHT_NECK_ANGLE = 30.0
# 90 degrees is best, but loosely estimated
SQUAT_ANGLE = 100.0
PLANK_ANGLE_UPPER = 100.0
PUSH_UP_ANGLE_UPPER = 100.0
PLANK_ANGLE_LOWER = 80.0
PUSH_UP_ANGLE_LOWER = 80.0

# detect points
X = 0
Y = 1
EX_POINT = 2

# detected points are reversed on the left and right
LANDMARK_INDEXES = [
    7,  # ear_right 0
    8,  # ear_left 1
    11,  # shoulder_right 2
    12,  # shoulder_left 3
    13,  # elbow_right 4
    14,  # elbow_left 5
    15,  # wrist_right 6
    16,  # wrist_left 7
    23,  # hip_right 8
    24,  # hip_left 9
    25,  # knee_right 10
    26,  # knee_left 11
    27,  # ankle_right 12
    28,  # ankle_left 13
    29,  # heel_right 14
    30,  # heel_left 15
    31,  # foot_index_right 16
    32,  # foot_index_left 17
    # ex center shoulder 18
    # ex center waist 19
]

# detct connection points
# detected points are reversed on the left and right
FULL_BODY_CONNECT = [
    (0, 2),  # right ear shoulder
    (1, 3),  # left ear shoulder
    (2, 4),  # right shoulder elbow
    (3, 5),  # left shoulder elbow
    (4, 6),  # right elbow wrist
    (5, 7),  # left elbow wrist
    (2, 8),  # right shoulder waist
    (3, 9),  # left shoulder waist
    (8, 10),  # right waist knee
    (9, 11),  # left waist knee
    (10, 12),  # right knee ankle
    (11, 13),  # left knee ankle
    (12, 14),  # right ankle heel
    (13, 15),  # left ankle heel
    (14, 16),  # right heel foot_index
    (15, 17),  # left heel foot_index
    (2, 18),  # right shoulder center_shoulder
    (3, 18),  # left shoulder center_shoulder
    (8, 19),  # right waist center_waist
    (9, 19),  # left waist center_waist
    (18, 19),  # center shoulder, center waist
]

# detected points are reversed on the left and right
RIGHT_SIDE_UPPER_CONNECT = [
    (0, 2),  # right ear shoulder
    (2, 4),  # right shoulder elbow
    (4, 6),  # right elbow wrist
    (2, 8),  # right shoulder waist
]

RIGHT_SIDE_LOWER_CONNECT = [
    (8, 10),  # right waist knee
    (10, 12),  # right knee ankle
    (12, 14),  # right ankle heel
    (14, 16),  # right heel foot_index
]

LEFT_SIDE_UPPER_CONNECT = [
    (1, 3),  # left ear shoulder
    (3, 5),  # left shoulder elbow
    (5, 7),  # left elbow wrist
    (3, 9),  # left shoulder waist
]

LEFT_SIDE_LOWER_CONNECT = [
    (9, 11),  # left waist knee
    (11, 13),  # left knee ankle
    (13, 15),  # left ankle heel
    (15, 17),  # left heel foot_index
]
