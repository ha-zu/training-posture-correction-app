from utils import constant_list as cl


def calculate_knee_toe_position(landmark) -> bool:

    right_knee = landmark[10]
    left_knee = landmark[11]
    right_foot_index = landmark[16]
    left_foot_index = landmark[17]
    sum_right_knee = sum(right_knee)
    sum_left_knee = sum(left_knee)
    sum_right_foot_index = sum(right_foot_index)
    sum_left_foot_index = sum(left_foot_index)

    right = sum([sum_right_knee, sum_right_foot_index])
    left = sum([sum_left_knee, sum_left_foot_index])

    if right > left:
        position = True if right_foot_index[cl.X] < right_knee[cl.X] else False
    else:
        position = True if left_foot_index[cl.X] < left_knee[cl.X] else False

    return position


def calculate_plank_posture(landmark) -> bool:
    """
    head angle
    shoulder points
    waist points
    knee points
    ankle points
    """
    pass
