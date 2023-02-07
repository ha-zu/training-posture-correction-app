import os
import uuid
from typing import Any

import cv2 as cv
from utils import constant_list as cl


def recording_format() -> Any:
    """recording video format"""

    fmt = cv.VideoWriter_fourcc("m", "p", "4", "v")
    filename = os.path.join(
        os.getcwd(), cl.VIDEO_OUT_DIR, "{}.mp4".format(uuid.uuid1())
    )
    writer = cv.VideoWriter(
        filename,
        fmt,
        cl.VIDEO_FRAME_RATE10,
        (cl.VIDEO_FRAME_WIDTH, cl.VIDEO_FRAME_HEIGHT),
    )

    return writer


def save_image(image):
    """save capture image"""

    filename = os.path.join(
        os.getcwd(), cl.IMAGE_OUT_DIR, "{}.png".format(uuid.uuid1())
    )

    cv.imwrite(filename, image)
