"""Image pre-processing helpers for the detection pipeline."""

import cv2


def clahe_bgr(bgr, clip=2.0, grid=(8, 8)):
    """Return a CLAHE-equalised copy of ``bgr``.

    Operates on the L channel of LAB so colour is preserved while local
    contrast is restored. Useful when lighting varies across the frame.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=float(clip),
                         tileGridSize=(int(grid[0]), int(grid[1])))
    l2 = cl.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
