"""ROS2 publisher + TF utilities for detected-object 3D positions.

Imports are lazy and guarded — a missing ``rclpy`` or ``vision_msgs`` does
not break the rest of the package. Probe ``ROS2_AVAILABLE`` before using.
"""

try:
    import rclpy  # noqa: F401
    from vision_msgs.msg import Detection3DArray  # noqa: F401
    ROS2_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001
    ROS2_AVAILABLE = False
    _IMPORT_ERROR = str(_exc)

__all__ = ["ROS2_AVAILABLE", "_IMPORT_ERROR"]

if ROS2_AVAILABLE:
    from .publisher import DetectionPublisher, RosSpinThread  # noqa: F401
    from .tf_utils import (  # noqa: F401
        static_transform_from_config,
        lookup_camera_to_robot,
        transform_point_mm,
    )
    __all__ += [
        "DetectionPublisher",
        "RosSpinThread",
        "static_transform_from_config",
        "lookup_camera_to_robot",
        "transform_point_mm",
    ]
