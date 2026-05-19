"""ROS2 node + spin thread that publishes detected-object 3D positions."""

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Pose
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from vision_msgs.msg import (
    Detection3DArray, Detection3D, BoundingBox3D,
    ObjectHypothesisWithPose,
)

from .tf_utils import (
    lookup_camera_to_robot,
    static_transform_from_config,
    transform_point_mm,
)


def _now_msg(node):
    t = node.get_clock().now().to_msg()
    out = TimeMsg()
    out.sec = t.sec
    out.nanosec = t.nanosec
    return out


def _build_detection3d(xyz_m, cls_id, cls_name, score, frame_id, stamp):
    """One ``vision_msgs/Detection3D`` (center pose only; size = zero unknown)."""
    d = Detection3D()
    d.header.frame_id = frame_id
    d.header.stamp = stamp

    hyp = ObjectHypothesisWithPose()
    # vision_msgs Humble: hypothesis.class_id (str), hypothesis.score (float)
    # vision_msgs Jazzy : same path. Use hasattr probe for cross-version safety.
    if hasattr(hyp, "hypothesis"):
        hyp.hypothesis.class_id = str(cls_name if cls_name else cls_id)
        hyp.hypothesis.score = float(score)
    else:  # very old (Foxy) fallback
        hyp.id = str(cls_name if cls_name else cls_id)  # type: ignore[attr-defined]
        hyp.score = float(score)  # type: ignore[attr-defined]

    pose = Pose()
    pose.position.x = float(xyz_m[0])
    pose.position.y = float(xyz_m[1])
    pose.position.z = float(xyz_m[2])
    pose.orientation.w = 1.0
    hyp.pose.pose = pose

    d.results.append(hyp)

    bbox = BoundingBox3D()
    bbox.center = pose
    bbox.size.x = 0.0
    bbox.size.y = 0.0
    bbox.size.z = 0.0
    d.bbox = bbox
    return d


class DetectionPublisher(Node):
    """ROS2 node: publishes ``Detection3DArray`` + static TF, optional TF lookup."""

    def __init__(self, ros_cfg):
        super().__init__(ros_cfg.get("node_name", "elp_detector"))
        self.camera_frame = str(ros_cfg.get("camera_frame", "camera_optical_frame"))
        self.robot_frame = str(ros_cfg.get("robot_frame", "robot_base"))
        self.use_tf_lookup = bool(ros_cfg.get("use_tf_lookup", False))

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.history = HistoryPolicy.KEEP_LAST
        topic = str(ros_cfg.get("topic", "/elp/detections"))
        self._pub = self.create_publisher(Detection3DArray, topic, qos)

        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        self._static_br = StaticTransformBroadcaster(self)
        st = ros_cfg.get("static_tf")
        if st:
            tf_msg = static_transform_from_config(st)
            tf_msg.header.stamp = _now_msg(self)
            self._static_br.sendTransform(tf_msg)
            self.get_logger().info(
                f"Static TF: {tf_msg.header.frame_id} -> {tf_msg.child_frame_id}"
            )

        self._last_lookup_ok = None  # tri-state: None/True/False

    @property
    def last_lookup_ok(self):
        return self._last_lookup_ok

    def publish_detections(self, detections):
        """Publish a list of ``DetectionWorker`` result dicts.

        Each item must have ``det`` (Detection) and ``xyz_mm`` (X,Y,Z mm in
        camera frame). Performs optional ``camera -> robot`` TF lookup.
        """
        stamp = _now_msg(self)
        msg = Detection3DArray()
        msg.header.frame_id = self.camera_frame
        msg.header.stamp = stamp

        M = None
        publish_frame = self.camera_frame
        if self.use_tf_lookup:
            M = lookup_camera_to_robot(
                self._tf_buf, self.robot_frame, self.camera_frame,
            )
            self._last_lookup_ok = M is not None
            if M is not None:
                publish_frame = self.robot_frame
                msg.header.frame_id = self.robot_frame

        for item in detections:
            xyz_mm = item["xyz_mm"]
            if M is not None:
                xyz_m = transform_point_mm(M, xyz_mm)
            else:
                xyz_m = (xyz_mm[0] / 1000.0, xyz_mm[1] / 1000.0, xyz_mm[2] / 1000.0)
            det = item["det"]
            msg.detections.append(_build_detection3d(
                xyz_m, det.cls_id, det.cls_name, det.score,
                publish_frame, stamp,
            ))

        self._pub.publish(msg)
        return len(msg.detections)


class RosSpinThread(threading.Thread):
    """Spin a node in a daemon thread so the PyQt event loop owns the main thread."""

    def __init__(self, node):
        super().__init__(daemon=True)
        self._node = node
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)
