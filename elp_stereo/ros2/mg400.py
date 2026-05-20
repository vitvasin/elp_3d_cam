"""MG400 ROS2 helpers used by the ELP hand-eye and pick-control apps."""

from __future__ import annotations

import math
import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from mg400_msgs.action import MovJ, MovL
from mg400_msgs.srv import ClearError, DisableRobot, DO, EnableRobot, GetPose, ToolDOExecute


class MG400Node(Node):
    """Small ROS2 facade for the MG400 services/actions used by vision apps."""

    def __init__(self, node_name="elp_mg400_node"):
        super().__init__(node_name)
        self.cli_clear_error = self.create_client(ClearError, "/mg400/clear_error")
        self.cli_enable = self.create_client(EnableRobot, "/mg400/enable_robot")
        self.cli_disable = self.create_client(DisableRobot, "/mg400/disable_robot")
        self.cli_pose = self.create_client(GetPose, "/mg400/get_pose")
        self.cli_do = self.create_client(DO, "/mg400/do_execute")
        self.cli_tool_do = self.create_client(ToolDOExecute, "/mg400/tool_do_execute")
        self.act_mov_j = ActionClient(self, MovJ, "/mg400/mov_j")
        self.act_mov_l = ActionClient(self, MovL, "/mg400/mov_l")

    def call_simple_service_async(self, client, request, name, on_done=None, timeout=2.0):
        """Call a simple MG400 service and report ``(ok, msg)``."""
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=timeout):
                if on_done:
                    on_done(False, f"{name} unavailable")
                return

        def _cb(fut):
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                if on_done:
                    on_done(False, str(exc))
                return
            err = getattr(res, "error_id", 0)
            ok = err == 0
            if on_done:
                on_done(ok, "ok" if ok else f"error_id={err}")

        client.call_async(request).add_done_callback(_cb)

    def clear_error_async(self, on_done=None):
        self.call_simple_service_async(
            self.cli_clear_error, ClearError.Request(), "/mg400/clear_error", on_done
        )

    def enable_robot_async(self, on_done=None):
        self.call_simple_service_async(
            self.cli_enable, EnableRobot.Request(), "/mg400/enable_robot", on_done
        )

    def disable_robot_async(self, on_done=None):
        self.call_simple_service_async(
            self.cli_disable, DisableRobot.Request(), "/mg400/disable_robot", on_done
        )

    def request_pose_async(self, done_cb, timeout=3.0):
        """Call ``/mg400/get_pose``. Callback receives ``(pose_tuple, err)``.

        Pose tuple is ``(x, y, z)`` in meters, matching the existing MG400 GUI.
        """
        if not self.cli_pose.service_is_ready():
            if not self.cli_pose.wait_for_service(timeout_sec=timeout):
                done_cb(None, f"/mg400/get_pose unavailable after {timeout}s")
                return

        future = self.cli_pose.call_async(GetPose.Request())

        def _on_done(fut):
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                done_cb(None, str(exc))
                return
            if res is None:
                done_cb(None, "empty pose response")
                return
            if getattr(res, "error_id", 0) != 0:
                done_cb(None, f"robot error_id={res.error_id}")
                return
            done_cb((float(res.pose1), float(res.pose2), float(res.pose3)), None)

        future.add_done_callback(_on_done)

    def move_cartesian_async(self, x, y, z, r_deg, is_linear=False, on_done=None):
        """Send a Cartesian move goal. Coordinates are meters, R is yaw degrees."""
        client = self.act_mov_l if is_linear else self.act_mov_j
        action = MovL if is_linear else MovJ
        if not client.wait_for_server(timeout_sec=2.0):
            if on_done:
                on_done(False, "move action server unavailable")
            return

        r_rad = math.radians(float(r_deg))
        goal = action.Goal()
        goal.pose.header.frame_id = "mg400_origin_link"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = float(z)
        goal.pose.pose.orientation.z = math.sin(r_rad / 2.0)
        goal.pose.pose.orientation.w = math.cos(r_rad / 2.0)

        def _goal_resp(fut):
            try:
                handle = fut.result()
            except Exception as exc:  # noqa: BLE001
                if on_done:
                    on_done(False, f"send failed: {exc}")
                return
            if not handle.accepted:
                if on_done:
                    on_done(False, "goal rejected")
                return

            def _result(rf):
                try:
                    result = rf.result().result
                    ok = bool(result.result)
                except Exception as exc:  # noqa: BLE001
                    if on_done:
                        on_done(False, f"result failed: {exc}")
                    return
                if on_done:
                    on_done(ok, "ok" if ok else "action returned false")

            handle.get_result_async().add_done_callback(_result)

        client.send_goal_async(goal).add_done_callback(_goal_resp)

    def set_do_async(self, use_tool_do, index, status, on_done=None, timeout=2.0):
        """Set base/tool DO. ``status`` is 1 or 0."""
        if use_tool_do:
            req = ToolDOExecute.Request()
            client = self.cli_tool_do
        else:
            req = DO.Request()
            client = self.cli_do
        req.index.index = int(index)
        req.status.status = int(status)

        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=timeout):
                if on_done:
                    on_done(False, "DO service unavailable")
                return

        def _cb(fut):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                if on_done:
                    on_done(False, str(exc))
                return
            if on_done:
                on_done(True, "ok")

        client.call_async(req).add_done_callback(_cb)


class RosSpinThread(threading.Thread):
    """Daemon spin thread for non-ROS GUI event loops."""

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
