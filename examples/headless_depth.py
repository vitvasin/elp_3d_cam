#!/usr/bin/env python3
"""Example: Headless depth capture using the StereoPipeline API.

This script demonstrates how to integrate the ELP 3D stereo camera depth
processing into a standalone Python application without using the PyQt5 GUI.
"""

import cv2
import numpy as np
from elp_stereo import StereoPipeline


def main():
    # 1. Initialize the pipeline.
    # It automatically loads the default config from config/default.yaml.
    pipeline = StereoPipeline()

    # 2. Load calibration (required for depth).
    # Update this path to your actual calibration file produced by the GUI.
    calib_path = "config/stereo_calib (backup).yaml"
    print(f"Attempting to load calibration from: {calib_path}")
    try:
        pipeline.load_calibration(calib_path)
        print("Calibration loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load calibration: {e}")
        print("Pipeline will operate in raw/rectified mode without depth.")

    # 3. Start the camera.
    print("Opening camera...")
    try:
        pipeline.start()
    except Exception as e:
        print(f"Error: Failed to open camera: {e}")
        return

    print("Pipeline started. Press 'q' in the window to exit.")
    try:
        while True:
            # 4. Grab a frame and compute depth.
            # Returns (left_rect, right_rect, depth_map).
            result = pipeline.grab_depth()
            if result is None:
                print("\nFailed to grab frame from camera.")
                break

            left_rect, right_rect, depth_map = result

            # 5. Visualization.
            # Show the rectified left image.
            cv2.imshow("Left Rectified", left_rect)

            # Show the colorized depth map if available.
            color_depth = pipeline.get_colorized_depth()
            if color_depth is not None:
                cv2.imshow("Depth Map", color_depth)

                # Example: Read depth at the center pixel.
                h, w = depth_map.shape
                z = depth_map[h // 2, w // 2]
                if np.isfinite(z):
                    print(f"\rDepth at center ({w//2}, {h//2}): {z:.1f} mm   ",
                          end="", flush=True)
                else:
                    print(f"\rDepth at center ({w//2}, {h//2}): Out of range",
                          end="", flush=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 6. Cleanup.
        print("\nStopping pipeline and releasing camera...")
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
