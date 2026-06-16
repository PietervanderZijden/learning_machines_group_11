"""Interactive HSV tuning for green food block detection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="HSV tuning for green food blocks")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--image", type=str, default=None, help="Path to image (skip sim)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read {args.image}")
            sys.exit(1)
    else:
        import os
        os.environ["COPPELIA_SIM_PORT"] = str(args.port)
        from robobo_interface import SimulationRobobo
        rob = SimulationRobobo()
        if rob.is_stopped():
            rob.play_simulation()
        img = rob.read_image_front()
        rob.stop_simulation()

    hsv_low = [35, 80, 80]
    hsv_high = [85, 255, 255]
    window = "HSV Tuner"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("H Low", window, hsv_low[0], 179, lambda x: None)
    cv2.createTrackbar("S Low", window, hsv_low[1], 255, lambda x: None)
    cv2.createTrackbar("V Low", window, hsv_low[2], 255, lambda x: None)
    cv2.createTrackbar("H High", window, hsv_high[0], 179, lambda x: None)
    cv2.createTrackbar("S High", window, hsv_high[1], 255, lambda x: None)
    cv2.createTrackbar("V High", window, hsv_high[2], 255, lambda x: None)

    print("Adjust trackbars. Press 'q' to quit, 's' to save.")

    while True:
        hsv_low[0] = cv2.getTrackbarPos("H Low", window)
        hsv_low[1] = cv2.getTrackbarPos("S Low", window)
        hsv_low[2] = cv2.getTrackbarPos("V Low", window)
        hsv_high[0] = cv2.getTrackbarPos("H High", window)
        hsv_high[1] = cv2.getTrackbarPos("S High", window)
        hsv_high[2] = cv2.getTrackbarPos("V High", window)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display = img.copy()
        cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        combined = np.hstack([display, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(window, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            print(f"hsv_low = ({hsv_low[0]}, {hsv_low[1]}, {hsv_low[2]})")
            print(f"hsv_high = ({hsv_high[0]}, {hsv_high[1]}, {hsv_high[2]})")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
