"""
3x3 Grid Cell Occupancy Detector — RealSense RGB-D Video
=========================================================
Input : MP4 where LEFT half = RGB, RIGHT half = depth colormap
Output: Per-frame grid cell status (EMPTY / FULL) with visual overlay

Algorithm overview
------------------
1.  Adaptive threshold → morphological H/V line detection to locate the grid.
2.  Projection profiles of the line image to find the 4 horizontal and
    4 vertical grid-line positions → 9 cell ROIs.
3.  Each cell's inner 50% patch is evaluated in HSV:
      EMPTY  →  low saturation (<35) AND moderate-high brightness (val > 100)
                (the background / mall interior is visible through the opening)
      FULL   →  everything else  (colored object present)
4.  Results are drawn on the frame and printed to stdout.

Usage
-----
  # Process the whole video and save an annotated output
  python detect_grid.py input.mp4 --output annotated.mp4

  # Show live (no output file)
  python detect_grid.py input.mp4 --show

  # Single frame (for debugging)
  python detect_grid.py input.mp4 --frame 150 --show
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
import argparse
import sys
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────
# Configuration — tweak these if lighting conditions change
# ─────────────────────────────────────────────────────────
CFG = {
    # Adaptive threshold block size (odd number)
    "ADAPTIVE_BLOCK": 31,
    "ADAPTIVE_C": 10,

    # Minimum length (px) of a line segment kept by morphological open
    "H_LINE_MIN_LEN": 40,
    "V_LINE_MIN_LEN": 40,

    # Peak-finding in projection profiles
    "PEAK_HEIGHT": 0.2,        # fraction of max projection value

    # Cell classification (HSV)
    "EMPTY_MAX_SAT": 35,       # saturation below this → potentially empty
    "EMPTY_MIN_VAL": 100,      # brightness above this → empty

    # Visualisation colours (BGR)
    "COLOR_EMPTY": (0, 220, 0),
    "COLOR_FULL": (0, 0, 220),
    "COLOR_GRID": (0, 220, 220),
}

# ─────────────────────────────────────────────────────────
# Cube Detection
# ─────────────────────────────────────────────────────────
MODEL_PATH = r"best.pt"
CONFIDENCE = 0.738

def load_model(model_path):
    return YOLO(model_path)

def run_yolo(model, frame, confidence):

    results = model(
        frame,
        conf=confidence
    )

    return results

# ─────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────

def detect_grid_region(gray):
    """
    Locate the bounding box of the 3×3 grid using morphological
    H/V line detection followed by largest-contour selection.

    Returns (gx, gy, gw, gh) in the input image coordinate system,
    or None if no grid is found.
    """
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        CFG["ADAPTIVE_BLOCK"],
        CFG["ADAPTIVE_C"],
    )

    h_kern = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CFG["H_LINE_MIN_LEN"], 1))
    v_kern = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, CFG["V_LINE_MIN_LEN"]))

    h_lines = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, h_kern)
    v_lines = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, v_kern)
    combined = cv2.add(h_lines, v_lines)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5000:   # too small → not the grid
        return None, None

    gx, gy, gw, gh = cv2.boundingRect(largest)
    margin = 5
    gx = max(0, gx - margin)
    gy = max(0, gy - margin)
    gw = min(gray.shape[1] - gx, gw + 2 * margin)
    gh = min(gray.shape[0] - gy, gh + 2 * margin)

    return (gx, gy, gw, gh), combined


def find_grid_lines(line_img, gx, gy, gw, gh):
    """
    Given the morphological line image and the grid bounding box,
    return exactly 4 vertical and 4 horizontal line positions
    (relative to the crop origin).

    Returns (vlines, hlines) each a sorted list of 4 pixel positions,
    or (None, None) if the grid lines cannot be reliably found.
    """
    crop = line_img[gy:gy + gh, gx:gx + gw]

    vproj = crop.sum(axis=0).astype(float)
    hproj = crop.sum(axis=1).astype(float)
    vproj_n = vproj / (vproj.max() + 1e-9)
    hproj_n = hproj / (hproj.max() + 1e-9)

    vpeaks, _ = find_peaks(vproj_n,
                            height=CFG["PEAK_HEIGHT"],
                            distance=max(1, gw // 8))
    hpeaks, _ = find_peaks(hproj_n,
                            height=CFG["PEAK_HEIGHT"],
                            distance=max(1, gh // 8))

    def cluster(peaks):
        if len(peaks) == 0:
            return []
        peaks = sorted(peaks)
        clusters = [[peaks[0]]]
        for p in peaks[1:]:
            if p - clusters[-1][-1] < 20:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [int(np.mean(c)) for c in clusters]

    def pick4(lines, total):
        n = len(lines)
        if n >= 4:
            step = n // 3
            return [lines[0], lines[step], lines[2 * step], lines[-1]]
        return None

    vc = cluster(vpeaks)
    hc = cluster(hpeaks)
    vlines = pick4(vc, gw)
    hlines = pick4(hc, gh)

    return vlines, hlines


def detect_cube(frame, model):
    points = []
    results = run_yolo(
        model,
        frame,
        CONFIDENCE
    )

    boxes = results[0].boxes

    if boxes is None:
        return points

    for box in boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        points.append((cx, cy))

    return points


def process_frame(frame, model):
    """
    Main entry point: process one video frame.

    Parameters
    ----------
    frame : np.ndarray  — full side-by-side frame (RGB left, depth right)

    Returns
    -------
    annotated : np.ndarray  — visual result (RGB half only, annotated)
    grid_status : list[list[str]]  — 3×3 table of "EMPTY"/"FULL"/"UNKNOWN"
    grid_bbox : tuple or None      — (gx, gy, gw, gh) of detected grid
    """
    h, w = frame.shape[:2]
    rgb = frame[:, : w // 2].copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # ── 1. Locate grid ──────────────────────────────────────
    bbox, line_img = detect_grid_region(gray)
    if bbox is None:
        return rgb, None, None

    gx, gy, gw, gh = bbox

    # ── 2. Find the 4×4 grid-line positions ─────────────────
    vlines, hlines = find_grid_lines(line_img, gx, gy, gw, gh)
    if vlines is None or hlines is None:
        return rgb, None, bbox

    grid_crop_rgb = rgb[gy: gy + gh, gx: gx + gw]

    # ── 3. Classify each of the 9 cells ─────────────────────
    grid_status = []
    cell_info = []   # for drawing

    cube_points = detect_cube(rgb, model)
    for row in range(3):
        row_status = []
        for col in range(3):
            y1, y2 = hlines[row], hlines[row + 1]
            x1, x2 = vlines[col], vlines[col + 1]

            # Use inner 50% of cell to avoid border contamination
            pad_x = (x2 - x1) // 100
            pad_y = (y2 - y1) // 100
            cx1, cx2 = x1 + pad_x, x2 - pad_x
            cy1, cy2 = y1 + pad_y, y2 - pad_y

            status = "EMPTY"
            if len(cube_points) > 0:
                for cx, cy in cube_points:
                    if (gx + cx1 <= cx <= gx + cx2) and (gy + cy1 <= cy <= gy + cy2):
                        status = "FULL"
                        break

            row_status.append(status)
            cell_info.append((row, col, cx1, cy1, cx2, cy2, status, cube_points))

        grid_status.append(row_status)

    # ── 4. Draw annotations ──────────────────────────────────
    vis = rgb.copy()

    # Grid lines
    for x in vlines:
        cv2.line(vis, (gx + x, gy), (gx + x, gy + gh),
                 CFG["COLOR_GRID"], 2)
    for y in hlines:
        cv2.line(vis, (gx, gy + y), (gx + gw, gy + y),
                 CFG["COLOR_GRID"], 2)

    # Outer bounding box
    cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh),
                  CFG["COLOR_GRID"], 2)

    # Cell labels
    for row, col, cx1, cy1, cx2, cy2, status, cube_points in cell_info:
        color = CFG["COLOR_EMPTY"] if status == "EMPTY" else CFG["COLOR_FULL"]
        label = "E" if status == "EMPTY" else "F"

        # Shaded cell background
        overlay = vis.copy()
        cv2.rectangle(overlay,
                      (gx + cx1, gy + cy1),
                      (gx + cx2, gy + cy2), color, -1)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        # Border
        cv2.rectangle(vis,
                      (gx + cx1, gy + cy1),
                      (gx + cx2, gy + cy2), color, 2)

        # Text
        cx_abs = gx + (cx1 + cx2) // 2
        cy_abs = gy + (cy1 + cy2) // 2
        cv2.putText(vis, label,
                    (cx_abs - 10, cy_abs + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if cube_points:
            for cx, cy in cube_points:
                cv2.circle(
                    vis,
                    (cx, cy),
                    5,
                    (0, 255, 0),
                    -1
                )

    # Summary legend
    legend_y = 20
    for r, row_status in enumerate(grid_status):
        line_txt = f"Row {r+1}: " + "  ".join(
            f"[{s[0]}]" for s in row_status)
        cv2.putText(vis, line_txt, (10, legend_y + r * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return vis, grid_status, bbox


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect 3×3 grid cell occupancy in a RealSense side-by-side video.")
    parser.add_argument("input", help="Path to input MP4 (side-by-side RGB+depth)")
    parser.add_argument("--output", default=None,
                        help="Path for annotated output video (optional)")
    parser.add_argument("--show", action="store_true",
                        help="Display frames in a window (press Q to quit)")
    parser.add_argument("--frame", type=int, default=None,
                        help="Process only this single frame index (0-based)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every N-th frame (default 1 = every frame)")
    args = parser.parse_args()

    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.input}", file=sys.stderr)
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  // 2   # RGB half
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fw*2}×{fh}px, {fps:.1f} fps, {total} frames")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (fw, fh))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Single-frame mode
        if args.frame is not None and frame_idx != args.frame:
            frame_idx += 1
            continue

        if frame_idx % args.skip == 0:
            annotated, status, bbox = process_frame(frame, model)

            # ── stdout report ──
            if status is not None:
                print(f"\nFrame {frame_idx:4d} │ Grid @ {bbox}")
                for r, row in enumerate(status):
                    cells = "  ".join(f"[{s:5s}]" for s in row)
                    print(f"           Row {r+1}: {cells}")
            else:
                print(f"Frame {frame_idx:4d} │ grid not detected")

            if writer:
                writer.write(annotated)

            if args.show:
                cv2.imshow("Grid Occupancy", annotated)
                key = cv2.waitKey(1 if args.frame is None else 0)
                if key in (ord("q"), ord("Q"), 27):
                    break

        frame_idx += 1
        if args.frame is not None:
            break   # done

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()