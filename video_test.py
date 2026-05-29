import numpy as np
import cv2
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================

VIDEO_FILE = r"C:\Users\thapa\Documents\FIBOX\Box_Detection\abu vision data\realsense_video_20260425_114141.mp4"

MODEL_PATH = r"best.pt"

OUTPUT_VIDEO = "yolo_output_with_depth.mp4"

CONFIDENCE = 0.738

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(MODEL_PATH)

# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_FILE}")

fps = cap.get(cv2.CAP_PROP_FPS)

full_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Side-by-side:
# LEFT  = RGB
# RIGHT = depth colormap
rgb_width = full_width // 2

# ==========================================
# VIDEO WRITER
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (full_width, height)
)

print("Processing started...")
print("Press Q to quit.")

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    # Split side-by-side frame
    rgb_frame = frame[:, :rgb_width]
    depth_frame = frame[:, rgb_width:]

    # ======================================
    # YOLO INFERENCE
    # ======================================

    results = model(
        rgb_frame,
        conf=CONFIDENCE,
        verbose=False
    )

    annotated_rgb = results[0].plot()

    # ======================================
    # DRAW CENTER POINTS
    # ======================================

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.circle(
                annotated_rgb,
                (cx, cy),
                5,
                (0, 255, 0),
                -1
            )

    # ======================================
    # COMBINE RGB + DEPTH
    # ======================================

    combined = np.hstack((
        annotated_rgb,
        depth_frame
    ))

    # Save
    writer.write(combined)

    # Display
    cv2.imshow(
        "YOLO + Depth",
        combined
    )

    # Quit
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()

writer.release()

cv2.destroyAllWindows()

print(f"Saved output: {OUTPUT_VIDEO}")