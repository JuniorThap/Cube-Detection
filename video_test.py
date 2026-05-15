import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================

BAG_FILE = r"C:\Users\thapa\Documents\FIBOX\Box_Detection\abu vision data\realsense_data_20260425_123816-002.bag"


MODEL_PATH = r"best.pt"

OUTPUT_VIDEO = "yolo_output.mp4"

CONFIDENCE = 0.738

# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO(MODEL_PATH)

# ==========================================
# REALSENSE PIPELINE
# ==========================================

pipeline = rs.pipeline()
config = rs.config()

rs.config.enable_device_from_file(
    config,
    BAG_FILE,
    repeat_playback=False
)

profile = pipeline.start(config)

# Disable realtime playback
playback = profile.get_device().as_playback()
playback.set_real_time(False)

# ==========================================
# GET VIDEO INFO
# ==========================================

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

frame = np.asanyarray(
    color_frame.get_data()
)

height, width = frame.shape[:2]

# ==========================================
# VIDEO WRITER
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    30,                 # FPS
    (width, height)
)

print("Recording started...")
print("Press 'q' to quit.")

# ==========================================
# INFERENCE LOOP
# ==========================================

try:

    while True:

        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(
            color_frame.get_data()
        )

        # ==========================================
        # YOLO INFERENCE
        # ==========================================

        results = model(
            frame,
            conf=CONFIDENCE
        )

        # Draw detections
        annotated_frame = results[0].plot()

        # ==========================================
        # SAVE VIDEO FRAME
        # ==========================================

        video_writer.write(annotated_frame)

        # ==========================================
        # DISPLAY
        # ==========================================

        cv2.imshow(
            "YOLO .bag Inference",
            annotated_frame
        )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# End of bag file
except RuntimeError:
    print("Reached end of .bag file.")

finally:

    pipeline.stop()

    video_writer.release()

    cv2.destroyAllWindows()

    print(f"Saved output video: {OUTPUT_VIDEO}")