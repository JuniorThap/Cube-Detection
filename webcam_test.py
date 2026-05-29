import numpy as np
import cv2
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = r"best.pt"

OUTPUT_VIDEO = "webcam_yolo_output.mp4"

CONFIDENCE = 0.738

FPS = 30

# ==========================================
# LOAD MODEL
# ==========================================

def load_model(model_path):
    return YOLO(model_path)


# ==========================================
# INITIALIZE PC WEBCAM
# ==========================================

def initialize_webcam():

    cap = cv2.VideoCapture(0)

    # Optional settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    return cap


# ==========================================
# GET FRAME SIZE
# ==========================================

def get_frame_info(cap):

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height


# ==========================================
# CREATE VIDEO WRITER
# ==========================================

def create_video_writer(output_path, width, height, fps):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    return writer


# ==========================================
# RUN YOLO INFERENCE
# ==========================================

def run_yolo(model, frame, confidence):

    results = model(
        frame,
        conf=confidence
    )

    return results


# ==========================================
# MAIN LOOP
# ==========================================

def process_webcam():

    # Load YOLO model
    model = load_model(MODEL_PATH)

    # Open webcam
    cap = initialize_webcam()

    # Get frame size
    width, height = get_frame_info(cap)

    # Create video writer
    video_writer = create_video_writer(
        OUTPUT_VIDEO,
        width,
        height,
        FPS
    )

    print("Webcam started...")
    print("Press 'q' to quit.")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # YOLO inference
            results = run_yolo(
                model,
                frame,
                CONFIDENCE
            )

            # Draw YOLO results
            annotated_frame = results[0].plot()

            # Save video
            video_writer.write(annotated_frame)

            # Display
            cv2.imshow(
                "YOLO Webcam Detection",
                annotated_frame
            )

            # Quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        cap.release()

        video_writer.release()

        cv2.destroyAllWindows()

        print(f"Saved output video: {OUTPUT_VIDEO}")


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    process_webcam()