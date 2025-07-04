from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt") 

# Define cat class id
cat_class_id = 15 

# For track obj
track_point = []

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)

    for box in boxes:
        class_id = int(box.cls[0])
        if class_id != cat_class_id:
            continue
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Blue box and label
        label = f"Cat"
        annotator.box_label(xyxy, label=label, color=(255, 0, 0))  

        # Center point cat
        track_point.append((cx, cy))

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model(frame)[0]

    frame = draw_boxes(frame, results.boxes)

    # Tracking line
    for i in range(1, len(track_point)):
        cv2.line(frame, track_point[i - 1], track_point[i], (255, 0, 0), 2)

    # Name on top right
    cv2.putText(frame, "Pubodee - Clicknext-Internship-2024", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture("CatZoomies.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame
        frame = cv2.resize(frame, (640, 360))
        # Detect and display
        frame = detect_object(frame)
        cv2.imshow("Cat Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
