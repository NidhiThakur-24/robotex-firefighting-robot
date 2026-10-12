import cv2
import time
from ultralytics import YOLO

# =====================================
# Load Model
# =====================================

print("Loading model...")
model = YOLO(r"models/competition_final.pt")
print("✅ Model Loaded!")

# =====================================
# Open Webcam
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Webcam Started!")

# =====================================
# Real-Time Detection
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Couldn't read frame")
        break

    start = time.time()

    results = model.predict(
        source=frame,
        conf=0.50,      # Try 0.60 later if false detections occur
        verbose=False
    )

    fps = 1 / (time.time() - start)

    annotated = frame.copy()

    # Draw detections
    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        label = model.names[cls]

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # FPS
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show Window
    cv2.imshow("Robot Vision", annotated)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================================
# Cleanup
# =====================================

cap.release()
cv2.destroyAllWindows()