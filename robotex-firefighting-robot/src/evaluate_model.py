import cv2
import pandas as pd
import time
import os
from ultralytics import YOLO

# =====================================
# SETTINGS
# =====================================

MODEL_PATH = r"models/final_v2.pt"

CAMERA = 0       # USB Webcam

CONF_THRESHOLD = 0.55


OUTPUT_DIR = "evaluation"

LOW_CONF_DIR = os.path.join(OUTPUT_DIR, "low_confidence")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOW_CONF_DIR, exist_ok=True)

# =====================================
# LOAD MODEL
# =====================================

print("Loading model...")

model = YOLO(MODEL_PATH)

print("✅ Model Loaded!")

# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(CAMERA)

time.sleep(2)

if not cap.isOpened():

    print("❌ Could not open USB webcam")

    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera Resolution : {width} x {height}")

video_writer = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, "annotated_video.mp4"),
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (width, height)
)

# =====================================
# VARIABLES
# =====================================

rows = []
fps_list = []
frame_no = 0

print("\nPress Q or ESC to stop\n")

# =====================================
# LOOP
# =====================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Couldn't read frame")
            break

        start = time.time()

        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        fps = 1/(time.time()-start)

        fps_list.append(fps)

        annotated = frame.copy()

        detections = 0

        for box in results[0].boxes:

            detections += 1

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            label = model.names[cls]

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            rows.append({

                "Frame":frame_no,

                "Object":label,

                "Confidence":conf,

                "CenterX":cx,

                "CenterY":cy,

                "Width":x2-x1,

                "Height":y2-y1,

                "FPS":fps

            })

            color = (0,255,0)

            cv2.rectangle(
                annotated,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if conf < 0.60:

                cv2.imwrite(

                    os.path.join(

                        LOW_CONF_DIR,

                        f"{frame_no}_{label}.jpg"

                    ),

                    annotated

                )

        cv2.putText(
            annotated,
            f"FPS : {fps:.1f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            annotated,
            f"Detections : {detections}",
            (10,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        video_writer.write(annotated)

        cv2.imshow("Robot Vision", annotated)

        frame_no += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

except KeyboardInterrupt:

    print("Interrupted")

finally:

    cap.release()

    video_writer.release()

    cv2.destroyAllWindows()

# =====================================
# SAVE RESULTS
# =====================================

df = pd.DataFrame(rows)

df.to_csv(
    os.path.join(OUTPUT_DIR, "detections.csv"),
    index=False
)

with open(
    os.path.join(OUTPUT_DIR, "summary.txt"),
    "w"
) as f:

    f.write(f"Frames Processed : {frame_no}\n")

    if fps_list:
        f.write(f"Average FPS : {sum(fps_list)/len(fps_list):.2f}\n")
    else:
        f.write("Average FPS : 0\n")

    f.write(f"Total Detections : {len(df)}\n\n")

    if not df.empty:
        f.write(str(df.groupby("Object")["Confidence"].mean()))

print("\n✅ Evaluation Complete!")
print("Results saved in:", OUTPUT_DIR)