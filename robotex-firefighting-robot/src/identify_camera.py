import cv2

for i in [0, 1]:

    cap = cv2.VideoCapture(i)

    print(f"\nOpening Camera {i}")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.putText(
            frame,
            f"CAMERA {i}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow(f"Camera {i}", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            break

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()

print("Finished")