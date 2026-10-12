import cv2

for backend_name, backend in [
    ("Default", None),
    ("DirectShow", cv2.CAP_DSHOW),
    ("MediaFoundation", cv2.CAP_MSMF),
]:

    print("\n==============================")
    print(backend_name)

    for i in range(3):

        if backend is None:
            cap = cv2.VideoCapture(i)
        else:
            cap = cv2.VideoCapture(i, backend)

        print(f"Camera {i}:", cap.isOpened())

        if cap.isOpened():
            ret, frame = cap.read()
            print("Read Frame:", ret)

        cap.release()