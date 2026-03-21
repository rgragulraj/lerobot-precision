import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(
            f"index {i}: {'OK' if ret else 'opened but no frame'} - "
            f"{frame.shape if ret else ''}"
        )
        cap.release()
    else:
        print(f"index {i}: not available")
