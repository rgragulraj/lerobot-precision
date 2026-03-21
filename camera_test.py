import cv2

cap = cv2.VideoCapture(7)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("gripper_test.jpg", frame)
        print(f"Frame captured: {frame.shape}")
    else:
        print("ERROR: Cannot read frame")
    cap.release()

