import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(r"C:\Users\shubi\Desktop\lr_ra\Real-time\WhatsApp Video 2023-08-06 at 01.17.25.mp4")
cap.set (3,1280)
cap.set (4, 720)

while True:
    ret, frame =cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    annotations__frame = results[0].plot()
    cv2.imshow("Real_Time Object detection", annotations__frame)

    # q press = exit           
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()