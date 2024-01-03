from ultralytics import YOLO
import cv2

model = YOLO('yolov8s_parking.pt')
img = cv2.imread("PARKING/P2.jpg")
results = model(img, verbose=False)
count = 0
dimensions = []
for r in results:
        for c in r.boxes:
            if model.names[int(c.cls)] == "empty" : 
                count+=1
            x1, y1, x2, y2 = map(int, c.xyxy[0])
            dimensions.append([x1, y1, x2, y2,int(c.cls)])
    
for dimension in dimensions:
    x, y, w, h,cl = dimension
    if cl==1:
        cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)
    else:
        cv2.rectangle(img, (x, y), (w, h), ( 0, 255, 0), 2)

cv2.putText(img,f"Total empty places : {count}",(0,25),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)

cv2.imshow('FRAME', img)
cv2.waitKey(0)
