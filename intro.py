import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=4)
    mask = cv2.GaussianBlur(mask, (5,5), 100)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:
            hull = cv2.convexHull(max_contour)
            cv2.drawContours(roi, [max_contour], -1, (255,0,0), 2)
            cv2.drawContours(roi, [hull], -1, (0,0,255), 2)
            hull_indices = cv2.convexHull(max_contour, returnPoints=False)
            if hull_indices is not None and len(hull_indices) > 3:
                defects = cv2.convexityDefects(max_contour, hull_indices)
                if defects is not None:
                    count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])
                        a = np.linalg.norm(np.array(start) - np.array(end))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(far) - np.array(end))
                        if b * c == 0:
                            continue
                        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle) * 57
                        if angle <= 90:
                            count += 1
                            cv2.circle(roi, far, 5, (0, 255, 0), -1)
                    text = f"Fingers: {count + 1}"
                    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()