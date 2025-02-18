import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna dalam HSV
    colors = {
        "Merah": [(0, 120, 70), (10, 255, 255), (0, 0, 255)],
        "Hijau": [(40, 40, 40), (80, 255, 255), (0, 255, 0)],
        "Biru": [(100, 150, 0), (140, 255, 255), (255, 0, 0)]
    }

    for color_name, (lower, upper, bgr) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filter area kecil agar tidak menangkap noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

    cv2.imshow("Deteksi Warna", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
