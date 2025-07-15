import cv2
import numpy as np

# Kamera başlat
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# CUDA kullanılabilir mi kontrolü
if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    raise SystemError("❌ CUDA destekli GPU bulunamadı. Jetson CUDA desteğini kontrol et.")

print("✅ CUDA destekli kablo takibi başlatıldı.")

# Değişkenler
cx_list = []
son_yon = "FORWARD"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === GPU'ya gönder ===
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # === Gürültü azaltma ===
    gpu_blurred = cv2.cuda.GaussianBlur(gpu_frame, (5, 5), 0)

    # === HSV dönüşüm ===
    gpu_hsv = cv2.cuda.cvtColor(gpu_blurred, cv2.COLOR_BGR2HSV)

    # === Siyah için maskeleme (HSV) ===
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 50], dtype=np.uint8)
    gpu_mask = cv2.cuda.inRange(gpu_hsv, lower_black, upper_black)

    # === CPU'ya indir (CUDA'dan normal OpenCV'ye geçiş) ===
    mask = gpu_mask.download()

    # === ROI: alt yarı ===
    roi = mask[mask.shape[0] // 2:, :]

    # === Gürültü temizleme (CPU tarafında) ===
    kernel = np.ones((5, 5), np.uint8)
    roi = cv2.erode(roi, kernel, iterations=1)
    roi = cv2.dilate(roi, kernel, iterations=2)

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    command = son_yon
    position_text = "Kablo yok - Son yöne göre devam"

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 300]
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cx_list.append(cx)
                if len(cx_list) > 5:
                    cx_list.pop(0)
                cx_avg = int(np.mean(cx_list))

                # Yön açısı
                [vx, vy, x, y] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = np.degrees(np.arctan2(vy, vx))

                center = frame.shape[1] // 2
                margin = 40

                if cx_avg < center - margin:
                    command = "STRAFE_LEFT"
                    position_text = f"Kablo solda ({angle:.1f}°)"
                elif cx_avg > center + margin:
                    command = "STRAFE_RIGHT"
                    position_text = f"Kablo sağda ({angle:.1f}°)"
                else:
                    command = "FORWARD"
                    position_text = f"Kablo ortada ({angle:.1f}°)"

                son_yon = command

                # Görselleştirme (CPU tarafında)
                cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy + mask.shape[0] // 2), 5, (0, 0, 255), -1)
                cv2.line(frame, (center, 0), (center, frame.shape[0]), (255, 255, 255), 1)

    print("Komut:", command)

    # Ekrana yaz
    cv2.putText(frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Kablo Takibi - Xavier CUDA", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
