import cv2
import numpy as np
from ultralytics import YOLO

# === CUDA kontrolü ===
if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    raise SystemError("❌ CUDA destekli GPU bulunamadı.")

# === YOLO modeli ===
model = YOLO('best.pt')  # Eğittiğin kablo takip modeli

# === Kamera başlat ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cx_list = []
son_yon = "FORWARD"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === CUDA ile frame'i yükle ===
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # === (İsteğe bağlı) bulanıklaştırma ===
    gpu_blurred = cv2.cuda.GaussianBlur(gpu_frame, (3, 3), 0)

    # === YOLO ile tespit (CPU’da çalışır ama input CUDA'dan alınabilir) ===
    # Yol: CUDA'dan alınan frame'i CPU için indir
    cpu_frame = gpu_blurred.download()

    results = model.predict(cpu_frame, imgsz=640, conf=0.4)

    command = son_yon
    position_text = "Kablo tespit edilemedi"

    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            # En büyük kutuyu al
            largest_box = max(r.boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cx_list.append(cx)
            if len(cx_list) > 5:
                cx_list.pop(0)
            cx_avg = int(np.mean(cx_list))

            # Görüntü merkezine göre yön analizi
            center = cpu_frame.shape[1] // 2
            margin = 40

            if cx_avg < center - margin:
                command = "STRAFE_LEFT"
                position_text = "Kablo solda"
            elif cx_avg > center + margin:
                command = "STRAFE_RIGHT"
                position_text = "Kablo sağda"
            else:
                command = "FORWARD"
                position_text = "Kablo ortada"

            son_yon = command

            # Görsel işaretleme
            cv2.rectangle(cpu_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(cpu_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(cpu_frame, (center, 0), (center, cpu_frame.shape[0]), (255, 255, 255), 1)

    print("Komut:", command)

    # Metin ve gösterim
    cv2.putText(cpu_frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("YOLO Kablo Takibi - CUDA", cpu_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
