import cv2
import numpy as np
from ultralytics import YOLO

# === Ayarlar ===
DEBUG = True
IMG_SIZE = 416
ANOMALY_CLASS_IDS = [1]  # Anomali sınıf ID'leri (modeldeki ID'yi kontrol et)
anomaly_counter = 0

# === CUDA kontrolü ===
if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    raise SystemError("❌ CUDA destekli GPU bulunamadı.")

# === YOLOv8/YOLOv12 Model Yükle ===
model = YOLO("bestyolov12anomali+kablo.pt")

# === Kamera Başlat ===
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# === Ana Döngü ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === CUDA: GPU'ya yükle, CPU'ya indir ===
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    cpu_frame = gpu_frame.download()

    # === YOLO Tespiti ===
    results = model.predict(cpu_frame, imgsz=IMG_SIZE, conf=0.4, verbose=False)

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for b in r.boxes:
            class_id = int(b.cls[0])
            if class_id not in ANOMALY_CLASS_IDS:
                continue

            # Koordinatları al
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            label = model.names[class_id] if hasattr(model, 'names') else f"Class {class_id}"

            # Görüntüyü kırp ve kaydet
            anomaly_crop = cpu_frame[y1:y2, x1:x2]
            anomaly_counter += 1
            filename = f"anomaly_{anomaly_counter}.jpg"
            cv2.imwrite(filename, anomaly_crop)

            # Görselleştir
            if DEBUG:
                cv2.rectangle(cpu_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(cpu_frame, f"Anomali: {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if DEBUG:
        cv2.imshow("Anomali Tespiti (CUDA)", cpu_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Temizlik ===
cap.release()
cv2.destroyAllWindows()
