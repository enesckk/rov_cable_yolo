import cv2
import numpy as np
import serial
import time
import queue
import threading
from ultralytics import YOLO

# === Ayarlar ===
DEBUG = True
UART_PORT = '/dev/ttyTHS1'
UART_BAUDRATE = 9600
ANOMALY_CLASS_IDS = [1]  # YOLO modelindeki anomaly class ID'leri
IMG_SIZE = 416           # YOLO input boyutu (küçük tutarsan FPS artar)

uart_queue = queue.Queue()

# === UART Gönderici Thread ===
def uart_sender():
    with serial.Serial(UART_PORT, UART_BAUDRATE) as ser:
        time.sleep(2)
        while True:
            try:
                command = uart_queue.get(timeout=1)
                ser.write((command + "\n").encode())
                uart_queue.task_done()
            except queue.Empty:
                pass
            time.sleep(0.001)

threading.Thread(target=uart_sender, daemon=True).start()

# === CUDA kontrolü ===
if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    raise SystemError("❌ CUDA destekli GPU bulunamadı.")

# === YOLOv8/YOLOv12 Model Yükle ===
model = YOLO('bestyolov12anomali+kablo.pt')  # YOLOv8 ya da YOLOv12 modeli fark etmez

# === Kamera Başlat ===
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# === Değişkenler ===
cx_list = []
son_komut = "F300"
anomaly_counter = 0

# === Ana Döngü ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === CUDA işlemi ===
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    cpu_frame = gpu_frame.download()

    results = model.predict(cpu_frame, imgsz=IMG_SIZE, conf=0.4, verbose=False)

    command = son_komut
    position_text = "Kablo tespit edilemedi"
    kablo_var = False
    center = cpu_frame.shape[1] // 2
    margin = 40

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        # === Kablolar ve anomalileri ayır ===
        kablo_boxes = [b for b in r.boxes if int(b.cls[0]) == 0]
        anomaly_boxes = [b for b in r.boxes if int(b.cls[0]) in ANOMALY_CLASS_IDS]

        # === Kablo yön tespiti ===
        if kablo_boxes:
            kablo_var = True
            largest_box = max(kablo_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
            cx = (x1 + x2) // 2
            cx_list.append(cx)
            if len(cx_list) > 5:
                cx_list.pop(0)
            cx_avg = int(np.mean(cx_list))

            if cx_avg < center - margin:
                command = "L300"
                position_text = "Kablo solda"
            elif cx_avg > center + margin:
                command = "R300"
                position_text = "Kablo sağda"
            else:
                command = "F300"
                position_text = "Kablo ortada"

            son_komut = command

            if DEBUG:
                cv2.rectangle(cpu_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cpu_frame, "Kablo", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # === Anomali tespiti ===
        for b in anomaly_boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            anomaly_crop = cpu_frame[y1:y2, x1:x2]
            anomaly_counter += 1
            filename = f"anomaly_{anomaly_counter}.jpg"
            cv2.imwrite(filename, anomaly_crop)
            if DEBUG:
                cv2.rectangle(cpu_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(cpu_frame, "Anomali", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    if not kablo_var:
        command = "Q300"  # Kablo yoksa 90° sağa dön

    uart_queue.put(command)

    if DEBUG:
        cv2.putText(cpu_frame, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Anomali + Kablo Takibi (CUDA)", cpu_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
uart_queue.join()
